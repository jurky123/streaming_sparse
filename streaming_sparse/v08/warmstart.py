"""Fresh K-only P8 calibration on original frozen-backbone QKV; no old S8 migration."""
import argparse
import json
import math
import torch
from .common import HERE,protocol,paired_seed,source_identity,atomic_json,save,output,tensor_digest,require_audit,nmse,pair_index
from .router import InformationRouter
from .attention import attend,block_logz


def datasets():
    from ..v04.data import SnapshotData,BalancedSchedule
    root=protocol()['warmstart']['legacy_qkv_root']
    return SnapshotData(root,'train'),SnapshotData(root,'val'),BalancedSchedule


@torch.no_grad()
def normalize(router,train,device):
    sketch=router.sketch
    if sketch.normalization_locked:raise ValueError('Train-only initial normalizers are immutable')
    cal=[];hard=[]
    for pair in sorted(train.strata):
        events=train.load(sorted(train.strata[pair])[0],device);e=events[0]
        index=pair_index(*pair);features=[]
        state=router.build(e['k_hist'],e['v_hist'],index)
        for e in events:
            q=e['q'][:16];raw,_,f,_=sketch.raw(q,state);features.append(f.flatten(0,1))
            true=block_logz(q,e['k_hist']);cal.append(torch.nn.functional.huber_loss(raw,true))
            route=router(q,e['k_current'],e['v_current'],state,step=e['step_id'],mass_only=True)
            sparse=attend(q,e['k_hist'],e['v_hist'],e['k_current'],e['v_current'],route['hard'])
            dense=attend(q,e['k_hist'],e['v_hist'],e['k_current'],e['v_current'],torch.ones_like(route['hard']))
            hard.append(nmse(sparse,dense))
        f=torch.cat(features).float()
        sketch.feature_mean[index].copy_(f.mean(0));sketch.feature_std[index].copy_(f.std(0,unbiased=False).clamp_min(1e-3))
        sketch.feature_count[index]=len(f)
    sketch.cal_scale.copy_(torch.stack(cal).mean().clamp_min(1e-8));sketch.normalization_locked.fill_(True)
    return dict(calibration_initial_scale=float(sketch.cal_scale),hard_initial_scale=float(torch.stack(hard).mean().clamp_min(1e-8)),
                split='train',feature_count=sketch.feature_count.cpu().tolist())


@torch.no_grad()
def validate(router,ds,device,scales):
    from ..v06.common import monitor_versions
    cal=[];errors=[];bias=[];mass_bias=[];saturation=[]
    for version in monitor_versions(ds):
        events=ds.load(version,device);e=events[0];pair=pair_index(e['layer_id'],e['head_id'])
        state=router.build(e['k_hist'],e['v_hist'],pair)
        for e in events:
            for start in range(0,len(e['q']),16):
                q=e['q'][start:start+16]
                r=router(q,e['k_current'],e['v_current'],state,step=e['step_id'],mass_only=True)
                est=r['estimate'];truth=block_logz(q,e['k_hist'])
                cal.append(float(router.sketch.calibration_loss(q,e['k_hist'],e['k_current'],state,est)))
                bias.append(float((est['logz']-truth).mean()))
                lc=est['current_logz'];true_current=torch.cat((truth,lc[:,None]),-1).softmax(-1)[:,-1]
                mass_bias.append(float((est['current_mass']-true_current).mean()))
                target=attend(q,e['k_hist'],e['v_hist'],e['k_current'],e['v_current'],torch.ones_like(r['hard']))
                pred=attend(q,e['k_hist'],e['v_hist'],e['k_current'],e['v_current'],r['hard'])
                errors.append(float(nmse(pred,target)))
                saturation.append(float(((r['soft']<1e-4)|(r['soft']>1-1e-4)).float().mean()))
    avg=lambda values:sum(values)/len(values)
    return dict(selection_score=avg(cal)+avg(errors)/scales['hard_initial_scale'],
                normalized_cal_huber=avg(cal),mean_tile_hard_nmse=avg(errors),logz_bias=avg(bias),
                current_mass_bias=avg(mass_bias),gate_saturated_fraction=avg(saturation),
                split='legacy_val',snapshots=36,tiles=len(errors))


def run(seed,device='cuda',resume=None):
    require_audit();paired_seed(seed);p=protocol();settings=p['warmstart']
    train,val,Schedule=datasets();router=InformationRouter('K').to(device)
    directory=output(HERE/f'runs/seed{seed}/calibration')
    if any(directory.iterdir()) and resume is None:raise FileExistsError(directory)
    optimizer=torch.optim.AdamW(router.sketch.parameters(),lr=settings['sketch_lr'],weight_decay=.01)
    schedule=Schedule(sorted(train.groups),seed)
    best=float('inf');best_step=0;curve=[];begin=0
    if resume:
        saved=torch.load(resume,weights_only=True,map_location=device)
        if saved['source_sha256']!=source_identity():raise ValueError('Warm-start source mismatch')
        router.load_state_dict(saved['router']);optimizer.load_state_dict(saved['optimizer']);schedule.load_state_dict(saved['sampler'])
        scales=saved['scales'];best=saved['best'];best_step=saved['best_step'];curve=saved['curve'];begin=saved['updates']
    else:scales=normalize(router,train,device)
    def snapshot(step):
        return dict(kind='calibration',information='K',seed=seed,router=router.state_dict(),scales=scales,
            source_sha256=source_identity(),teacher_sha256=p['teacher_sha256'],updates=step,best=best,best_step=best_step,curve=curve,
            optimizer=optimizer.state_dict(),sampler=schedule.state_dict(),train_fingerprint=train.fingerprint(),
            val_fingerprint=val.fingerprint(),sketch_digest=tensor_digest(router.sketch.state_dict().items()))
    def evaluation(step):
        nonlocal best,best_step
        result=validate(router,val,device,scales);curve.append(dict(updates=step,**result))
        if result['selection_score']<best:
            best=result['selection_score'];best_step=step;save(directory/'best.pt',snapshot(step))
        atomic_json(directory/'validation_curve.json',curve)
        save(directory/'latest.pt',snapshot(step))
        atomic_json(directory/'status.json',dict(stage='calibration',updates=step,max_updates=settings['sketch_updates'],best_step=best_step,best=best))
        print(json.dumps(curve[-1]),flush=True)
    if begin==0:save(directory/'initial.pt',snapshot(0));evaluation(0)
    for step in range(begin+1,settings['sketch_updates']+1):
        optimizer.zero_grad(set_to_none=True);losses=[]
        for version,tile in schedule.sample(4):
            events=train.load(version,device);first=events[0]
            state=router.build(first['k_hist'],first['v_hist'],pair_index(first['layer_id'],first['head_id']))
            for e in events:
                q=e['q'][tile*16:(tile+1)*16]
                route=router(q,e['k_current'],e['v_current'],state,step=e['step_id'],mass_only=True)
                with torch.no_grad():
                    target=attend(q,e['k_hist'],e['v_hist'],e['k_current'],e['v_current'],torch.ones_like(route['mask']))
                pred=attend(q,e['k_hist'],e['v_hist'],e['k_current'],e['v_current'],route['mask'])
                losses.append(router.sketch.calibration_loss(q,e['k_hist'],e['k_current'],state,route['estimate'])+.1*nmse(pred,target))
        loss=torch.stack(losses).mean()
        if not torch.isfinite(loss):raise FloatingPointError('Calibration objective')
        loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(router.sketch.parameters(),1.,error_if_nonfinite=True));optimizer.step()
        with (directory/'metrics.jsonl').open('a') as f:f.write(json.dumps(dict(updates=step,loss=float(loss),gradient_norm=norm))+'\n')
        if step%settings['validate_every']==0:evaluation(step)
    save(directory/'last.pt',snapshot(settings['sketch_updates']))
    atomic_json(directory/'coverage_final.json',schedule.state_dict()['visits'])
    atomic_json(directory/'completed.json',dict(source_sha256=source_identity(),best_step=best_step,updates=settings['sketch_updates']))


def main():
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,choices=[0,1,2],required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume')
    a=p.parse_args();run(a.seed,a.device,a.resume)


if __name__=='__main__':main()
