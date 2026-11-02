"""Frozen high90 source rollouts, streaming on-policy inputs, case-atomic training."""
import argparse,gc,json,os,signal,time
from pathlib import Path
import torch
from .broad_common import Engine,cases,ROOT as DATA
from .high_sparse_runtime import HighSparseRuntime
from .rollout_suite import generate_chunk
from .high_sparse_learn import Layer,event,route_score,pick,swap_features,gain_targets,evaluate,mixed,nmse
from .learned_cache.io import atomic,digest,RunLock,disk_guard,checkpoint,restore
ROOT=Path('artifacts/high_sparse_train_20261030');LAYERS=list(range(1,30))
class Observer:
    def __init__(self,rt,base,callback):self.rt=rt;self.base=base;self.callback=callback
    def __getitem__(self,grid):
        def call(*args,**kwargs):
            c=self.rt.engine.current
            if c['chunk'] in [2,5,8] and c['step'] in [0,3]:self.callback(self.rt,event(self.rt,args[0],args[1],args[2],args[5]))
            self.base[grid](*args,**kwargs)
        return call

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--epochs',type=int,default=3);args=ap.parse_args()
    lock=RunLock(ROOT);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    try:
        torch.set_num_threads(4);torch.manual_seed(1771);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        sources=['high_sparse_train.py','high_sparse_learn.py','high_sparse_runtime.py','compensation_stable_kernels.py','compensation_ordered_kernels.py','broad_common.py']
        identity=dict(code={x:digest(Path('local_experiments')/x) for x in sources},initial=digest('artifacts/compensation_probe_20261030/model.pt'),train=cases('train'),val=cases('val'),epochs=args.epochs,source='frozen mean_var global90 comp closed-loop; no dense-video replay; student does not alter source trajectories',sample=dict(chunks=[2,5,8],steps=[0,3],tiles=[0,23,47,71],offsets=[0,21,42,63],layers=LAYERS))
        if (ROOT/'identity.json').exists():assert json.loads((ROOT/'identity.json').read_text())==identity
        else:atomic(ROOT/'identity.json',identity)
        old=torch.load('artifacts/compensation_probe_20261030/model.pt',weights_only=True)['models']
        models={l:Layer(old[l]).cuda() for l in LAYERS}
        opts={l:dict(value=torch.optim.Adam([dict(params=[models[l].basis,models[l].center],lr=.0003),dict(params=[models[l].uk,models[l].uv],lr=.001)]),route=torch.optim.Adam([models[l].route],lr=.01),swap=torch.optim.Adam([models[l].swap],lr=.01)) for l in LAYERS}
        saved,_=restore(ROOT);done=[]
        if saved is not None:
            assert saved['identity']==identity
            for l in LAYERS:
                models[l].load_state_dict(saved['models'][l])
                for k,opt in opts[l].items():opt.load_state_dict(saved['optimizers'][l][k])
            done=saved['done']
        engine=Engine()
        def state():return dict(identity=identity,done=done,models={l:{k:v.detach().cpu() for k,v in m.state_dict().items()} for l,m in models.items()},optimizers={l:{k:v.state_dict() for k,v in opts[l].items()} for l in LAYERS})
        for epoch in range(args.epochs):
            order=cases('train');shift=epoch%len(order);order=order[shift:]+order[:shift]
            for case in order:
                key=f'e{epoch}_{case["id"]}'
                if key in done:continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted',done=len(done)));return
                disk_guard(ROOT,50,1)
                atomic(ROOT/'status.json',dict(status='training',epoch=epoch,case=case['id'],done=len(done),expected=args.epochs*8,pid=os.getpid()))
                emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();rt=HighSparseRuntime(engine,'global',.9,True);stats=[]
                def train(rt,e):
                    l=rt.active_layer;m=models[l];o=opts[l]
                    with torch.enable_grad():
                        # Pure mass objective: no teacher tensors are used at inference.
                        score=route_score(e['mass'],e['feat'],m.route);target=e['truth'].mean(2)
                        lossR=(target*(target.clamp_min(1e-12).log()-score.log_softmax(-1))).sum(-1).mean()
                        o['route'].zero_grad();lossR.backward();torch.nn.utils.clip_grad_norm_([m.route],1.,error_if_nonfinite=True);o['route'].step()
                        code,av=m.value(e['k'],e['v']);y,_,_=mixed(e,av,e['source_mask'])
                        lossV=nmse(y,e)+.0001*(m.uk.square().mean()+m.uv.square().mean())
                        o['value'].zero_grad();lossV.backward();torch.nn.utils.clip_grad_norm_([m.basis,m.center,m.uk,m.uv],1.,error_if_nonfinite=True);o['value'].step()
                        with torch.no_grad():
                            code,av=m.value(e['k'],e['v']);mask=pick(route_score(e['mass'],e['feat'],m.route),e['keep'],e['own'])
                            targetG=gain_targets(e,av,mask);feat=swap_features(e['feat'],e['mass'],code,mask)
                        pred=torch.einsum('htpf,hf->htp',feat,m.swap)
                        losses=torch.nn.functional.smooth_l1_loss(pred,targetG,reduction='none')
                        lossS=(losses*mask).sum()/mask.sum()+(losses*(~mask)).sum()/(~mask).sum()
                        o['swap'].zero_grad();lossS.backward();torch.nn.utils.clip_grad_norm_([m.swap],1.,error_if_nonfinite=True);o['swap'].step();m.project()
                    stats.append(dict(layer=l,chunk=rt.engine.current['chunk'],step=rt.engine.current['step'],value=float(lossV.detach()),route=float(lossR.detach()),swap=float(lossS.detach())))
                rt.kernel=Observer(rt,rt.kernel,train)
                try:
                    for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(stats)==174,len(stats)
                done.append(key);checkpoint(ROOT,state());atomic(ROOT/'training'/f'{key}.json',dict(rows=stats,mean={x:sum(r[x] for r in stats)/len(stats) for x in ['value','route','swap']}))
                print(json.dumps(dict(epoch=epoch,case=case['id'],done=len(done),mean={x:sum(r[x] for r in stats)/len(stats) for x in ['value','route','swap']})),flush=True)
            modelpath=ROOT/f'epoch{epoch}.pt'
            if not modelpath.exists():atomic(modelpath,dict(models=state()['models'],epoch=epoch,identity=identity),True)
            # Validation is read-only and uses four prompts excluded from fitting.
            for case in cases('val'):
                path=ROOT/'validation'/f'e{epoch}_{case["id"]}.json'
                if path.exists():continue
                if stop[0]:atomic(ROOT/'status.json',dict(status='interrupted_validation',epoch=epoch,case=case['id']));return
                atomic(ROOT/'status.json',dict(status='validation',epoch=epoch,case=case['id'],done=len(done),pid=os.getpid()))
                emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();rt=HighSparseRuntime(engine,'global',.9,True);rows=[]
                def validation(rt,e):
                    l=rt.active_layer;row=evaluate(models[l],e,rt.basis[l],rt.centers[l]);row.update(layer=l,chunk=rt.engine.current['chunk'],step=rt.engine.current['step']);rows.append(row)
                rt.kernel=Observer(rt,rt.kernel,validation)
                try:
                    with torch.no_grad():
                        for c in range(9):generate_chunk(engine,emb,case['seed'],c)
                finally:rt.close();del rt;gc.collect();torch.cuda.empty_cache()
                assert len(rows)==174;atomic(path,dict(rows=rows));print(json.dumps(dict(epoch=epoch,validation=case['id'],means={x:sum(r['nmse'][x] for r in rows)/len(rows) for x in rows[0]['nmse']})),flush=True)
        select=[]
        for epoch in range(args.epochs):
            rows=[r for p in (ROOT/'validation').glob(f'e{epoch}_*.json') for r in json.loads(p.read_text())['rows']]
            select.append(dict(epoch=epoch,means={x:sum(r['nmse'][x] for r in rows)/len(rows) for x in rows[0]['nmse']}))
        choice=min(select,key=lambda x:min(x['means']['joint'],x['means']['swap']))
        atomic(ROOT/'selection.json',dict(rows=select,selected_epoch=choice['epoch'],scope='validation attention only, no video pass; final ablations all retained'))
        atomic(ROOT/'status.json',dict(status='training_and_validation_complete',done=len(done)))
    except Exception as exc:atomic(ROOT/'status.json',dict(status='failed',error=repr(exc)));raise
    finally:lock.close()
if __name__=='__main__':main()
