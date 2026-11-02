"""Stream compact dense replays; train per-layer matched objectives with resume."""
import argparse
import copy
import json
import random
import signal
from pathlib import Path
import torch
from .mass_common import ROOT,CHUNKS,cases,guard,Engine,manifest
from .mass_router import Events,evaluate_event,make_model,objective,METHODS
from .refine_pages import loss_terms
from .learned_cache.io import atomic,checkpoint,restore,digest,RunLock




def source_identity(layers):
    paths=['mass_common.py','mass_router.py','mass_train.py','tile_router.py','broad_common.py','broad_router.py','response_hard.py','cache_grouping.py','refine_pages.py','learned_cache/model.py','learned_cache/data.py','learned_cache/io.py']
    data={}
    for case in cases('train')+cases('val'):
        folder=ROOT/'trajectories'/case['id']
        assert json.loads((folder/'status.json').read_text())['status']=='completed'
        for p in sorted(folder.glob('chunk*.pt')):data[str(p)]=digest(p)
    return dict(layers=layers,epochs=6,seed=0,lr=.001,code={p:digest(Path('local_experiments')/p) for p in paths},
        data=data,dictionary=digest(ROOT/'dictionary.pt'),protocol=digest(ROOT/'manifest.json'))


def run(args):
    manifest();layers=list(range(args.first,args.last));root=ROOT/f'train_{args.first:02d}_{args.last:02d}';lock=RunLock(root)
    stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    ident=source_identity(layers);models={m:{} for m in METHODS};optimizers={m:{} for m in METHODS}
    for layer in layers:
        for method in METHODS:
            models[method][layer]=make_model(layer,method).cuda()
            optimizers[method][layer]=torch.optim.AdamW(models[method][layer].parameters(),lr=.001,weight_decay=.0001)
    cursor=dict(epoch=0,phase='val',case=0,chunk=-1);best={m:{l:float('inf') for l in layers} for m in METHODS};best_models={m:{} for m in METHODS}
    updates={l:0 for l in layers};curves=[];saved,name=restore(root)
    if saved:
        assert saved['identity']==ident,'Training resume identity mismatch'
        for m in METHODS:
            for l in layers:
                models[m][l].load_state_dict(saved['models'][m][l]);optimizers[m][l].load_state_dict(saved['optimizers'][m][l])
        cursor=saved['cursor'];best=saved['best'];best_models=saved['best_models'];updates=saved['updates'];curves=saved['curves']
        print('restored',name,cursor,flush=True)
    def state():return dict(identity=ident,cursor=cursor.copy(),models={m:{l:models[m][l].state_dict() for l in layers} for m in METHODS},
        optimizers={m:{l:optimizers[m][l].state_dict() for l in layers} for m in METHODS},best=best,best_models=best_models,updates=updates,curves=curves,
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    if cursor['phase']=='completed':lock.close();return
    engine=Engine();events=Events(layers);units=0
    if saved:
        random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
    def save_status(status):atomic(root/'status.json',dict(status=status,**cursor,updates=updates,disk=guard()))
    def save():checkpoint(root,state());save_status('running')
    try:
        save_status('running')
        while cursor['phase']!='completed' and not stop[0]:
            phase=cursor['phase'];events.training=phase=='train';epoch=cursor['epoch'];items=cases('train' if phase=='train' else 'val')
            if phase=='train':random.Random(epoch).shuffle(items)
            while cursor['case']<len(items) and not stop[0]:
                ci=cursor['case'];case=items[ci];folder=ROOT/'trajectories'/case['id']
                emb=torch.load(ROOT/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda();engine.reset();events.reset();engine.callback=None
                for chunk in range(41):
                    data=torch.load(folder/f'chunk{chunk:02d}.pt',weights_only=True)
                    if chunk in CHUNKS and chunk>cursor['chunk']:
                        rows=[]
                        def callback(current,q,k,v):
                            layer=current['layer']
                            if layer not in events.layers:return
                            e=events.event(current,q,k,v)
                            if phase=='train':
                                for method in METHODS:
                                    with torch.enable_grad():
                                        opt=optimizers[method][layer];opt.zero_grad(set_to_none=True)
                                        loss=objective(models[method][layer],e,method)
                                        assert torch.isfinite(loss),(case['id'],current,method)
                                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(models[method][layer].parameters(),1.,error_if_nonfinite=True);opt.step();models[method][layer].project()
                                        rows.append(dict(layer=layer,step=current['step'],method=method,loss=loss.item(),grad=norm.item()))
                                updates[layer]+=1
                            else:
                                for method in METHODS+(['mean'] if epoch==0 else []):
                                    metrics=evaluate_event(None if method=='mean' else models[method][layer],e)
                                    rows.append(dict(layer=layer,step=current['step'],method=method,case=case['id'],chunk=chunk,**metrics))
                        engine.callback=callback
                        for step in range(4):engine.forward(data['noisy'][step].cuda(),emb,chunk,step)
                        engine.callback=None
                        atomic(root/phase/f'epoch{epoch:02d}'/f'{case["id"]}_c{chunk:02d}.json',rows)
                        cursor['chunk']=chunk;save();units+=1
                        print(json.dumps(dict(shard=layers,epoch=epoch,phase=phase,case=case['id'],chunk=chunk,updates=updates[layers[0]])),flush=True)
                        if args.max_units and units>=args.max_units:stop[0]=True
                        if stop[0]:break
                    engine.forward(data['clean'].cuda(),emb,chunk,4)
                if not stop[0]:cursor['case']+=1;cursor['chunk']=-1;save()
            if stop[0]:break
            if phase=='val':
                allrows=[]
                for p in (root/'val'/f'epoch{epoch:02d}').glob('*.json'):allrows.extend(json.loads(p.read_text()))
                scores={m:{} for m in METHODS}
                for method in METHODS:
                    for layer in layers:
                        vals=[x for r in allrows if r['method']==method and r['layer']==layer for x in r['nmse']]
                        assert len(vals)==4*3*4*12,(method,layer,len(vals))
                        score=sum(vals)/len(vals);scores[method][layer]=score
                        if score<best[method][layer]:
                            best[method][layer]=score;best_models[method][layer]={k:v.detach().cpu().clone() for k,v in models[method][layer].state_dict().items()}
                curves.append(dict(epoch=epoch,scores=scores));atomic(root/'best.pt',dict(identity=ident,models=best_models,metrics=best,curves=curves),True)
                if epoch==6:cursor['phase']='completed'
                else:cursor.update(epoch=epoch+1,phase='train',case=0,chunk=-1)
            else:cursor.update(phase='val',case=0,chunk=-1)
            save()
        save();save_status('completed' if cursor['phase']=='completed' else 'interrupted')
    except Exception as exc:
        atomic(root/'status.json',dict(status='failed',cursor=cursor,error=repr(exc)));raise
    finally:engine.callback=None;lock.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int,default=30);p.add_argument('--max-units',type=int,default=0)
    a=p.parse_args();run(a)
