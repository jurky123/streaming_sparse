"""Small matched replay study; atomic chunk checkpoints and bounded disk."""
import argparse,json,random,signal
from pathlib import Path
import torch
from .broad_common import Engine,cases,CHUNKS,ROOT as DATA
from .improve_models import Model,Events,METHODS,LAYERS,objective,evaluate,swaps
from .learned_cache.io import atomic,checkpoint,restore,digest,RunLock,disk_guard
ROOT=Path('artifacts/rollout_improve_20261030')

def identity(args,layers,methods):
    sources=['improve_train.py','improve_models.py','response_hard.py','cache_grouping.py','broad_common.py','learned_cache/data.py','learned_cache/model.py','learned_cache/io.py']
    return dict(code={p:digest(Path('local_experiments')/p) for p in sources},layers=layers,methods=methods,
        epochs=args.epochs,lr=.001,seed=0,history_pages=338,selected_history_pages=72,queries='8tiles*4queries',
        dictionary=digest(DATA/'dictionary.pt'),dataset_manifest=digest(DATA/'manifest.json'),
        inputs={str(p):digest(p) for c in cases('train')+cases('val') for p in sorted((DATA/'trajectories'/c['id']).glob('chunk*.pt'))})

def run(args):
    root=ROOT/args.name;lock=RunLock(root);stop=[False]
    for sig in [signal.SIGTERM,signal.SIGINT]:signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    layers=[int(l) for l in args.layers.split(',')];methods=args.methods.split(',');ident=identity(args,layers,methods)
    models={m:{l:Model(l,m).cuda() for l in layers} for m in methods}
    optim={m:{l:torch.optim.AdamW(models[m][l].parameters(),lr=.001,weight_decay=.0001) for l in layers} for m in methods}
    cursor=dict(epoch=0,phase='val',case=0,chunk=-1);best={m:{l:float('inf') for l in layers} for m in methods};best_models={m:{} for m in methods};curves=[];updates={l:0 for l in layers}
    saved,_=restore(root)
    if saved:
        assert saved['identity']==ident,'resume identity mismatch'
        for m in methods:
            for l in layers:models[m][l].load_state_dict(saved['models'][m][l]);optim[m][l].load_state_dict(saved['optimizers'][m][l])
        cursor=saved['cursor'];best=saved['best'];best_models=saved['best_models'];curves=saved['curves'];updates=saved['updates']
    def state():return dict(identity=ident,cursor=cursor.copy(),models={m:{l:v.state_dict() for l,v in mm.items()} for m,mm in models.items()},
        optimizers={m:{l:v.state_dict() for l,v in mm.items()} for m,mm in optim.items()},best=best,best_models=best_models,curves=curves,updates=updates,
        python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
    def save(status='running'):
        checkpoint(root,state());atomic(root/'status.json',dict(status=status,cursor=cursor,updates=updates,disk=disk_guard(ROOT,50,4)))
    engine=None
    try:
        if cursor['phase']=='completed':return
        engine=Engine();events=Events(layers)
        if saved:
            random.setstate(saved['python_rng']);torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng'])
        units=0;save()
        while cursor['phase']!='completed' and not stop[0]:
            phase=cursor['phase'];epoch=cursor['epoch'];items=cases('train' if phase=='train' else 'val')
            if phase=='train':random.Random(epoch).shuffle(items)
            while cursor['case']<len(items) and not stop[0]:
                case=items[cursor['case']];engine.reset();events.reset();engine.callback=None
                emb=torch.load(DATA/'embeddings'/f'{case["name"]}.pt',weights_only=True).cuda()
                for chunk in range(41):
                    data=torch.load(DATA/'trajectories'/case['id']/f'chunk{chunk:02d}.pt',weights_only=True)
                    if chunk in CHUNKS and chunk>cursor['chunk']:
                        rows=[]
                        def callback(current,q,k,v):
                            l=current['layer']
                            if l not in events.layers:return
                            e=events.event(current,q,k,v)
                            for m in methods:
                                model=models[m][l]
                                if phase=='train':
                                    with torch.enable_grad():
                                        opt=optim[m][l];opt.zero_grad(set_to_none=True);loss=objective(model,e)
                                        assert torch.isfinite(loss),(m,l,current)
                                        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();model.project()
                                    rows.append(dict(method=m,layer=l,step=current['step'],loss=loss.item(),grad=norm.item()))
                                else:
                                    result,z,mask=evaluate(model,e)
                                    row=dict(method=m,layer=l,step=current['step'],case=case['id'],chunk=chunk,**result)
                                    # Final-val oracle/proxy uses same fixed candidate sets and actual page budget.
                                    if epoch==args.epochs:
                                        row['oracle_swap']=swaps(e,z,mask,False);row['proxy_swap']=swaps(e,z,mask,True)
                                    rows.append(row)
                            if phase=='train':updates[l]+=1
                        engine.callback=callback
                        for step in range(4):engine.forward(data['noisy'][step].cuda(),emb,chunk,step)
                        engine.callback=None
                        atomic(root/phase/f'epoch{epoch:02d}'/f'{case["id"]}_c{chunk:02d}.json',rows)
                        cursor['chunk']=chunk;save();units+=1
                        print(json.dumps(dict(phase=phase,epoch=epoch,case=case['id'],chunk=chunk,updates=updates[layers[0]])),flush=True)
                        if args.max_units and units>=args.max_units:stop[0]=True
                        if stop[0]:break
                    engine.forward(data['clean'].cuda(),emb,chunk,4)
                if not stop[0]:cursor['case']+=1;cursor['chunk']=-1;save()
            if stop[0]:break
            if phase=='val':
                rows=[r for p in (root/'val'/f'epoch{epoch:02d}').glob('*.json') for r in json.loads(p.read_text())]
                scores={m:{} for m in methods}
                for m in methods:
                    for l in layers:
                        vals=[x for r in rows if r['method']==m and r['layer']==l for x in r['nmse']]
                        assert len(vals)==4*3*4*12,(m,l,len(vals))
                        score=sum(vals)/len(vals);scores[m][l]=score
                        if score<best[m][l]:best[m][l]=score;best_models[m][l]={k:v.detach().cpu().clone() for k,v in models[m][l].state_dict().items()}
                curves.append(dict(epoch=epoch,scores=scores));atomic(root/'best.pt',dict(identity=ident,models=best_models,metrics=best,curves=curves),True)
                if epoch==args.epochs:cursor['phase']='completed'
                else:cursor.update(epoch=epoch+1,phase='train',case=0,chunk=-1)
            else:cursor.update(phase='val',case=0,chunk=-1)
            save()
        save('completed' if cursor['phase']=='completed' else 'interrupted')
    except Exception as exc:atomic(root/'status.json',dict(status='failed',cursor=cursor,error=repr(exc)));raise
    finally:
        if engine:engine.callback=None
        lock.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='pilot');p.add_argument('--layers',default=','.join(map(str,LAYERS)));p.add_argument('--methods',default=','.join(METHODS));p.add_argument('--epochs',type=int,default=3);p.add_argument('--max-units',type=int,default=0)
    run(p.parse_args())
