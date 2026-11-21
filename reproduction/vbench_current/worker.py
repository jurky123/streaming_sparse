import json,os,sqlite3,sys,time,traceback
from pathlib import Path
import torch
from local_experiments.stream_eval_common import native_vae,decode,save_video
from local_experiments.broad_common import noise
from paths import RUN as r, REPO
gpu=int(os.environ['CUDA_VISIBLE_DEVICES']);method=sys.argv[1];kind=sys.argv[2]
db=sqlite3.connect(r/'jobs.sqlite',timeout=60,isolation_level=None)
torch.set_num_threads(3);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
@torch.no_grad()
def main():
    # Both lengths fit the native three-latent block, then crop decoded RGB at 16 fps.
    vae=None if method in ('intersection','intersection_fast') else native_vae()
    if method=='lightforcing':
        from light import build
        pipe=None
        loaded_chunks=None
        hits=[0]
    else:
        from local_experiments.video_runtime_v1.accepted_tiers import config
        from local_experiments.video_runtime_v1.deployment import configure,make_runtime,load_weights
        from local_experiments.broad_common import Engine
        from local_experiments.rollout_suite import generate_chunk
        conditional=method in ('intersection','intersection_fast')
        cfg=config('base' if method=='dense' else method,local_assets=REPO)
        if method=='dense':cfg.mode='dense_native'
        configure(cfg);engine=Engine();engine.wrapper.requires_grad_(False)
        if method=='dense':
            from local_experiments.video_runtime_v1.backbone_ops import install
            install(engine,cfg);rt=engine
        else:
            rt=make_runtime(engine,*load_weights(cfg)[:4],cfg)
            if conditional:
                from queue_policy import attach
                generate_conditional=attach(rt,engine,10)
    print('ready',method,kind,flush=True)
    while True:
        db.execute('begin immediate')
        candidates=db.execute("select rowid,kind,i,seed from jobs where state='pending' and method=? and (kind=? or ?='both') order by priority,rowid",(method,kind,kind)).fetchall()
        row=next((x for x in candidates if (r/'embeddings'/f'{x[1]}_{x[2]}.pt').exists()),None)
        if row:db.execute("update jobs set state='running',worker=?,tries=tries+1,started=? where rowid=?",(gpu,time.time(),row[0]))
        db.execute('commit')
        if not row:
            if not candidates:return
            time.sleep(3);continue
        rid,k,i,seed=row;chunks=7 if k=='short' else 41; limit=80 if k=='short' else 480
        try:
            emb=torch.load(r/'embeddings'/f'{k}_{i}.pt',weights_only=True).cuda()
            if method=='lightforcing':
                if chunks!=loaded_chunks:
                    pipe=None
                    pipe=build(chunks)
                    if loaded_chunks is None:
                        import wan.modules.sparse_attention as sparse
                        if sparse.DEVICE_SM in (80,120):
                            target,name=sparse._attention,'apply'
                        elif sparse.DEVICE_SM in (90,100) and sparse.FLASH_ATTN_4_AVAILABLE:
                            target,name=sparse,'flash_attn_func_v4'
                        else:
                            raise RuntimeError(f'LightForcing sparse backend unavailable on SM{sparse.DEVICE_SM}')
                        kernel=getattr(target,name)
                        def count_sparse(*args,**kw):
                            hits[0]+=1
                            return kernel(*args,**kw)
                        setattr(target,name,count_sparse)
                    loaded_chunks=chunks
                pipe.text_encoder.embeds=emb
                init=torch.cat([noise(seed,c,0) for c in range(chunks)],1)
                phases=[noise(seed,c,p).flatten(0,1) for c in range(chunks) for p in (1,2,3)]
                calls=iter(phases);saved=torch.randn_like;hits[0]=0
                def paired(x,*a,**kw):
                    z=next(calls);assert z.shape==x.shape
                    return z
                torch.randn_like=paired
                torch.cuda.synchronize();t=time.monotonic()
                try:_,z=pipe.inference(init,['cached'],return_latents=True,profile=False)
                finally:torch.randn_like=saved
                assert next(calls,None) is None and hits[0]>0
            else:
                rt.reset();torch.cuda.synchronize();t=time.monotonic()
                z=generate_conditional(emb,seed,chunks) if conditional else torch.cat([generate_chunk(engine,emb,seed,c) for c in range(chunks)],1)
            torch.cuda.synchronize();g=time.monotonic()-t
            assert torch.isfinite(z).all()
            if method=='intersection' and k=='long' and i in (0,1) and seed==701+i:
                from queue_policy import verify_reference
                verify_reference(z,i,10)
            if vae is None:vae=native_vae()
            t=time.monotonic();frames=decode(vae,z,limit);dt=time.monotonic()-t
            path=r/'videos'/method/k/f'{i}_{seed}.mp4';path.parent.mkdir(exist_ok=True,parents=True)
            save_video(path,frames);del frames,z
            db.execute("update jobs set state='done',gen=?,decode=? where rowid=?",(g,dt,rid))
            print('done',rid,k,i,seed,g,dt,flush=True)
        except BaseException:
            db.execute("update jobs set state=case when tries<2 then 'pending' else 'failed' end,error=? where rowid=?",(traceback.format_exc(),rid))
            raise
if __name__=='__main__':main()
