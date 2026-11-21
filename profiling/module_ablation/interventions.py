import torch

def install(rt,case):
    if case=='no_tail':
        rt.packed_attention=SelectedOnly(rt.packed_attention)
    if case=='whole_window':
        from whole_window import install as install_window
        install_window(rt)
    if case=='set_off':
        rt.set_pset_schedule({})
    if case=='mean_tail':
        rt.packed_attention=MeanReadout(rt.packed_attention)
    if case=='mean':
        build=rt.summary_graph
        def mean(model,k,v,*args,**kwargs):
            s=build(model,k,v,*args,**kwargs)
            s.keys.copy_(k.float().mean(2)[:,:,None].expand_as(s.keys))
            s.mean.copy_(v.float().mean(2));s.detail.zero_()
            return s
        rt.summary_graph=mean
    if case in ('ungrouped','natural_matched'):
        if case=='natural_matched':
            from natural_matched import install as match
            match(rt)
        def natural(inputs):
            outputs=[]
            for model,k,v in inputs:
                shape=(12,72,64,128)
                order=torch.arange(4608,device=k.device).expand(12,-1)
                outputs.append((k[0].transpose(0,1).reshape(shape).contiguous(),v[0].transpose(0,1).reshape(shape).contiguous(),order))
            return outputs
        assert rt.clean_group_batch
        rt.batched_group=natural
    if case=='rebuild':
        rebuild(rt)

def rebuild(rt):
    # Seven-chunk experiment: retain original clean inputs through the first roll.
    records={}
    commit=rt.commit_jobs
    def remember(jobs):
        for l,m,k,v,start,enabled in jobs:
            if start==0:records[l]=[]
            records[l].append((m,k.clone(),v.clone(),start,enabled))
        commit(jobs)
    rt.commit_jobs=remember
    original=rt.forward
    def forward(l,module,x,grid,freqs,cache,start):
        store=rt.stores[l]
        if store.n>72:
            torch.cuda.current_stream().wait_stream(rt.commit_stream)
            c=start//4608
            hist=records[l]
            assert c==len(hist) and c<=6
            if c==6:
                anchor=cache['k'][:,:1536].clone()
            previous=(store.write,store.n,store.commits)
            store.write=store.n=store.commits=0
            for m,k,v,pos,enabled in hist:
                kg,vg,order=rt.batched_group([(m,k,v)])[0]
                rt.commit_group(l,m,kg,vg,order,pos,enabled)
            assert previous==(store.write,store.n,store.commits)
            packed=getattr(rt,'packed_attention',None)
            if packed:packed.summary_versions.pop(store,None)
            if c==6:
                cache['k'][:,:1536].copy_(anchor)
                packed=getattr(rt,'packed_attention',None)
                if packed:
                    from local_experiments.video_runtime_v1.fixed15_layout import pack_range
                    pk,pv=packed.cache(store)
                    pack_range[(24,12,16)](cache['k'],cache['v'],pk,pv,0)
                    packed.summary_versions.pop(store,None)
                def group(t):return t[0].transpose(0,1).reshape(12,24,64,128)
                s=rt.summary_graph(rt.model[l],group(anchor),group(cache['v'][:,:1536]))
                for name in ('keys','mean','detail','stats'):
                    getattr(store.summary,name)[:,:24].copy_(getattr(s,name))
                if l in rt.pset_layers:rt.write_encoding(l,rt.encode(l,s),start=0)
        return original(l,module,x,grid,freqs,cache,start)
    rt.forward=forward



class MeanReadout:
    """Change only omitted-page compensation after the production route selected IDs."""
    def __init__(self,packed):
        self.packed=packed

    def __getattr__(self,name):
        return getattr(self.packed,name)

    def __call__(self,store,q,k,v,s,basis,ids,workspace,**kwargs):
        from local_experiments.review_v1.ops import Summary
        pages=k.size(1)//64
        def mean(t):
            return t[0].transpose(0,1).reshape(12,pages,64,128).float().mean(2).to(s.mean.dtype)
        keys=mean(k)[:,:,None].expand(-1,-1,4,-1).contiguous()
        alt=Summary(keys,mean(v).contiguous(),torch.zeros_like(s.detail).contiguous(),s.stats)
        return self.packed(store,q,k,v,alt,basis,ids,workspace,**kwargs)



class SelectedOnly(MeanReadout):
    def __call__(self,store,q,k,v,s,basis,ids,workspace,**kwargs):
        from local_experiments.video_runtime_v1.attention import exact
        value,_=exact(q,k,v,ids,workspace)
        return value.to(q.dtype)
