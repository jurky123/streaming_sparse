"""Same hard grouping; omit dead gathers and borrow graph output until next replay."""
import copy,math
import torch
from ..cache_grouping import reorder


FUSED_SMALL_DIRECTION = False
PAIRED_GATHER = False
ONE_HOT_GATHER = False

def balanced(x,factors,one_hot=False,exact_small=False,fused_norm=False,fused_sort=False):
    h,n,d=x.shape;ids=torch.arange(n,device=x.device).expand(h,n).clone();groups=1
    for i,factor in enumerate(factors):
        size=n//groups;x=x.reshape(h*groups,size,d);ids=ids.reshape(h*groups,size)
        if one_hot and ((exact_small=='raw' and size in (32,64,128)) or (exact_small in ('large','root') and size in (32,64,128,256,512))):
            from .group_exact_direction import scores
            score=scores(x)
        elif FUSED_SMALL_DIRECTION and size <= 512 and x.is_contiguous():
            from .group_direction import scores
            score = scores(x)
        else:
            xc=x-x.mean(1,keepdim=True);var=xc.square().mean(1)
            axis=var.argmax(-1,keepdim=True)
            if exact_small=='root' and one_hot and size in (1536,4608):
                from .group_exact_direction import root_scores
                score=root_scores(xc,axis)
            elif exact_small and one_hot and size in (32,64,128):
                from .group_exact_direction import scores
                score=scores(xc,axis)
            else:
                if one_hot:
                    projection=xc.gather(2,axis[:,None].expand(-1,size,-1))
                else:
                    direction=torch.zeros_like(var).scatter_(1,axis,1.)
                for iteration in range(4):
                    if iteration or not one_hot:
                        projection=torch.bmm(xc,direction.unsqueeze(-1))
                    direction=torch.bmm(xc.transpose(1,2),projection).squeeze(-1)
                    if fused_norm:
                        from ..gap_group_norm import normalize
                        direction=normalize(direction)
                    else:
                        direction=direction/direction.norm(dim=-1,keepdim=True).clamp_min(1e-8)
                score=torch.bmm(xc,direction.unsqueeze(-1)).squeeze(-1)
        use_radix=fused_sort=='radix' and size in (1536,4608)
        if use_radix or (fused_sort and size in (32,64,128,256,512)):
            from .group_sort import propagate
            x,ids=propagate(score,x,ids,i+1<len(factors),radix=use_radix)
        else:
            order=score.argsort(dim=-1,stable=True)
            if i+1<len(factors):x=x.gather(1,order[:,:,None].expand_as(x))
            ids=ids.gather(1,order)
        groups*=factor
    return ids.reshape(h,n)


def group_features(model,k,v,spatial_softmax=False,fused_features=False,staged_softmax=False):
    kc=k-k.mean(-2,keepdim=True);score=torch.einsum('hptd,hrd->hptr',kc,model.bank)/math.sqrt(128)
    rms=score.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    if not fused_features:key=score/rms
    if staged_softmax:
        from .group_softmax import staged
        a=staged(score,4)
    elif spatial_softmax:
        from .group_softmax import softmax
        a=softmax(score)
    else:a=score.transpose(-2,-1).contiguous().softmax(-1).transpose(-2,-1)
    vp=torch.einsum('hptd,hdr->hptr',v,model.group_v);mean=(a*vp).sum(-2,keepdim=True)
    if fused_features:
        from .group_feature_epilogue import finish
        return finish(score,rms,a,vp,mean,.1*model.group_gate.tanh())
    response=a*k.shape[-2]*(vp-mean);response=response/response.square().mean((-2,-1),keepdim=True).sqrt().clamp_min(1e-6)
    return key+.1*model.group_gate.tanh()[:,None,None]*response


def group(model,k,v,page_order,contiguous_softmax=False,spatial_softmax=False,paired_gather=False,one_hot=False,exact_small=False,fused_norm=False,fused_features=False,staged_softmax=False,fused_sort=False):
    k=k[0].transpose(0,1);v=v[0].transpose(0,1)
    f=(group_features(model,k.float()[:,None],v.float()[:,None],spatial_softmax,fused_features,staged_softmax) if contiguous_softmax or spatial_softmax else model.group_features(k.float()[:,None],v.float()[:,None])).squeeze(1)
    order=balanced(f,[3,3,2,2,2],one_hot,exact_small,fused_norm,fused_sort);fp=reorder(f,order)
    inner=balanced(fp.reshape(-1,64,32),[2,2],one_hot,exact_small,fused_norm,fused_sort).reshape(12,72,64)
    order=order.reshape(12,72,64).gather(2,inner)[:,page_order].flatten(1)
    if paired_gather:
        from .group_gather import gather_pair
        kg,vg=gather_pair(k,v,order)
    else:
        kg,vg=reorder(k,order),reorder(v,order)
    return kg.reshape(12,72,64,128),vg.reshape(12,72,64,128),order


class GraphGrouping:
    """Output aliases internal buffers: consumer must finish before next call."""
    def __init__(self,config,contiguous_softmax=False,spatial_softmax=False,paired_gather=None,one_hot=None,exact_small=False,fused_norm=False,fused_features=False,staged_softmax=False):
        self.fused_sort=False
        self.fused_features=fused_features
        self.staged_softmax=staged_softmax
        self.exact_small=exact_small
        self.fused_norm=fused_norm
        self.paired_gather=paired_gather
        self.one_hot=one_hot
        self.spatial_softmax=spatial_softmax;self.contiguous_softmax=contiguous_softmax;self.config=config;self.entries={};self.calls=0;self.pool=torch.cuda.graph_pool_handle()
    @torch.no_grad()
    def __call__(self,model,k,v):
        key=(tuple(k.shape),k.stride(),v.stride(),k.device,k.dtype)
        gather_flags=(PAIRED_GATHER if self.paired_gather is None else self.paired_gather,
                      ONE_HOT_GATHER if self.one_hot is None else self.one_hot)
        key=(*key,*gather_flags,self.fused_sort)
        if key not in self.entries:
            kk=torch.empty_strided(k.shape,k.stride(),device=k.device,dtype=k.dtype);vv=torch.empty_strided(v.shape,v.stride(),device=v.device,dtype=v.dtype)
            kk.copy_(k);vv.copy_(v);mm=copy.deepcopy(model).requires_grad_(False);order=self.config.page_order(k.device)
            stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):group(mm,kk,vv,order,self.contiguous_softmax,self.spatial_softmax,*gather_flags,self.exact_small,self.fused_norm,self.fused_features,self.staged_softmax,self.fused_sort)
            torch.cuda.current_stream().wait_stream(stream);graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool):out=group(mm,kk,vv,order,self.contiguous_softmax,self.spatial_softmax,*gather_flags,self.exact_small,self.fused_norm,self.fused_features,self.staged_softmax,self.fused_sort)
            self.entries[key]=(kk,vv,mm,order,graph,out)
        kk,vv,mm,order,graph,out=self.entries[key];kk.copy_(k);vv.copy_(v)
        for name in ('bank','group_v','group_gate'):mm.get_parameter(name).copy_(model.get_parameter(name))
        graph.replay();self.calls+=1
        return out
