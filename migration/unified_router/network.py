"""Unified H/current hard attention inside the unchanged frozen streaming model."""
import json
import torch
from torch import nn

from streaming_sparse.v08.backbone import FunctionalBackbone, load_teacher
from .common import config, sha
from .core import build, scores, initial_mask, candidates, features, apply_action
from .learning import ActionHead, choose


class LayerState(nn.Module):
    def __init__(self, parameters):
        super().__init__()
        for name,value in parameters.items():self.register_buffer(name,value)

    def dictionary(self):return dict(self.named_buffers())


class OnlineRouter(nn.Module):
    def __init__(self, root, m, training_seed=0):
        super().__init__();self.states=nn.ModuleDict();self.heads=nn.ModuleDict()
        self.beta=m['monitor_beta'];self.tile=m.get('runtime_query_tile',128)
        self.kind='V32-Set';self.weights={}
        for layer in m['layers']:
            basis_path=root/f'action_basis/layer{layer:02d}.pt'
            basis=torch.load(basis_path,map_location='cpu',weights_only=True)
            self.states[str(layer)]=LayerState(basis['parameters'])
            normal=torch.load(root/f'normalization/layer{layer:02d}.pt',map_location='cpu',weights_only=True)
            for kind in ['K32-Set','V32-Set']:
                path=root/f'P3_training/layer{layer:02d}/{kind}/seed{training_seed}/best.pt'
                state=torch.load(path,map_location='cpu',weights_only=True)
                record=json.loads((path.parent/'best_validation.json').read_text())
                assert state['updates']==record['updates']and state['best_metric']=='nmse'and state['best_value']==record['nmse']
                head=ActionHead(normal[kind]);head.load_state_dict(state['head'])
                self.heads[f'{layer}_{kind}']=head
                self.weights[str(path)]=sha(path)
            self.weights[str(basis_path)]=sha(basis_path)
        self.eval().requires_grad_(False)

    def sketch(self,layer,kh,vh,kc,vc):
        return build(kh,vh,kc,vc,self.states[str(layer)].dictionary(),need_value=self.kind=='V32-Set')

    def route(self,layer,q,sketch,step):
        if torch.backends.cuda.matmul.allow_tf32:
            raise RuntimeError('Router FP32 precision changed after loading; TF32 must stay disabled')
        content='K32' if self.kind=='K32-Set' else 'V32' if self.kind=='V32-Set' else None
        score,value=scores(q,sketch,content=content)
        mask=initial_mask(score,sketch,self.beta)
        if self.kind=='U-Sol':return mask
        actions=candidates(score,mask,sketch.origins,self.beta,seed=16)
        geometry,key=features(q,sketch,score,mask,actions,self.beta,step,value)
        x=torch.cat((geometry,key),-1).reshape(12,q.shape[1],81,24)
        pred=self.heads[f'{layer}_{self.kind}'](x)
        valid=actions['valid'].reshape_as(pred)
        chosen=choose(pred,valid,torch.zeros_like(pred),0.)
        return apply_action(mask,actions,chosen.flatten())


class Network(FunctionalBackbone):
    def __init__(self,wrapper,router,layers):
        super().__init__(wrapper,router,layers=layers);self.routing_records=[];self.set_mode('Dense')

    def set_mode(self,kind):
        assert kind in ['Dense','U-Sol','K32-Set','V32-Set']
        self.router.kind=kind;self.router_version=kind
        self._eval_history=None;self._eval_sketch.clear();self._parameter_versions=None
        self.routing_records=[]

    def _block(self,block,layer,x,e0,grid,freqs,context,history_k,history_v,start,sparse,step,keep,
               collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity):
        if not sparse or layer not in self.layers or not history_k.shape[1]or self.router.kind=='Dense':
            return super()._block(block,layer,x,e0,grid,freqs,context,history_k,history_v,start,False,step,keep,
                collect_aux,aux_seed,detach_router,mass_only,collect_mechanisms,cache_identity)
        if torch.is_grad_enabled()or collect_aux:raise RuntimeError('Unified network adapter is evaluation-only')
        from wan.modules.causal_model import causal_rope_apply
        frames=e0.shape[1];tokens=x.shape[1]//frames
        e=(block.modulation.unsqueeze(1)+e0).chunk(6,dim=2)
        mod=(block.norm1(x).unflatten(1,(frames,tokens))*(1+e[1])+e[0]).flatten(1,2)
        att=block.self_attn;b,s=mod.shape[:2];h=att.num_heads;d=att.head_dim
        q=att.norm_q(att.q(mod)).view(b,s,h,d);k=att.norm_k(att.k(mod)).view(b,s,h,d)
        v=att.v(mod).view(b,s,h,d)
        q=causal_rope_apply(q,grid,freqs,start_frame=start).type_as(v)
        k=causal_rope_apply(k,grid,freqs,start_frame=start).type_as(v)
        kh=history_k[0].transpose(0,1);vh=history_v[0].transpose(0,1)
        kc=k[0].transpose(0,1);vc=v[0].transpose(0,1);qq=q[0].transpose(0,1)
        sketch=self.router.sketch(layer,kh,vh,kc,vc)
        raw_k=torch.cat((kh,kc),1);raw_v=torch.cat((vh,vc),1)
        outputs=[];retained=torch.zeros(2,device=x.device);nq=0
        for offset in range(0,s,self.router.tile):
            qt=qq[:,offset:offset+self.router.tile]
            mask=self.router.route(layer,qt,sketch,step)
            visible=mask.repeat_interleave(sketch.costs.long(),dim=-1)
            # Final attention uses only ORIGINAL KV, including sparse current KV.
            out=torch.nn.functional.scaled_dot_product_attention(qt[None],raw_k[None],raw_v[None],
                attn_mask=visible[None],dropout_p=0.,is_causal=False)
            outputs.append(out[0]);count=mask*sketch.costs[None,None]
            retained[0]+=count[...,sketch.origins==0].sum();retained[1]+=count[...,sketch.origins==1].sum()
            nq+=h*qt.shape[1]
        y=torch.cat(outputs,1).transpose(0,1)[None]
        x=x+(att.o(y.flatten(2)).unflatten(1,(frames,tokens))*e[2]).flatten(1,2)
        x=x+block.cross_attn(block.norm3(x),context,None,crossattn_cache=None)
        ff=block.ffn((block.norm2(x).unflatten(1,(frames,tokens))*(1+e[4])+e[3]).flatten(1,2))
        x=x+(ff.unflatten(1,(frames,tokens))*e[5]).flatten(1,2)
        rh,rc=map(float,retained)
        self.routing_records.append(dict(layer=layer,start=start,step=step,queries=nq,
            history_tokens=kh.shape[1],current_tokens=kc.shape[1],retained_H=rh,retained_C=rc,
            keep=(rh+rc)/(nq*(kh.shape[1]+kc.shape[1])),keep_H=rh/(nq*kh.shape[1]),keep_C=rc/(nq*kc.shape[1])))
        zero=x.new_zeros((),dtype=torch.float32)
        diagnostic=x.new_tensor([rh+rc,nq,rh,rc,nq*(kh.shape[1]+kc.shape[1]),1],dtype=torch.float32)
        return x,k,v,zero,zero,diagnostic,zero,[]


def load_network(directory,training_seed=0):
    root,m=config(directory)
    router=OnlineRouter(root,m,training_seed).cuda()
    net=Network(load_teacher('cuda'),router,m['layers']).eval().requires_grad_(False)
    # Backbone dependencies enable TF32 during import. Training features were
    # computed with TF32 disabled; restore that setting AFTER loading them.
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_num_threads(4)
    net.model_version=m['teacher_sha256']
    return net,m
