"""Check grouped teacher and page selection against direct attention."""
import math
import torch
from .broad_common import ROOT, Engine, cases
from .broad_router import Events,LayerRouter
from .learned_cache.model import predict,initial_mask,output_for_masks
from .learned_cache.io import atomic

@torch.no_grad()
def run():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    engine=Engine();events=Events(range(30));rows=[]
    def callback(current,q,k,v):
        if current['step']!=0:return
        l=current['layer'];e=events.event(current,q,k,v);h=e['h']
        kk=k[0].transpose(0,1).float();vv=v[0].transpose(0,1).float()
        direct=torch.nn.functional.scaled_dot_product_attention(e['q'],kk,vv)
        delta=(direct-e['target']).abs().max().item()
        torch.testing.assert_close(direct,e['target'],atol=3e-5,rtol=3e-4)
        model=LayerRouter(l).cuda();pk,pv,count=model(h)
        torch.testing.assert_close(count.sum(-1),torch.full_like(count[...,0],64.))
        torch.testing.assert_close((pk*count[...,None]).sum(2),h['k'].reshape(12,360,64,128).sum(2),atol=1e-3,rtol=1e-4)
        torch.testing.assert_close((pv*count[...,None]).sum(2),h['v'].reshape(12,360,64,128).sum(2),atol=1e-3,rtol=1e-4)
        z,_,_=predict(e['route'],(pk,pv,count),e['kc'],e['vc']);mask=initial_mask(z)
        via_stats=output_for_masks(e['z'],e['values'],mask[:,:,None])[:,:,0]
        keep=mask[0,0,:-1]
        sk=torch.cat((h['k'][0].reshape(360,64,128)[keep].flatten(0,1),e['kc'][0]))
        sv=torch.cat((h['v'][0].reshape(360,64,128)[keep].flatten(0,1),e['vc'][0]))
        assert sk.shape==(9216,128)
        out=(e['q'][0,:1]@sk.T/math.sqrt(128)).softmax(-1)@sv
        torch.testing.assert_close(out[0],via_stats[0,0],atol=3e-5,rtol=3e-4)
        rows.append(dict(layer=l,dense_max_abs=delta,sparse_max_abs=(out[0]-via_stats[0,0]).abs().max().item()))
    engine.replay(cases('val')[0],callback,stop_chunk=5)
    assert len(rows)==30
    atomic(ROOT/'routing_checks.json',dict(passed=True,rows=rows))
    print('All30 layers: direct dense/sparse attention and page-local conservation passed',flush=True)
if __name__=='__main__':run()
