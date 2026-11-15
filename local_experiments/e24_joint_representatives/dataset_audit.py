from .train import *
import concurrent.futures

def check(path):
 x=torch.load(path,weights_only=True);kv=torch.load(path.parent/x['kv_file'],weights_only=True);ids=kv['ids']
 checks=dict(unique=ids.unique().numel()==ids.numel(),own_ids=torch.equal(ids[x['own']],torch.arange(ids[x['own'],0],ids[x['own'],0]+64)),query_shape=x['q'].shape==(64,128),fp64=x['N'].dtype==torch.float64 and x['Z'].dtype==torch.float64,positive_Z_in_log_domain=bool(torch.isfinite(x['page_logz']).all() and (x['Z'].sum(1)>0).all() and ((x['Z']>0)|((x['page_logz']-x['log_scale'][:,None])< -744)).all()),finite_N=bool(torch.isfinite(x['N']).all()),dense_identity=torch.allclose(x['dense'],x['N'].sum(1)/x['Z'].sum(1)[:,None],atol=1e-12,rtol=1e-12),mask=bool(x['old_mask'][x['own']] and x['old_mask'].sum()==math.floor(len(ids)*.1)))
 assert all(checks.values()),(str(path),checks)
 return str(path.relative_to(OUT)),sha(path),str((path.parent/x['kv_file']).relative_to(OUT))

def run():
 torch.set_num_threads(1);cases=json.loads((OUT/'protocol.json').read_text())['cases'];assert len(set(c['name'] for c in cases))==14;data=files('train')+files('development')+files('test');assert len(data)==2016;hashes={};kvfiles=set()
 with concurrent.futures.ThreadPoolExecutor(8) as ex:
  for name,h,k in ex.map(check,data):hashes[name]=h;kvfiles.add(k)
 with concurrent.futures.ThreadPoolExecutor(8) as ex:
  for name,h in ex.map(lambda n:(n,sha(OUT/n)),sorted(kvfiles)):hashes[name]=h
 dump(OUT/'dataset_manifest.json',dict(events=2016,shared_KV_blocks=len(kvfiles),trajectories=cases,files=hashes,bytes=sum((OUT/p).stat().st_size for p in hashes),passed=True));print('dataset audit passed',len(hashes))
if __name__=='__main__':run()
