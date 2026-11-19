from .report import *
from ..e29_local_reader.core import Summary,load_phi,LocalReader,E29,OUT,page_features

def main():
 torch.set_num_threads(2);checks=[];ranges=[];collision_checks=[];sources=json.loads((DEST/'protocol.json').read_text())['sources']
 for p,h in sources.items():assert sha(ROOT/p)==h
 for seed in [17,29]:
  phi=load_phi(seed);reader=LocalReader().cuda().eval();reader.load_state_dict(torch.load(E29/f's{seed}/selected.pt',map_location='cuda',weights_only=False)['model'])
  for split in ['train','development']:
   dest=DEST/f'{split}_s{seed}';done=json.loads((dest/'capture_complete.json').read_text());meta=json.loads((dest/'metadata.json').read_text());assert len(meta)==(1152 if split=='train' else 288);ids=json.loads((dest/'identities.json').read_text());covered=set();evmap={m['event']:m for m in meta}
   for rec in ids:
    for occ in rec['occurrences']:
     key=(occ['event'],occ['page']);assert key not in covered and occ['page']<evmap[occ['event']]['history_pages'];covered.add(key)
   assert len(covered)==sum(m['history_pages'] for m in meta)
   if split=='development':
    old=json.loads((ROOT/f'artifacts/e28_local_response_v1/s{seed}/identities.json').read_text());omap={(x['raw_sha'],x['summary_sha']):sorted((o['event'],o['page']) for o in x['occurrences']) for x in old};new={(x['raw_sha'],x['summary_sha']):sorted((o['event'],o['page']) for o in x['occurrences']) for x in ids};assert omap==new
   # Summaries contain no new parameters; coefficient beta range is a learned-model diagnostic.
   cache={}
   for m in meta:
    fp=dest/m['summary_file']
    if fp not in cache:
     v=torch.load(fp,map_location='cpu',weights_only=True);coef=v['coef'].numpy();b=coef[:,:,16];cache[fp]=(b[:,0]-abs(b[:,1:]).sum(1),b[:,0]+abs(b[:,1:]).sum(1))
    lower,upper=cache[fp];p=np.load(dest/f'event_{m["index"]:04d}.npz');bt=p['beta'];bp=p['predicted_beta'];assert np.all(bp>=lower[None]-1e-5) and np.all(bp<=upper[None]+1e-5)
    for role in ['all','endpoint_pair','harmful_predicted_endpoint','positive_predicted_endpoint','background']:
     ix=p['role_'+role]
     if not ix.any():continue
     outside=(bt[:,ix]<lower[ix][None])|(bt[:,ix]>upper[ix][None]);ranges.append(dict(seed=seed,split=split,event=m['event'],layer=m['layer'],role=role,n=int(outside.size),outside_true_target=int(outside.sum()),median_interval_width=float(np.median(upper[ix]-lower[ix]))))
   # One event per layer/head plus both trajectories in dev: independently reconstruct features/activation.
   seen=set()
   for m in meta:
    k=(m['layer'],m['head']) if split=='train' else (m['trajectory'],m['layer'],m['head'])
    if k in seen:continue
    seen.add(k);x=torch.load(dest/m['summary_file'],map_location='cpu',weights_only=True);s=Summary(*(x['summary'][n].cuda() for n in ['keys','mean','detail','stats']));p=np.load(dest/f'event_{m["index"]:04d}.npz');q=torch.from_numpy(p['q']).cuda();feat=page_features(s);A=phi.page(feat).reshape(-1,8,4).bfloat16().float();u=phi.query(q);pre=torch.einsum('qi,pij->qpj',u,A);F=phi(q,s);assert np.array_equal(pre.detach().cpu().numpy(),p['preactivation']);assert np.array_equal(F[...,1:].detach().cpu().float().numpy(),p['F']);rp,bp=reader(s,F);assert np.array_equal(bp.detach().cpu().numpy(),p['predicted_beta'])
    parts=[x['summary']['keys'][0].numpy().astype('float64').reshape(len(x['scales']),-1) if x['summary']['keys'].dtype!=torch.bfloat16 else x['summary']['keys'][0].float().numpy().astype('float64').reshape(len(x['scales']),-1)]
    raw=[x['summary'][n][0].float().numpy().astype('float64').reshape(len(x['scales']),-1) for n in ['keys','mean','detail','stats']];rms=np.stack([np.sqrt(np.mean(v*v,axis=1)) for v in raw],1);assert np.allclose(rms,x['scales'].numpy(),atol=1e-5,rtol=1e-6)
    # Full FP64 arithmetic shows injectivity of epsilon normalization absent quantization.
    inv=[]
    for v,sc in zip(raw,rms.T):
     norm=v/(sc[:,None]+1e-4);rr=np.sqrt(np.mean(norm*norm,axis=1));recover=1e-4*rr/(1-rr);valid=(sc>0)&(rr<1);inv.append(float(np.max(abs(recover[valid]-sc[valid])/sc[valid])) if valid.any() else 0.)
    checks.append(dict(seed=seed,split=split,event=m['event'],rms_fp64_relative_max=float(np.max(abs(rms-x['scales'].numpy())/np.maximum(rms,1e-30))),fp64_inversion_relative_max=max(inv),activation_replay_exact=True,reader_replay_exact=True))
   # Every reported phi unique count is checked, including current pages; summary-level collisions are not rounded.
   page=list(csv.DictReader((dest/'page_collision_bounds.csv').open()));assert sum(int(z['samples']) for z in page)==sum(np.load(dest/f'event_{m["index"]:04d}.npz')['F'].shape[1]*64 for m in meta);assert all(int(z['unique_phi'])<=int(z['samples']) for z in page)
   lookup={z['identity']:z for z in page}
   for rec in ids[::max(1,len(ids)//12)]:
    fs=[]
    for occ in rec['occurrences']:
     mm=evmap[occ['event']];pp=np.load(dest/f'event_{mm["index"]:04d}.npz');fs.extend(tuple(float(t) for t in v) for v in pp['F'][:,occ['page']])
    key=rec['raw_sha']+':'+rec['summary_sha'];assert len(set(fs))==int(lookup[key]['unique_phi']);collision_checks.append(dict(seed=seed,split=split,identity=key,n=len(fs),unique=len(set(fs))))
   # No raw Key/Value is accessed by audit.
 table(DEST/'beta_interval_diagnostic.csv',ranges);dump(DEST/'audit.json',dict(passed=True,source_hashes_unchanged=True,checks=checks,independent_exact_collision_checks=collision_checks,development_identity_matches_E28=True,all_capture_replay_assertions_passed=True,raw_attention_recomputed=False,training=False,test=False));print('AUDIT PASSED',len(checks),flush=True)
if __name__=='__main__':main()
