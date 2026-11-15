from .core import *
from ..broad_common import Engine
from ..review_v1.model import Model
from ..group_response.core import Runtime
from ..group_response.run import BASE
from ..rollout_suite import generate_chunk
import argparse,os,concurrent.futures,gc

def save(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);tmp.replace(path)
@torch.no_grad()
def encode():
 from utils.wan_wrapper import WanTextEncoder
 cases=json.loads((OUT/'protocol.json').read_text())['cases'];folder=OUT/'embeddings';folder.mkdir(exist_ok=True);missing=[c for c in cases if not (folder/(c['name']+'.pt')).exists()]
 if not missing:return
 encoder=WanTextEncoder().cuda().to(torch.bfloat16).eval().requires_grad_(False)
 for c in missing:save(folder/(c['name']+'.pt'),encoder([c['prompt']])['prompt_embeds'].cpu());print('encoded',c['name'],flush=True)
class Collector:
 def __init__(self,engine,model,case):self.engine=engine;self.model=model;self.case=case;self.count=0;self.files=[];self.pool=concurrent.futures.ThreadPoolExecutor(2);self.pending=[]
 @torch.no_grad()
 def __call__(self,l,q,kh,vh,store,start):
  c=self.engine.current['chunk'];step=self.engine.current['step']
  if c not in [2,5,8] or step not in [0,2] or l not in [1,15,29]:return
  hp=store.n;kk=kh[0].transpose(0,1).reshape(12,-1,64,128);vv=vh[0].transpose(0,1).reshape_as(kk);current=SummaryOps.build(self.model[l],kk[:,hp:],vv[:,hp:],cheap=True);hist=store.summary.slice(0,hp);folder=OUT/'data'/self.case['id'];base=(17*l+13*c+7*step+self.case['seed'])%72
  for h in [0,3,6,9]:
   tag=f'c{c}_l{l}_s{step}_h{h}';kvfile=folder/(tag+'_kv.pt');ids=torch.cat([store.token_ids[h,:hp],torch.arange(start,start+4608,device=q.device).reshape(72,64)]);assert ids.unique().numel()==ids.numel();s=Summary(*(torch.cat([getattr(hist,n),getattr(current,n)],1)[h:h+1] for n in ['keys','mean','detail','stats']));payload=dict(k=kk[h].cpu(),v=vv[h].cpu(),ids=ids.cpu(),summary={n:getattr(s,n).cpu() for n in ['keys','mean','detail','stats']},history_pages=hp,layer=l,head=h,chunk=c,step=step,trajectory=self.case['id'],split=self.case['split']);self.pending.append(self.pool.submit(save,kvfile,payload))
   for t in [base,(base+36)%72]:
    query=q[0,t*64:(t+1)*64,h];qd=query.double();logit=torch.einsum('qd,ptd->qpt',qd,kk[h].double())/math.sqrt(128);scale=logit.flatten(1).amax(1);w=(logit-scale[:,None,None]).exp();Z=w.sum(-1);N=torch.einsum('qpt,ptd->qpd',w,vv[h].double());od=N.sum(1)/Z.sum(1)[:,None];basis=self.model[l].basis[h:h+1];cc=contributions(query,s,basis,scale.float());offset=torch.linspace(0,63,16,device=q.device).round().long();own=hp+t;features,_=RouteOps.features(query[None].index_select(1,offset),s);rs=(features*self.model[l].theta[h:h+1,None,None]).sum(-1).mean(1)[0];mask=RouteOps.select(rs[None,None],torch.tensor([[own]],device=q.device),Config())['mask'][0,0];oldj=objective(cc,N.float(),Z.float(),od.float(),mask);eid=tag+f'_t{t}';path=folder/(eid+'.pt');event=dict(q=query.cpu(),N=N.cpu(),Z=Z.cpu(),dense=od.cpu(),log_scale=scale.cpu(),kv_file=kvfile.name,own=own,old_mask=mask.cpu(),old_nmse=float(oldj),trajectory=self.case['id'],split=self.case['split'],layer=l,head=h,chunk=c,step=step,tile=t);self.pending.append(self.pool.submit(save,path,event));self.files.append(str(path.relative_to(OUT)));self.count+=1
   while len(self.pending)>4:self.pending.pop(0).result()
 def finish(self):
  for f in self.pending:f.result()
  self.pool.shutdown();assert self.count==144
@torch.no_grad()
def run(index):
 cases=json.loads((OUT/'protocol.json').read_text())['cases'];case=cases[index];root=OUT/'trajectories'/case['id'];root.mkdir(parents=True,exist_ok=True)
 if (root/'complete.json').exists():return
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;dump(root/'status.json',dict(status='loading',pid=os.getpid()));engine=Engine();model=Model(BASE);model.load_state_dict(torch.load(PRE,map_location='cuda',weights_only=False)['model']);model.requires_grad_(False);collector=Collector(engine,model,case);rt=Runtime(engine,model,'G-KVR','L3',collector);emb=torch.load(OUT/'embeddings'/(case['name']+'.pt'),weights_only=True).cuda();begin=time.time()
 for c in range(9):
  generate_chunk(engine,emb,case['seed'],c);dump(root/'status.json',dict(status='capturing',chunk=c,events=collector.count,seconds=time.time()-begin));print(case['id'],c,collector.count,flush=True)
 collector.finish();dump(root/'complete.json',dict(case=case,events=collector.count,files=collector.files,seconds=time.time()-begin,identity=rt.diagnostics()['identity_checks']));rt.close();dump(root/'status.json',dict(status='complete',events=collector.count))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--encode',action='store_true');p.add_argument('--index',type=int);a=p.parse_args();encode() if a.encode else run(a.index)
