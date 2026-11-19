"""SVG-EAR official core with separate Q/KV lengths and per-layer centroids."""
import sys
from pathlib import Path
import torch
EXT=Path('/home/shuchenweng/jzk/streaming_sparse/external_baselines/Sparse-VideoGen')
sys.path.insert(0,str(EXT));sys.path.insert(0,'/tmp/svg_flashinfer_pinned')
from svg.kmeans_utils import batch_kmeans_Euclid,scatter_mean_fused,identify_dynamic_map_estimated,dynamic_block_sparse_prune_fwd_flashinfer
from svg.kernels.triton.permute import permute_tensor_by_labels_triton,apply_inverse_permutation_triton
class EAR:
 def __init__(self):self.centroids={}
 def reset(self):self.centroids={}
 def __call__(self,q,k,v,layer):
  q,k,v=[x.transpose(1,2).contiguous() for x in (q,k,v)];b,h,n,d=q.shape;nk=k.shape[2]
  initial=layer not in self.centroids;old=self.centroids.get(layer,(None,None));iterations=50 if initial else 2
  ql,qc,qs,_=batch_kmeans_Euclid(q.reshape(b*h,n,d),300,max_iters=iterations,init_centroids=old[0])
  kl,kc,ks,_=batch_kmeans_Euclid(k.reshape(b*h,nk,d),1000,max_iters=iterations,init_centroids=old[1]);self.centroids[layer]=(qc,kc)
  qp,order=permute_tensor_by_labels_triton(q,ql,dim=2);kp,korder=permute_tensor_by_labels_triton(k,kl,dim=2);vp,_=permute_tensor_by_labels_triton(v,kl,dim=2,sorted_indices=korder)
  qs=qs.reshape(b,h,300);ks=ks.reshape(b,h,1000);qc=qc.reshape(b,h,300,d);kc=kc.reshape(b,h,1000,d)
  vc=scatter_mean_fused(vp,ks)
  mask=identify_dynamic_map_estimated(qp,kp,vp,qs,ks,qc,kc,vc,top_p=.9,gamma=1,min_kc_ratio=.05).contiguous()
  out=dynamic_block_sparse_prune_fwd_flashinfer(qp,kp,vp,kc,vc,mask,qs,ks,is_cpu=False)
  return apply_inverse_permutation_triton(out,order,dim=2).transpose(1,2).contiguous()
