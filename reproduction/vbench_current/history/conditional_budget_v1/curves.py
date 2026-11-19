from calibrate import train,d
def read(x):
 with np.load(d/(x+".npz")) as f:return f["sse"][2:,...,:54].astype(float),f["dense_energy"][2:].astype(float)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
a,b=map(sum,zip(*(read(x) for x in train)))
err=a[3:].sum(0)/b[3:].sum(0)[...,None]
np.savez_compressed(d/'calibration_curves.npz',sse=a,energy=b)
fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
for ax,(s,l,h) in zip(axes.flat,[(0,0,0),(0,0,1),(0,4,4),(1,4,4),(2,14,7),(3,28,11)]):
 for j,t in enumerate((5,8,10,12,15,20)):
  ax.plot(np.arange(9)/8,err[s,l,h,j*9:j*9+9],label=str(t)+'%')
 ax.set_title(f'Step {s}, layer {l+1}, head {h}');ax.set_xlabel('Fraction of non-anchor slots for current chunk');ax.set_ylabel('Local attention NMSE')
axes[0,0].legend(ncol=2,fontsize=8)
fig.savefig(d/'head_curves.png',dpi=150)
