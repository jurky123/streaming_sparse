from .cached import *
from scipy.stats import spearmanr

def readcols(path,strings=(),booleans=(),numeric=()):
 keys=list(strings)+list(booleans)+list(numeric);out={k:[] for k in keys}
 with path.open() as f:
  for r in csv.DictReader(f):
   for k in strings:out[k].append(r[k])
   for k in booleans:out[k].append(r[k]=='True')
   for k in numeric:out[k].append(float(r[k]) if r[k] else np.nan)
 return {k:np.array(v) for k,v in out.items()}
def corr(x,y):
 ok=np.isfinite(x)&np.isfinite(y);x=x[ok];y=y[ok]
 return float(spearmanr(x,y).statistic) if len(x)>2 and np.ptp(x)>0 and np.ptp(y)>0 else None

def report():
 sat=[];bounds=[];scale_rows=[];invert=[];similar=[];thresholds=[];context=[];curves=[];source=[]
 names=['K','mean','detail','stats'];fields=['layer','head','page','beta_mae','mass2_beta_mse','local_mse']+['log_scale_'+n for n in names]+['scale_'+n for n in names]+['normalized_rms_'+n for n in names]+['scale_reconstruction_relative_'+n for n in names]
 for seed in [17,29]:
  train=readcols(DEST/f'train_s{seed}/page_scales.csv',strings=['trajectory','identity'],booleans=['endpoint_pair','harmful_predicted_endpoint','positive_predicted_endpoint','background'],numeric=fields)
  pairtrain=readcols(DEST/f'train_s{seed}/similar_page_pairs.csv',strings=['kind','trajectory'],numeric=['layer','head','feature_distance','log_scale_distance','beta_target_rmse','response_target_rmse']+['log_scale_diff_'+n for n in names]);locks={};pairlocks={}
  for l in [1,15,29]:
   for h in [0,3,6,9]:
    ix=np.flatnonzero((train['layer']==l)&(train['head']==h));_,unique=np.unique(train['identity'][ix],return_index=True);uix=ix[unique]
    for n in names:
     lo,hi=np.quantile(train['log_scale_'+n][uix],[.1,.9]);locks[l,h,n]=(lo,hi);thresholds.append(dict(seed=seed,layer=l,head=h,scale=n,p10_log=float(lo),p90_log=float(hi),unique_pages=len(uix)))
    pix=(pairtrain['layer']==l)&(pairtrain['head']==h)&(pairtrain['kind']=='nearest');pairlocks[l,h]=np.quantile(pairtrain['feature_distance'][pix],[.25,.5,.75]).tolist()
  for split in ['train','development']:
   dest=DEST/f'{split}_s{seed}';done=json.loads((dest/'complete.json').read_text());cap=json.loads((dest/'capture_complete.json').read_text());source.append(cap);assert done['events']==(1152 if split=='train' else 288)
   er=readcols(dest/'event_roles.csv',strings=['trajectory','role'],numeric=['layer','samples','pages','component_abs_max','exact_saturated_components','exact_any_samples','exact_all_samples','near99_any_samples','near99_all_samples','near999_any_samples','mean_tanh_derivative','preactivation_abs_max','preactivation_abs_mean','beta_mae','local_mse'])
   for dim in ['ALL','layer','trajectory']:
    for v in (['ALL'] if dim=='ALL' else np.unique(er[dim])):
     for role in np.unique(er['role']):
      ix=(er['role']==role)&(np.ones(len(er['role']),bool) if dim=='ALL' else er[dim]==v)
      if not ix.any():continue
      n=er['samples'][ix].sum();row=dict(seed=seed,split=split,dimension=dim,value=str(v),role=role,events=int(ix.sum()),samples=int(n),abs_phi_max=float(er['component_abs_max'][ix].max()),abs_activation_max=float(er['preactivation_abs_max'][ix].max()),mean_abs_activation=float(np.average(er['preactivation_abs_mean'][ix],weights=er['samples'][ix])),mean_derivative=float(np.average(er['mean_tanh_derivative'][ix],weights=er['samples'][ix])),beta_mae_event_equal=float(er['beta_mae'][ix].mean()))
      for k in ['exact_any_samples','exact_all_samples','near99_any_samples','near99_all_samples','near999_any_samples']:row[k+'_rate']=float(er[k][ix].sum()/n)
      sat.append(row)
   pb=readcols(dest/'page_collision_bounds.csv',strings=['trajectory','kind'],numeric=['layer','samples','collision_samples','contexts','normalized_collision_sse','normalized_prediction_sse','log_total_weight','collision_sse_log','prediction_sse_log'])
   for dim in ['ALL','layer','trajectory']:
    for v in (['ALL'] if dim=='ALL' else np.unique(pb[dim])):
     for kind in ['all','history','current']:
      ix=(np.ones(len(pb['kind']),bool) if kind=='all' else pb['kind']==kind)&(np.ones(len(pb['kind']),bool) if dim=='ALL' else pb[dim]==v);n=int(ix.sum())
      if not n:continue
      col=pb['collision_sse_log'][ix];pred=pb['prediction_sse_log'][ix];a=col[np.isfinite(col)];b=pred[np.isfinite(pred)];frac=math.exp(lse(a)-lse(b)) if len(a) and len(b) else 0. if len(b) else None
      bounds.append(dict(seed=seed,split=split,dimension=dim,value=str(v),kind=kind,page_groups=n,samples=int(pb['samples'][ix].sum()),collision_samples=int(pb['collision_samples'][ix].sum()),mass2_irreducible_fraction_of_reader_sse=frac,page_equal_collision_mse=float(pb['normalized_collision_sse'][ix].mean()),page_equal_reader_mse=float(pb['normalized_prediction_sse'][ix].mean())))
   cpath=dest/'collision_groups.csv'
   if cpath.exists():
    for row in csv.DictReader(cpath.open()):context.append(dict(split=split,**row))
   data=train if split=='train' else readcols(dest/'page_scales.csv',strings=['trajectory','identity'],booleans=['endpoint_pair','harmful_predicted_endpoint','positive_predicted_endpoint','background'],numeric=fields)
   # Identity scales must remain constant; aggregate repeated query contexts for cross-page analysis.
   for l in [1,15,29]:
    for h in [0,3,6,9]:
     base=(data['layer']==l)&(data['head']==h);indices=np.flatnonzero(base)
     for name in names:
      z=data['log_scale_'+name];lo,hi=locks[l,h,name];err=data['scale_reconstruction_relative_'+name];valid=np.isfinite(err);vi=base&valid;norm=data['normalized_rms_'+name]
      invert.append(dict(seed=seed,split=split,layer=l,head=h,block=name,n=len(indices),scale_p10=float(np.quantile(data['scale_'+name][base],.1)),scale_p50=float(np.median(data['scale_'+name][base])),scale_p90=float(np.quantile(data['scale_'+name][base],.9)),recoverable_fraction=float(valid[base].mean()),normalized_rms_eq_one_fraction=float((norm[base]==1).mean()),median_relative_error=float(np.median(err[vi])) if vi.any() else None,p90_relative_error=float(np.quantile(err[vi],.9)) if vi.any() else None))
      for role in ['all','endpoint_pair','harmful_predicted_endpoint','positive_predicted_endpoint','background']:
       ix=base&(np.ones(len(base),bool) if role=='all' else data[role]);nn=int(ix.sum())
       if nn<3:continue
       for stratum,extra in [('all',np.ones(len(base),bool)),('low',z<=lo),('middle',(z>lo)&(z<hi)),('high',z>=hi)]:
        sx=ix&extra
        if not sx.any():continue
        scale_rows.append(dict(seed=seed,split=split,layer=l,head=h,role=role,stratum=stratum,block=name,n=int(sx.sum()),beta_mae=float(np.mean(data['beta_mae'][sx])),local_mse=float(np.mean(data['local_mse'][sx])),scale_beta_spearman=corr(z[sx],data['beta_mae'][sx]),scale_local_spearman=corr(z[sx],data['local_mse'][sx])))
      # Explicit per-trajectory correlations avoid a layer29 pooled effect masquerading as replication.
      for traj in np.unique(data['trajectory'][base]):
       ix=base&(data['trajectory']==traj);scale_rows.append(dict(seed=seed,split=split,layer=l,head=h,role='trajectory_all',stratum=traj,block=name,n=int(ix.sum()),beta_mae=float(data['beta_mae'][ix].mean()),local_mse=float(data['local_mse'][ix].mean()),scale_beta_spearman=corr(z[ix],data['beta_mae'][ix]),scale_local_spearman=corr(z[ix],data['local_mse'][ix])))
   pair=pairtrain if split=='train' else readcols(dest/'similar_page_pairs.csv',strings=['kind','trajectory'],numeric=['layer','head','feature_distance','log_scale_distance','beta_target_rmse','response_target_rmse']+['log_scale_diff_'+n for n in names])
   for l in [1,15,29]:
    for h in [0,3,6,9]:
     for typ in ['nearest','fixed_control']:
      base=(pair['layer']==l)&(pair['head']==h)&(pair['kind']==typ);bins=np.digitize(pair['feature_distance'],pairlocks[l,h])
      for b in ['ALL',0,1,2,3]:
       ix=base&(np.ones(len(base),bool) if b=='ALL' else bins==b)
       if ix.sum()<3:continue
       row=dict(seed=seed,split=split,layer=l,head=h,pair_kind=typ,feature_distance_train_quartile=b,n=int(ix.sum()),median_feature_distance=float(np.median(pair['feature_distance'][ix])),median_log_scale_distance=float(np.median(pair['log_scale_distance'][ix])),median_beta_target_rmse=float(np.median(pair['beta_target_rmse'][ix])),median_response_target_rmse=float(np.median(pair['response_target_rmse'][ix])),scale_contrast_beta_spearman=corr(pair['log_scale_distance'][ix],pair['beta_target_rmse'][ix]),scale_contrast_response_spearman=corr(pair['log_scale_distance'][ix],pair['response_target_rmse'][ix]))
       for name in names:row[name+'_beta_spearman']=corr(pair['log_scale_diff_'+name][ix],pair['beta_target_rmse'][ix])
       similar.append(row)
   # Cross-context near features are descriptive and cannot supply an irreducible bound.
   nr=list(csv.DictReader((dest/'near_phi_descriptive.csv').open())) if (dest/'near_phi_descriptive.csv').exists() else []
   for l in [1,15,29]:
    for scope in ['other_event','same_chunk_step_other_tile','same_chunk_other_step','other_chunk','strict_past_chunk']:
     zz=[r for r in nr if int(r['layer'])==l and r['scope']==scope]
     if zz:curves.append(dict(seed=seed,split=split,layer=l,scope=scope,page_groups=len(zz),median_nearest_phi_distance=float(np.median([float(r['median_phi_distance']) for r in zz])),median_mean_beta_gap=float(np.median([float(r['mean_beta_difference']) for r in zz])),distance_gap_spearman=corr(np.array([float(r['mean_phi_distance']) for r in zz]),np.array([float(r['mean_beta_difference']) for r in zz]))))
 table(DEST/'saturation_summary.csv',sat);table(DEST/'collision_bounds_summary.csv',bounds);table(DEST/'collision_contexts.csv',context);table(DEST/'scale_error_association.csv',scale_rows);table(DEST/'scale_reconstruction.csv',invert);table(DEST/'similar_page_summary.csv',similar);table(DEST/'scale_thresholds_trainonly.csv',thresholds);table(DEST/'near_phi_summary.csv',curves);dump(DEST/'capture_provenance.json',source)
 dump(DEST/'stage_status.json',dict(input_capture='complete',phi_diagnostic='complete',scale_descriptive='complete',conditional_probe='not_started',reader_training=False,test='sealed'));print('REPORT TABLES DONE',flush=True)
if __name__=='__main__':report()
