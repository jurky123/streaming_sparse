from .core import *
if __name__=='__main__':
 path=OUT/'stage1'/EVENTS[0]/'result.json';r=json.loads(path.read_text());original,centered=r['methods'];rows=[]
 for x in [original,centered]:
  score=next(y for y in x['performance'] if y['panel']=='full_legal' and y['probes']==16);error=next(y['net_nmse'] for y in x['page_metrics'] if y['page_subset']=='risk_unselected');rows.append(dict(form=score['method'],city_gain_pct=score['gated_relative_pct'],city_net_nmse=error))
 a,b=rows;chosen=b if b['city_gain_pct']>a['city_gain_pct']+1e-10 or (abs(b['city_gain_pct']-a['city_gain_pct'])<=1e-10 and b['city_net_nmse']<a['city_net_nmse']-1e-12) else a
 dump(OUT/'form_lock.json',dict(created=time.time(),chosen=chosen['form'],city_only=True,rows=rows,city_result_sha=sha(path),bases_sha=sha(OUT/'bases.npz')));print(chosen)
