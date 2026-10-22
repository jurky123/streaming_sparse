"""Val-only preregistered teacher/sketch decisions and honest partial reports."""
import argparse,json,math,statistics
from pathlib import Path
from .common import HERE,source_identity,atomic_json,protocol


def teacher_decision():
    s=json.loads((HERE/'phase1/full_val/summary.json').read_text())
    if s['identity']['split']!='val' or not s['identity']['teacher_diagnostic']:raise ValueError('Teacher gate must use Val only')
    m=s['metrics']['keep=0.2'];betas=protocol()['teacher']['coefficients']
    beta=min(betas,key=lambda b:(m[f'T3_beta{b}']['macro_error'],b));name=f'T3_beta{beta}'
    delta=1-m[name]['macro_error']/m['T2']['macro_error']
    paired=m['paired'][f'{name}_vs_T2']
    headroom=delta>=.005 and paired['group_win_fraction']>=.5
    result=dict(source_sha256=source_identity(),selection_split='val',beta=beta,headroom=headroom,
        T2=m['T2'],T3=m[name],relative_macro_improvement=delta,paired=paired,
        rule=protocol()['teacher']['headroom_gate'],no_test_reads=True,
        action='Continue small set-correction controls' if headroom else
               'No stable one-pass teacher headroom: run fixed-query local-swap diagnostic, continue Sketch/base/local; skip full R4 training and do not enlarge networks')
    atomic_json(HERE/'reports/teacher_decision.json',result);print(json.dumps(result,indent=2))


def sketch_decision():
    baseline=json.loads((HERE/'runs/seed0/S1/full_val/summary.json').read_text())['sketch_statistics']
    values=[]
    for seed in (0,1,2):
        path=HERE/f'runs/seed{seed}/S4/full_val/summary.json';s=json.loads(path.read_text())
        if s['identity']['split']!='val':raise ValueError('Sketch gate must use Val only')
        stats=s['sketch_statistics'];p0=baseline['all'];p4=stats['all']
        pairs=[k for k in baseline if k.startswith('pair=')]
        values.append(dict(seed=seed,S1=p0,S4=p4,pair_fraction_improved=sum(stats[k]['dense_micro_error']<baseline[k]['dense_micro_error'] for k in pairs)/len(pairs)))
    sufficient=(sum(v['S4']['dense_micro_error']<v['S1']['dense_micro_error'] for v in values)>=2 and
                sum(v['S4']['logz_rmse']<v['S1']['logz_rmse'] for v in values)>=2 and
                statistics.mean(v['pair_fraction_improved'] for v in values)>.5)
    result=dict(source_sha256=source_identity(),sufficient=sufficient,train_secondary_sketches=not sufficient,seeds=values,
                rule=protocol()['sufficient_gate'],no_test_reads=True,S1='Deterministic parameter-free control, evaluated once; not3 independent training seeds')
    atomic_json(HERE/'reports/sketch_decision.json',result);print(json.dumps(result,indent=2))


def final_report():
    output=dict(source_sha256=source_identity(),teacher=json.loads((HERE/'reports/teacher_decision.json').read_text()),
                sketch=json.loads((HERE/'reports/sketch_decision.json').read_text()),results={},
                independent_test='Not run: existing Test exposed in prior research; do not claim publication-level independent Test.',
                scope=protocol()['scope'])
    lines=['CAVIR v0.7 offline Train/Val report','',output['scope'],'',output['independent_test'],'']
    for variant in ('S1','S4','S8','S2','S4-var','S4-K','R3','R2','correction'):
        seeds=(0,) if variant=='S1' else (0,1,2);summaries=[]
        for seed in seeds:
            file=HERE/f'runs/seed{seed}/{variant}/full_val/summary.json'
            if file.exists():summaries.append(json.loads(file.read_text()))
        if len(summaries)!=len(seeds):continue
        output['results'][variant]=summaries
        for name in summaries[0]['metrics']['keep=0.2']:
            rows=[s['metrics']['keep=0.2'].get(name,{}) for s in summaries]
            if not all(isinstance(r,dict) and 'micro_error' in r for r in rows):continue
            micro=[r['micro_error'] for r in rows];macro=[r['macro_error'] for r in rows]
            lines.append(f'{variant} {name}: micro {statistics.mean(micro):.8f} +/- {statistics.stdev(micro) if len(micro)>1 else 0.:.8f}; macro {statistics.mean(macro):.8f}; worst seed micro {max(micro):.8f}')
    atomic_json(HERE/'reports/final_train_val.json',output);(HERE/'reports/RESULTS_SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher-decision',action='store_true');p.add_argument('--sketch-decision',action='store_true');a=p.parse_args()
    if a.teacher_decision:teacher_decision()
    elif a.sketch_decision:sketch_decision()
    else:final_report()


if __name__=='__main__':main()
