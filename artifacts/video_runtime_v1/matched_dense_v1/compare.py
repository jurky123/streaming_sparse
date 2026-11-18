"""Match each replaced self-attention to its ordinal in the Dense trajectory."""
from collections import Counter
import json
from pathlib import Path
from statistics import median

root=Path(__file__).resolve().parent
def compare(index):
    old=json.loads((root/f'dense{index}.json').read_text())
    new=json.loads((root/f'base{index}.json').read_text())
    assert len(new)==len(old)==9
    by_shape={}
    chunks=[]
    for i,(d,s) in enumerate(zip(old,new)):
        assert d['chunk']==s['chunk']==i
        dense,sparse=d['attention_events'],s['attention_events']
        assert len(dense)==len(sparse)==150
        assert all(v['name']=='flash' and v['grid']==[36,1,12] for v in dense)
        timing=Counter()
        for ordinal,(v,w) in enumerate(zip(dense,sparse)):
            step,layer=divmod(ordinal,30)
            replaced=i>=2 and step<4 and layer>0
            assert w['name']==('exact' if replaced else 'flash')
            if not replaced:
                timing['common_dense_ms']+=v['us']/1000
                timing['common_base_ms']+=w['us']/1000
                continue
            assert w['grid']==[72,12,1]
            timing['calls']+=1
            timing['replaced_dense_ms']+=v['us']/1000
            timing['kernel_ms']+=w['us']/1000
        if i<2:continue
        assert timing['calls']==116
        pages=min(i+1,6)*72
        mma=192*(pages*15//100)+102*((pages+15)//16-1)-32
        main_flop=mma*3456*4096
        work_fraction=main_flop/(4*4608*12*128*pages*64)
        timing['work_tflop']=116*main_flop/1e12
        timing['matched_reference_ms']=work_fraction*timing['replaced_dense_ms']
        timing['subtraction_reference_ms']=timing['matched_reference_ms']+work_fraction*(timing['common_dense_ms']-timing['common_base_ms'])
        by_shape.setdefault(pages,Counter()).update(timing)
        chunks.append(dict(chunk=i,pages=pages,**timing))
    total=sum(by_shape.values(),Counter())
    def result(c):
        return dict(c,throughput_fraction=c['matched_reference_ms']/c['kernel_ms'],
                    gap_ms=c['kernel_ms']-c['matched_reference_ms'])
    return dict(total=result(total),windows={p:result(c) for p,c in by_shape.items()},chunks=chunks)
if __name__=='__main__':
    rows=[compare(i) for i in (1,2)]
    (root/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    for row in rows:
        print(row['total'])
        print([(p,round(c['throughput_fraction'],5)) for p,c in row['windows'].items()])
