"""Equivalence gates and measured full-input timings before accelerated dispatch."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import signal
import time
import torch
from . import accelerate
from ..common import HERE,atomic_json,paired_seed,protocol,source_identity,nmse


def selection_checks(device):
    from ..attention import select
    original=select
    accelerate.install()
    cases=0;timings={}
    try:
        for pages in (1,2,5,73,147,220):
            for tail in (1,8,16,24,63,64):
                c=torch.full((pages,),64.,device=device);c[-1]=tail
                s=torch.randint(-3,4,(41,pages),device=device).float()
                ids=torch.randperm(pages,device=device)
                for budget in sorted(set([0,1,7,8,16,24,63,64,65,127,128,int(c.sum()),
                                          int(float(c.sum())*.1),int(float(c.sum())*.2),int(float(c.sum())*.3)])):
                    a=original(s,c,budget,ids);b=accelerate.page_select(s,c,budget,ids)
                    assert torch.equal(a,b),(pages,tail,budget)
                    cases+=1
        s=torch.randn(4680,220,device=device);c=torch.full((220,),64.,device=device);c[-1]=24
        for label,function in [('original',original),('optimized',accelerate.page_select)]:
            torch.cuda.synchronize();start=time.perf_counter()
            for _ in range(3):function(s,c,2808)
            torch.cuda.synchronize();timings[label]=(time.perf_counter()-start)/3
    finally:
        accelerate.uninstall()
    return dict(exact_cases=cases,full_query_page_selection_seconds=timings)


def multi_gradient_checks(device):
    from ..attention import attend_multi,costs,masks
    paired_seed(8)
    q=torch.randn(257,3,128,device=device,dtype=torch.bfloat16)*.2
    kh=torch.randn(133,3,128,device=device,dtype=torch.bfloat16)*.2
    vh=torch.randn_like(kh);kc=torch.randn(31,3,128,device=device,dtype=torch.bfloat16)*.2;vc=torch.randn_like(kc)
    hard=masks(torch.randn(3*257,3,device=device),costs(133,device=device),.2)[1].reshape(3,257,3)
    weight=torch.randn_like(q)
    saved={}
    for label,enabled in [('original',False),('optimized',True)]:
        if enabled:accelerate.install()
        raw=[x.detach().clone().requires_grad_()for x in(q,kh,vh,kc,vc,hard)]
        out=attend_multi(*raw)
        gradients=torch.autograd.grad((out.float()*weight.float()).sum(),raw)
        saved[label]=(out.detach(),[x.detach()for x in gradients])
        accelerate.uninstall()
    torch.testing.assert_close(saved['original'][0],saved['optimized'][0],atol=0,rtol=0)
    for index,(a,b)in enumerate(zip(saved['original'][1],saved['optimized'][1])):
        torch.testing.assert_close(a,b,atol=5e-5 if index==5 else 0,rtol=5e-5 if index==5 else 0)
    return dict(native_output_exact=True,raw_QKV_gradients_exact=True,
                gate_max_gradient_abs=float((saved['original'][1][-1]-saved['optimized'][1][-1]).abs().max()))


def score_checks(device):
    from ...v04.data import SnapshotData
    from ..backbone import FunctionalBackbone
    from ..router import InformationRouter
    ds=SnapshotData(protocol()['warmstart']['legacy_qkv_root'],'train')
    events=ds.load(sorted(ds.groups)[0],device);e=events[0]
    q=e['q'];kh=e['k_hist'];vh=e['v_hist'];kc=e['k_current'];vc=e['v_current']
    checks={}
    checkpoint=torch.load(HERE/'runs/seed0/calibration/best.pt',weights_only=True,map_location=device)
    for kind in ('K','V'):
        router=InformationRouter(kind).to(device);router.load_state_dict(checkpoint['router'])
        with torch.no_grad():
            for net in router.head.networks:
                net[-1].weight.normal_(0,.01)
        params=[p for p in router.parameters()if p.requires_grad]
        dummy=type('HeadContext',(),{'query_microbatch':16,'router':router})()
        saved={};chosen={0,2,4,6}
        for label,enabled in [('original',False),('optimized',True)]:
            if enabled:accelerate.install()
            st=router.build(kh,vh,0)
            function=FunctionalBackbone._head
            result=function(dummy,q,kh,vh,kc,vc,st,st,.2,0,chosen,True,True,False,False,4,0)
            mask,att,cal,diagnostic,count,_=result
            loss=(mask*torch.linspace(-.2,.2,mask.numel(),device=device).reshape_as(mask)).sum()+.1*att+.01*cal
            grads=torch.autograd.grad(loss,params,allow_unused=True)
            saved[label]=(mask.detach(),att.detach(),cal.detach(),count,[g.detach()if g is not None else None for g in grads])
            accelerate.uninstall()
        a,b=saved['original'],saved['optimized']
        assert torch.equal(a[0],b[0]),'Hard mask changed under query batching'
        assert a[3]==b[3]==len(chosen)
        torch.testing.assert_close(a[1],b[1],atol=5e-5,rtol=5e-5)
        torch.testing.assert_close(a[2],b[2],atol=5e-5,rtol=5e-5)
        differences=[]
        for ga,gb in zip(a[4],b[4]):
            assert (ga is None)==(gb is None)
            if ga is not None:
                torch.testing.assert_close(ga,gb,atol=5e-5,rtol=5e-5)
                differences.append(float((ga-gb).abs().max()))
        checks[kind]=dict(real_queries=len(q),history=len(kh),hard_mask_exact=True,auxiliary_sample_count=a[3],
                          maximum_parameter_gradient_abs=max(differences))
    return checks


def full_input_check(device):
    from ..backbone import FunctionalBackbone,load_teacher
    from ..replay import ReplayDataset
    from ..router import InformationRouter
    from ..lora import inject,configure,lora_parameters
    from ...capture.capture_runner import sdpa_backend
    wrapper=load_teacher(device);inject(wrapper.model,seed=0)
    router=InformationRouter('V').to(device)
    initial=torch.load(HERE/'runs/seed0/warmstart_V/initial.pt',weights_only=True,map_location=device)
    router.load_state_dict(initial['router'])
    # Exercise a nonzero trained correction, not merely the zero residual init.
    with torch.no_grad():
        for net in router.head.networks:net[-1].weight.normal_(0,.001)
    configure(wrapper.model,router,'J-V')
    network=FunctionalBackbone(wrapper,router).eval()
    ds=ReplayDataset('train',require_complete=True)
    index=next(i for i,r in enumerate(ds.rows)if r['history_tokens']==14040 and r['step']==0)
    sample=ds.load(index);parameters=[(n,p)for n,p in network.named_parameters()if p.requires_grad]
    results={}
    for label,enabled in [('original',False),('optimized',True)]:
        if enabled:accelerate.install()
        for _,p in parameters:p.grad=None
        network._eval_history=None;network._eval_sketch.clear();gc.collect();torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with sdpa_backend():
            result=network.replay(sample,collect_aux=True,checkpoint=True,audit=True)
            torch.cuda.synchronize();forward=time.perf_counter()-start
            loss=nmse(result['x0'],sample['target'].to(device))+.1*result['att_aux']+.01*result['cal_aux']
            value=float(loss.detach());output=result['x0'].detach().cpu()
            loss.backward();torch.cuda.synchronize();elapsed=time.perf_counter()-start
        results[label]=dict(forward_seconds=forward,forward_backward_seconds=elapsed,loss=value,
            peak_GiB=torch.cuda.max_memory_allocated()/2**30,output=output,
            gradients={n:p.grad.detach().cpu().clone()for n,p in parameters if p.grad is not None})
        del result,loss
        accelerate.uninstall()
        print(json.dumps(dict(phase='full_input',implementation=label,seconds=elapsed,forward_seconds=forward,loss=value)),flush=True)
    a,b=results['original'],results['optimized']
    torch.testing.assert_close(a['output'],b['output'],atol=.125,rtol=0)
    output_nmse=float(nmse(a['output'],b['output']))
    assert output_nmse<=1e-4
    assert a['gradients'].keys()==b['gradients'].keys()
    max_abs=0.;relative={}
    for n,ga in a['gradients'].items():
        gb=b['gradients'][n]
        # Batching FP32 routing can alter floating accumulation, but all native
        # QKV execution and the original numerical audit limits remain fixed.
        torch.testing.assert_close(ga,gb,atol=5e-5,rtol=2e-4)
        max_abs=max(max_abs,float((ga-gb).abs().max()))
        label='lora'if n.endswith(('lora_A','lora_B'))else'router'
        previous=relative.setdefault(label,[0.,0.])
        previous[0]+=float((ga-gb).double().square().sum());previous[1]+=float(ga.double().square().sum())
    for value in results.values():value.pop('output');value.pop('gradients')
    return dict(sample_id=sample['sample_id'],history_tokens=14040,current_queries=4680,all_pairs=36,
        output_nmse=output_nmse,max_parameter_gradient_abs=max_abs,
        relative_gradient_l2={k:math.sqrt(a/max(b,1e-30))for k,(a,b)in relative.items()},
        measurements=results,speedup=a['forward_backward_seconds']/b['forward_backward_seconds'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pause-pids',type=int,nargs='*',default=[])
    parser.add_argument('--full',action='store_true');args=parser.parse_args()
    path=HERE/'reports/execution/benchmark.json';paused=[]
    report=dict(source_sha256=source_identity(),execution_sha256=accelerate.identity(),passed=False,
        tolerances=dict(hard_mask='exact',raw_native_output='exact',raw_QKV_gradients='exact',
                        parameter_gradient_atol=5e-5,parameter_gradient_rtol=2e-4,
                        full_output_nmse=1e-4,full_output_max_abs=.125),checks={})
    def interrupt(signum,frame):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    try:
        for pid in args.pause_pids:
            cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
            if not any(x in cmd for x in (b'streaming_sparse.v08.train',b'streaming_sparse.v08.workflow')):
                raise ValueError(f'Refuse to pause unrelated PID {pid}')
            os.kill(pid,signal.SIGSTOP);paused.append(pid)
        paired_seed(13)
        for name,function in [('selection',selection_checks),('multi_attention_gradient',multi_gradient_checks),('real_router',score_checks)]:
            print('checking '+name,flush=True);report['checks'][name]=function('cuda');atomic_json(path,report)
        from ..audit import unit_checks
        print('checking existing unit audit',flush=True)
        accelerate.install();report['checks']['original_unit_audit']=unit_checks();accelerate.uninstall()
        if args.full:
            print('checking full input',flush=True);report['checks']['full_input']=full_input_check('cuda')
        report['passed']=True
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        accelerate.uninstall();atomic_json(path,report)
        for pid in reversed(paused):
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        print(json.dumps(dict(passed=report['passed'],report=str(path),error=report.get('error'))),flush=True)


if __name__=='__main__':main()
