"""Intervention, surrogate, metric, coverage, serialization and resume contracts."""
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
import sys
import torch
from .model import PairedRouter,RouterConfig,load_pair
from .data import BalancedSchedule
from .metrics import summarize,group_summary,across_router_seeds
from .train import train_update,configuration
from .evaluate import evaluate_snapshot
from .evaluate import evaluate
from .paths import protect_output
from ..v03.sanity import synthetic_events
from ..v03.evaluate import statistics
from ..v03.metrics import error_parts
from ..attention.block_statistics import exact_attention


class Correctness(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)
        self.events=synthetic_events()
        self.model=PairedRouter(RouterConfig(target_size=16,bf16=False))

    def state(self):
        e=self.events[0]
        return self.model.precompute(e['k_hist'],e['v_hist'])

    def test_equal_capacity_and_matched_initialization(self):
        a,b=self.model.branches.values()
        self.assertEqual(self.model.capacity()['branches']['A'],self.model.capacity()['branches']['B'])
        self.assertEqual(len(a.interaction),2);self.assertEqual(len(b.interaction),2)
        for (n,p),(m,q) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(n,m);self.assertNotEqual(id(p),id(q))
            torch.testing.assert_close(p,q,rtol=0,atol=0)
        self.assertFalse(hasattr(self.model.frontend,'encoder'))

    def test_shared_z_and_no_query_leak_or_recompute(self):
        count={'frontend':0,'A':0,'B':0}
        handles=[]
        def hook(name):
            def add(module,args,out):count[name]+=1
            return add
        handles.append(self.model.frontend.register_forward_hook(hook('frontend')))
        for name,branch in self.model.branches.items():
            for layer in branch.interaction:handles.append(layer.register_forward_hook(hook(name)))
        state=self.state()
        self.assertIs(state['contexts']['A'],state['z'])
        b=state['contexts']['B'].detach().clone()
        q=torch.randn(16,128,requires_grad=True)
        self.assertIsNone(torch.autograd.grad(state['contexts']['B'].sum(),q,allow_unused=True,retain_graph=True)[0])
        first=self.model.route(q,state,.2)
        last=self.model.route(q*3+2,state,.2)
        torch.testing.assert_close(b,state['contexts']['B'].detach(),rtol=0,atol=0)
        self.assertEqual(count,dict(frontend=1,A=4,B=2))
        for name in ('A','B'):
            self.assertGreater(float((first[name]['scores']-last[name]['scores']).abs().max()),1e-6)
            g,=torch.autograd.grad(first[name]['scores'].sum(),q,retain_graph=True)
            self.assertGreater(float(g.norm()),0)
        for h in handles:h.remove()

    def test_dense_raw_gather_and_integer_budget_all_keeps(self):
        state=self.state();e=self.events[0]
        stat=statistics(e['q'],e,state['assignment'])
        torch.testing.assert_close(stat.dense(),e['o_dense'],rtol=5e-5,atol=5e-5)
        for keep in (0,.1,.2,.3,1):
            for route in self.model.route(e['q'],state,keep).values():
                tokens=state['hard']@route['hard_mask']>0
                self.assertLessEqual(int(tokens.sum()),route['budget'])
                direct=exact_attention(e['q'],torch.cat((e['k_hist'][tokens],e['k_current'])),
                                       torch.cat((e['v_hist'][tokens],e['v_current'])))
                torch.testing.assert_close(stat.output(route['mask']),direct,rtol=5e-5,atol=5e-5)

    def test_frontend_gets_equal_mean_loss_gradient(self):
        state=self.state();e=self.events[0]
        stat=statistics(e['q'],e,state['assignment'])
        routes=self.model.route(e['q'],state,.2)
        losses=[]
        for name in ('A','B'):
            se,en=error_parts(stat.output(routes[name]['mask']),e['o_dense'])
            losses.append(se/(en+1e-8))
        params=list(self.model.frontend.parameters())
        gradients=[torch.autograd.grad(loss,params,retain_graph=True) for loss in losses]
        joint=.5*(losses[0]+losses[1])
        actual=torch.autograd.grad(joint,params,retain_graph=True)
        # Early vs late VJP summation has FP32 cancellation in clustering/biases.
        for a,b,c in zip(actual,*gradients):torch.testing.assert_close(a,.5*(b+c),rtol=1e-4,atol=5e-6)
        cluster,=torch.autograd.grad(joint,state['probabilities'],retain_graph=True)
        self.assertGreater(float(cluster.norm()),0)
        joint.backward()
        for module in (self.model.frontend,*self.model.branches.values()):
            self.assertGreater(sum(float(p.grad.norm()) for p in module.parameters() if p.grad is not None),0)

    def test_sampler_coverage_and_exact_state_restore(self):
        sampler=BalancedSchedule([str(i) for i in range(8)],0)
        seen={v:set() for v in sampler.versions}
        for _ in range(16):
            batch=sampler.sample(4)
            self.assertEqual(len({v for v,tile in batch}),4)
            for v,tile in batch:seen[v].add(tile)
        self.assertTrue(all(len(tiles)==8 for tiles in seen.values()))
        state=deepcopy(sampler.state_dict());expected=sampler.sample(4)
        restored=BalancedSchedule(sampler.versions,123);restored.load_state_dict(state)
        self.assertEqual(restored.sample(4),expected)

    def test_optimizer_serialization_and_update_resume(self):
        opt=torch.optim.AdamW(self.model.parameters(),lr=1e-4,weight_decay=.01)
        batch=[(self.events,0)]
        train_update(self.model,batch,.2,opt)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'pair.pt'
            torch.save(dict(**self.model.checkpoint(),optimizer=opt.state_dict()),path)
            resumed,value=load_pair(path)
            newopt=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=.01)
            newopt.load_state_dict(value['optimizer'])
            train_update(self.model,batch,.2,opt);train_update(resumed,batch,.2,newopt)
            for a,b in zip(self.model.parameters(),resumed.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_evaluation_pair_schema_and_single_tile_diagnostic(self):
        events=[]
        for i,e in enumerate(self.events):
            events.append(dict(**e,cache_version='test',prompt_id='train',seed=0))
        report=evaluate_snapshot(self.model,events,(.2,))
        self.assertEqual(len(report['rows']),4)
        result=summarize(report['rows'])['keep=0.2']
        self.assertEqual(result['budget_violations'],0)
        self.assertIn('p99',result['A']['tile_relative_error'])
        self.assertEqual(report['query_sensitivity'][0]['pair_count'],0)
        with self.assertRaises(ValueError):summarize(report['rows']+report['rows'][:1])

    def test_energy_global_metric_is_not_mean_ratio(self):
        rows=[]
        for i,(sse,energy) in enumerate([(1.,1.),(1.,100.)]):
            errors={name:dict(sse=sse,energy=energy,error=sse/(energy+1e-8)) for name in ('A','B','Mass')}
            rows.append(dict(cache_version='x',step_id=0,tile_id=i,errors=errors,budget=1,retained=dict(A=1,B=1,Mass=1)))
        result=group_summary(rows)
        self.assertAlmostEqual(result['A']['global_attention_error'],2/(101+1e-8))
        self.assertGreater(result['A']['tile_relative_error']['mean'],.5)
        self.assertEqual(result['paired']['win_rate_A'],0.)

    def test_formal_configuration_and_output_guards(self):
        self.assertEqual(configuration(.2,0)['steps'],6000)
        with self.assertRaises(ValueError):configuration(.1,0)
        with self.assertRaises(ValueError):configuration(.2,3)
        with self.assertRaises(ValueError):protect_output(Path(__file__).parents[1]/'v03/runs/bad')

    def test_snapshot_commit_resume_and_checkpoint_identity(self):
        events=[{**e,'cache_version':'test','prompt_id':'train','seed':0,
                 'q':e['q'].repeat(8,1),'o_dense':e['o_dense'].repeat(8,1),'query_indices':torch.arange(128)}
                for e in self.events]
        class View:
            split='train'
            groups={'test':None}
            calls=0
            def fingerprint(self):return 'fixture'
            def load(self,version,device):self.calls+=1;return events
        view=View()
        with tempfile.TemporaryDirectory() as tmp:
            first=evaluate(self.model,view,tmp,keeps=(.2,),checkpoint_sha='first')
            second=evaluate(self.model,view,tmp,keeps=(.2,),checkpoint_sha='first')
            self.assertEqual(view.calls,1)
            self.assertEqual(first['metrics'],second['metrics'])
            with self.assertRaises(ValueError):evaluate(self.model,view,tmp,keeps=(.2,),checkpoint_sha='different')

    def test_three_router_seed_reporting_and_missing_seed_guard(self):
        events=[dict(**e,cache_version='test',prompt_id='train',seed=0) for e in self.events]
        metrics=summarize(evaluate_snapshot(self.model,events,(.2,))['rows'])
        reports={seed:deepcopy(metrics) for seed in (0,1,2)}
        expected=metrics['keep=0.2']['A']['global_attention_error']
        for seed in (0,1,2):
            for value in reports[seed].values():value['A']['global_attention_error']*=seed+1
        result=across_router_seeds(reports)
        self.assertAlmostEqual(result['keep=0.2']['A']['mean'],2*expected)
        self.assertAlmostEqual(result['keep=0.2']['A']['std'],expected)
        with self.assertRaises(ValueError):across_router_seeds({0:metrics})

    def test_dual_gpu_scheduler_with_cpu_only_fixture_processes(self):
        from .workflow import execute,validate
        from .identity import source_identity
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            script=('from pathlib import Path; import os,sys,time; '
                    'r=Path(sys.argv[1]); me=sys.argv[2]; other=sys.argv[3]; '
                    '(r/("start"+me)).write_text("started"); '
                    'deadline=time.monotonic()+3; '
                    '\nwhile not (r/("start"+other)).exists() and time.monotonic()<deadline: time.sleep(.01)\n'
                    'assert (r/("start"+other)).exists(),"workers did not overlap"\n'
                    '(r/("done"+me)).write_text(os.environ["CUDA_VISIBLE_DEVICES"])')
            jobs=[dict(id=f'worker{i}',gpu_physical_index=i,depends_on=[],environment={'CUDA_VISIBLE_DEVICES':str(i)},
                       argv=[sys.executable,'-c',script,str(base),str(i),str(1-i)]) for i in (0,1)]
            jobs.append(dict(id='joined',gpu_physical_index=None,depends_on=['worker0','worker1'],
                environment={'CUDA_VISIBLE_DEVICES':''},argv=[sys.executable,'-c',
                    'from pathlib import Path;import sys; p=Path(sys.argv[1]); assert (p/"done0").read_text()=="0"; assert (p/"done1").read_text()=="1"',str(base)]))
            plan=dict(source_sha256=source_identity(),execution=dict(commands=jobs,cwd=tmp,environment={}))
            execute(plan,base/'workflow')
            status=json.loads((base/'workflow/status.json').read_text())
            self.assertTrue(all(v['status']=='completed' for v in status['jobs'].values()))
            bad=deepcopy(plan);bad['execution']['commands'][0]['environment']['CUDA_VISIBLE_DEVICES']='0,1'
            with self.assertRaises(ValueError):validate(bad)


if __name__=='__main__':unittest.main()
