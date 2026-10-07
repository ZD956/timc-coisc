"""Data-free control tests. No project checkpoint or experimental dataset is opened."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
import numpy as np
import torch
from scipy.optimize import minimize
from control_mpc import (MPCController, MPCConfig, apply_plant_step, future_power,
                         objective, quantize_command, warm_start, run_closed_loop)


class ToyPlant:
    def __init__(self, gain=0.1, bias=50.0, broken=False):
        self.gain,self.bias,self.broken = gain,bias,broken

    def forward_one_step(self, history, future):
        next_temperature = history[:,-1,0] + self.gain*(future[:,1]/5.0-self.bias)
        if self.broken:
            next_temperature = next_temperature*torch.tensor(float('nan'))
        return next_temperature[:,None].repeat(1,7)

    def rollout_native_120s(self, history, future):
        state = history
        outputs = []
        for k in range(future.shape[1]):
            y = self.forward_one_step(state,future[:,k])
            outputs.append(y)
            state = torch.cat((state[:,1:],torch.cat((y[:,:6],future[:,k]),dim=1)[:,None]),dim=1)
        return torch.stack(outputs,dim=1)


def history(value=37.0, uh=50.0):
    h=np.full((20,12),value,dtype=np.float32)
    h[:,6:]=10.0
    h[:,7]=uh*5
    return h


def decide(plant, target, initial=50.0):
    return MPCController(plant).decide(history(uh=initial),np.full(12,target),np.full((12,5),10.0),initial)


class ControlTests(unittest.TestCase):
    def test_zero_error(self):
        self.assertAlmostEqual(decide(ToyPlant(),37.)['command_percent'],50.,places=1)

    def test_heating_needed(self):
        self.assertGreater(decide(ToyPlant(),39.)['command_percent'],50.)

    def test_saturation(self):
        self.assertEqual(decide(ToyPlant(gain=.001),100.)['command_percent'],100.)

    def test_quantization(self):
        self.assertEqual(quantize_command(12.25),12.3)
        self.assertEqual(quantize_command(100.01),100.)

    def test_initial_warm_start(self):
        np.testing.assert_array_equal(warm_start(31.),np.full(12,31.))

    def test_shift_warm_start(self):
        np.testing.assert_array_equal(warm_start(0.,np.arange(12.)),np.r_[np.arange(1.,12.),11.])

    def test_solver_failure_hold(self):
        result=decide(ToyPlant(broken=True),37.,initial=33.3)
        self.assertFalse(result['success'])
        self.assertEqual(result['command_percent'],33.3)

    def test_identical_models_identical_actions(self):
        a=decide(ToyPlant(),39.)
        b=decide(ToyPlant(),39.)
        self.assertEqual(a['command_percent'],b['command_percent'])
        np.testing.assert_array_equal(a['optimal_percent'],b['optimal_percent'])

    def test_repeat_determinism(self):
        self.assertEqual(decide(ToyPlant(),36.)['command_percent'],decide(ToyPlant(),36.)['command_percent'])

    def test_bounds(self):
        for target in (-100.,37.,100.):
            u=decide(ToyPlant(),target)['command_percent']
            self.assertTrue(0<=u<=100)

    def test_cost_formula(self):
        c=MPCConfig(horizon=2,sigma_y_c=1.,lambda_du=1.,lambda_u=0.)
        y=torch.tensor([38.,38.]); u=torch.tensor([60.,60.])
        self.assertAlmostEqual(float(objective(y,[37.,37.],u,50.,c)),1.005,places=6)

    def test_recursive_rollout(self):
        plant=ToyPlant(gain=.1)
        f=future_power(torch.full((12,),60.),np.full((12,5),10.))[None]
        y=plant.rollout_native_120s(torch.tensor(history())[None],f)
        self.assertAlmostEqual(float(y[0,11,6]),49.,places=4)

    def test_no_future_truth_argument(self):
        import inspect
        sig=inspect.signature(MPCController.decide)
        self.assertNotIn('future_truth',sig.parameters)
        self.assertEqual(set(sig.parameters),{'self','history','reference_c','background_raw','previous_percent'})

    def test_plant_state_not_truth(self):
        next_state,y=apply_plant_step(ToyPlant(),history(),60.,np.full(5,10.))
        self.assertAlmostEqual(float(next_state[-1,0]),38.,places=4)
        self.assertAlmostEqual(float(y[6]),38.,places=4)

    def test_background_unchanged(self):
        f=future_power(torch.tensor([60.]),np.array([[11.,12.,13.,14.,15.]]))
        np.testing.assert_array_equal(f[0].numpy(),[11.,300.,12.,13.,14.,15.])

    def test_invalid_shapes_rejected(self):
        with self.assertRaises(ValueError):
            MPCController(ToyPlant()).decide(np.zeros((19,12)),np.full(12,37.),np.full((12,5),10.),50.)

    def test_toy_full12_conditioning(self):
        result=decide(ToyPlant(),38.)
        self.assertTrue(result['success'])
        self.assertLessEqual(result['iterations'],100)
        self.assertTrue(np.isfinite(result['objective']))

    def test_full_vs_block_toy_conditioning(self):
        full=np.eye(12)
        blocked=np.zeros((12,5))
        j=0
        for column,size in enumerate((2,2,2,3,3)):
            blocked[j:j+size,column]=1.0
            j+=size
        for matrix in (full,blocked):
            def cost(x):
                u=matrix@x
                du=np.diff(np.r_[50.,u])/100.
                return float(np.mean(((u-60.)/10.)**2)+np.mean(du**2))
            result=minimize(cost,np.full(matrix.shape[1],50.),method='SLSQP',
                            bounds=[(0,100)]*matrix.shape[1],options={'maxiter':100,'ftol':1e-6})
            self.assertTrue(result.success)
            self.assertLessEqual(result.nit,100)

    def test_toy_closed_loop_separate_states(self):
        plant=ToyPlant(gain=.1)
        internal=ToyPlant(gain=.2)
        trace=run_closed_loop(plant,MPCController(internal),history(),
                              np.full(13,38.),np.full((13,5),10.),2)
        self.assertEqual(len(trace),2)
        self.assertNotEqual(float(trace[0]['internal_prediction'][0,6]),trace[0]['plant_tp'])

    def test_toy_closed_loop_requires_complete_schedule(self):
        with self.assertRaises(ValueError):
            run_closed_loop(ToyPlant(),MPCController(ToyPlant()),history(),np.full(12,37.),np.full((12,5),10.),2)

    def test_v2_default_weights(self):
        c=MPCConfig()
        self.assertEqual((c.sigma_y_c,c.lambda_du,c.lambda_u),(1.,1.,0.))
        self.assertAlmostEqual(float(objective(torch.full((12,),38.),[37.]*12,
                                     torch.full((12,),60.),50.,c)),1.+.1**2/12,places=6)

    def test_v2_scenario_background_and_uh_index(self):
        h=history()
        h[-1,6:]=[17.,225.,23.,29.,31.,37.]
        background=h[-1,[6,8,9,10,11]]
        f=future_power(torch.full((12,),80.),np.broadcast_to(background,(12,5)).copy())
        np.testing.assert_array_equal(f[:,[0,2,3,4,5]].numpy(),
                                      np.broadcast_to(background,(12,5)))
        np.testing.assert_array_equal(f[:,1].numpy(),np.full(12,400.))
        self.assertEqual(float(h[-1,7]/5.),45.)

    def test_v2_same_solver_configuration(self):
        a=MPCController(ToyPlant(),MPCConfig())
        b=MPCController(ToyPlant(),MPCConfig())
        self.assertEqual(a.config,b.config)
        self.assertEqual(a.config.maxiter,100)
        self.assertEqual(a.config.ftol,1e-6)


def run_report():
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(ControlTests)
    result=unittest.TestResult()
    suite.run(result)
    failures={str(test):message for test,message in result.failures+result.errors}
    cases=[]
    for name in sorted(x for x in dir(ControlTests) if x.startswith('test_')):
        title=f'{name} (test_control_mpc.ControlTests.{name})'
        cases.append(dict(test_name=name,expected='PASS',actual='FAIL' if title in failures else 'PASS',
                          status='FAIL' if title in failures else 'PASS',error=failures.get(title)))
    import hashlib
    file=Path(__file__)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(status='READY_FOR_REVIEW_NOT_AUTHORIZED',tests=cases,total_tests=result.testsRun,
                passed_tests=result.testsRun-len(result.failures)-len(result.errors),
                failed_tests=len(result.failures)+len(result.errors),
                implementation_sha256=sha(file.with_name('control_mpc.py')),test_script_sha256=sha(file),
                experimental_data_accessed=False,control_outcome_accessed=False)
    file.with_name('CONTROL_MPC_UNIT_TEST_REPORT_V2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('total_tests','passed_tests','failed_tests')}))
    if not result.wasSuccessful():
        print('\n'.join(failures.values()))
        raise SystemExit(1)


if __name__=='__main__':
    run_report()
