"""Shared pieces for the post-hoc onset-aligned COISC study (POSTHOC_ONSET_ALIGNED_PROTOCOL)."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np

from _paths import PKG as HERE, PROTOCOLS, CONFIGS, OUTPUT_ROOT, VERIFY_HASHES, load_checkpoint, private_path  # noqa: E402,F401
import screen  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG before torch use)
import torch  # noqa: E402

OUT = OUTPUT_ROOT / 'posthoc_onset_aligned'
PROTO = PROTOCOLS / 'posthoc_onset_aligned'
SCREEN_DEV = OUTPUT_ROOT / 'screen' / 'development'
FORMAL = OUTPUT_ROOT / 'formal_coisc'
ABLATION = OUTPUT_ROOT / 'posthoc_ablation'
DELIVERED23 = OUTPUT_ROOT / 'delivered23'
DRAFT = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'POSTHOC_ONSET_ALIGNED_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'

E1_SEEDS = (20260922, 20260923, 20260924)
E2_SEEDS = tuple(range(20261001, 20261011))
LABELS = ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'Tp')
TAU_PRIOR = (102.108, 77.091, 30.108, 50.112, 10.0535, 180.0, 39.104)
TAU_ONSET = (60.0, 40.0, 30.108, 45.0, 10.0535, 70.0, 20.0)
LAMBDA_PRIOR = 939.1430335805455
CAL_SEED = 20260922
ADAPT_STEPS = 75
assert tuple(screen.MIN_LAG_SECONDS) == TAU_PRIOR


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def masks(lags):
    return screen.lag_masks(lags)


def regularizer(values, lags):
    pre, post = [torch.as_tensor(v, device=values.device) for v in masks(lags)]
    return (torch.relu(values[:, pre].abs() - screen.EPS_PRE).square().mean(),
            torch.relu(-values[:, post]).square().mean())


def calibrate_both(train, stats, probs, stages, device):
    """screen.calibrate with SEED=CAL_SEED, evaluating prior and onset masks on the same pilots and probes."""
    start = time.perf_counter(); records = {'prior': [], 'onset': []}
    for excitation in (False, True):
        screen.seed_all(CAL_SEED); rng = np.random.default_rng(CAL_SEED)
        model = screen.build_model('B4', stats).to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=stages[0]['lr'])
        for _ in range(screen.PILOT_STEPS):
            h, u, y = screen.sample(train, rng, stages[0]['batch'], probs[excitation], device)
            loss = screen.factual(model, h, u, y, stages[0]); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
        for k in range(8):
            h, u, y = screen.sample(train, rng, stages[0]['batch'], probs[excitation], device)
            prediction = screen.factual(model, h, u, y, stages[0])
            values = screen.sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET])
            for name, lags in (('prior', TAU_PRIOR), ('onset', TAU_ONSET)):
                pre, sign = regularizer(values, lags); reg = pre + sign
                grads = torch.autograd.grad(reg, tuple(model.parameters()), allow_unused=True, retain_graph=True)
                gradnorm = float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
                if not gradnorm > 0: raise ValueError(f'{name} regularizer has no effective gradient')
                records[name].append(dict(excitation=excitation, batch=k, prediction=float(prediction.detach()),
                                          regularizer=float(reg.detach()), pre=float(pre.detach()), sign=float(sign.detach()),
                                          gradient_norm=gradnorm))
    out = {}
    for name, rec in records.items():
        groups = [screen.calibrated_lambda([r for r in rec if r['excitation'] == e]) for e in (False, True)]
        out[name] = dict(lambda_s=min(groups), per_sampler=groups, records=rec)
    out['time_s'] = time.perf_counter() - start
    return out


def adapt(model, train, probs, old, w_pre, w_sign, lags, seed, device, log_path, steps=ADAPT_STEPS):
    """formal_coisc.adapt_head / ablation adapt with explicit lag masks."""
    screen.seed_all(seed)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('head.'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=5e-5)
    rng = np.random.default_rng(seed); stage = dict(old['training']['stages'][2]); start = time.perf_counter()
    with Path(log_path).open('w', encoding='utf-8') as log:
        for step in range(1, steps + 1):
            h, u, y = screen.sample(train, rng, 128, probs[False], device)
            factual = screen.factual(model, h, u, y, stage)
            sens = screen.sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET]); pre, sign = regularizer(sens, lags)
            loss = factual + w_pre * pre + w_sign * sign
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
            log.write(json.dumps(dict(step=step, prediction_loss=float(factual.detach()), L_pre=float(pre.detach()),
                                      L_sign=float(sign.detach()), total_loss=float(loss.detach()), grad_norm=float(grad))) + '\n')
    model.eval()
    return time.perf_counter() - start


def cpu_state(model):
    return {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}


def states_equal(a, b):
    return set(a) == set(b) and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def load_r0(seed, stats, device, root=None):
    path = (root or FORMAL) / f'seed_{seed}/results/R0/selected.pt'
    ckpt = load_checkpoint(path, map_location=device)
    model = screen.build_model('B4', stats).to(device); model.load_state_dict(ckpt['state_dict']); model.eval()
    return model, ckpt, path


def frozen_protocol():
    """Return (protocol, sha) only if frozen and authorized by the author for that exact hash."""
    digest = sha(PROTOCOL)
    assert not VERIFY_HASHES or PROTOCOL_SHA_FILE.read_text(encoding='utf-8').split()[0] == digest, 'protocol changed after freeze'
    protocol = read_json(PROTOCOL)
    assert protocol['status'].startswith('FROZEN'), 'protocol not frozen'
    for name, h in protocol['code_sha256'].items():
        assert not VERIFY_HASHES or sha(HERE / name) == h, f'code changed after freeze: {name}'
    auth = read_json(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    return protocol, digest
