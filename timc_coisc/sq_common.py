"""Shared pieces for the pre-registered sequence-sensitivity study (SEQ_PROTOCOL).

COISC_seq extends the COISC penalty from a pulse at the first planned step to pulses at every step j = 0..11 of the
120-s plan. For a pulse at step j, the response at lead time t_i = 10 i s has delay d = t_i - 10 j; the lag masks of
COISC are applied to d (pre: 0 < d < tau - 10, post: d > tau + 10). With STEPS = (0,) every function below reduces to
the frozen COISC code path, which gates G1-G2 verify bitwise. Everything else (head-only, 75 steps, lr, data, lags,
loss-ratio rule for lambda) is unchanged.
"""
from __future__ import annotations
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from _paths import PKG as HERE, PROTOCOLS, RESULTS, OUTPUT_ROOT, VERIFY_HASHES, load_checkpoint, private_path  # noqa: E402,F401
import onset_common as oc  # noqa: E402  (sets CUBLAS_WORKSPACE_CONFIG)
from onset_common import screen, torch  # noqa: E402

OUT = OUTPUT_ROOT / 'sequence'
PROTO = PROTOCOLS / 'sequence'
SCREEN = OUTPUT_ROOT / 'screening'
SCREEN_PROTO = PROTOCOLS / 'screening'
SCREEN_STUDY = SCREEN / 'study'
CL = OUTPUT_ROOT / 'closedloop'
CL_PROTO = PROTOCOLS / 'closedloop'
CL_STUDY = CL / 'study'
ONSET = OUTPUT_ROOT / 'posthoc_onset_aligned'

DRAFT = PROTO / 'SEQ_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'SEQ_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'SEQ_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
STUDY = OUT / 'study'
SMOKE = OUT / 'smoke'
GATES = OUT / 'gates'

NEW_SEEDS = tuple(range(20261101, 20261141))
VARIANT = 'COISC_seq'
SEQ_STEPS = tuple(range(12))
REPRO_SEED = 20261001
SMOKE_SEED = 1
CODE = ('sq_common.py', 'sq_train.py', 'sq_eval.py', 'sq_mil.py', 'sq_analyze.py', 'sq_freeze.py')
PYTHON = sys.executable
LAMBDA_FILE = PROTO / 'LAMBDA_SEQ.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=float), encoding='utf-8')


def utc():
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def code_hashes(names=CODE):
    return {n: sha(HERE / n) for n in names}


def seq_masks(lags, steps):
    """Boolean (J, 12, 7) pre-lag and post-lag masks on the delay d = 10 i - 10 j."""
    t = np.arange(1, 13)[None, :, None] * 10
    j = np.asarray(steps)[:, None, None] * 10
    tau = np.asarray(lags)[None, None, :]
    d = t - j
    return (d > 0) & (d < np.maximum(0, tau - 10)), d > tau + 10


def seq_sensitivity(model, history, future, steps):
    """(B, J, 12, 7): central difference of the rollout to a +-25 raw (5 % full-scale) pulse of u_h at plan step j."""
    b, n = history.shape[0], len(steps)
    h = history.repeat_interleave(n, dim=0) if n > 1 else history
    base = future.repeat_interleave(n, dim=0) if n > 1 else future
    idx = torch.as_tensor(np.tile(np.asarray(steps), b), device=future.device)
    rows = torch.arange(b * n, device=future.device)
    plus, minus = base.clone(), base.clone()
    plus[rows, idx, 1] = (base[rows, idx, 1] + 25.).clamp(0., 500.)
    minus[rows, idx, 1] = (base[rows, idx, 1] - 25.).clamp(0., 500.)
    denominator = (plus[rows, idx, 1] - minus[rows, idx, 1]) / 500.
    if torch.any(denominator <= 0):
        raise ValueError('Invalid clipped UH perturbation denominator')
    values = (screen.rollout(model, h, plus) - screen.rollout(model, h, minus)) / denominator[:, None, None]
    return values.reshape(b, n, *values.shape[1:])


def seq_regularizer(values, lags, steps):
    pre, post = [torch.as_tensor(m, device=values.device) for m in seq_masks(lags, steps)]
    return (torch.relu(values[:, pre].abs() - screen.EPS_PRE).square().mean(),
            torch.relu(-values[:, post]).square().mean())


def calibrate_seq(train, stats, probs, stages, device, steps):
    """onset_common.calibrate_both (prior lags) with the sequence penalty; same pilots, probes and loss-ratio rule."""
    start = time.perf_counter(); records = []
    for excitation in (False, True):
        screen.seed_all(oc.CAL_SEED); rng = np.random.default_rng(oc.CAL_SEED)
        model = screen.build_model('B4', stats).to(device).train()
        opt = torch.optim.Adam(model.parameters(), lr=stages[0]['lr'])
        for _ in range(screen.PILOT_STEPS):
            h, u, y = screen.sample(train, rng, stages[0]['batch'], probs[excitation], device)
            loss = screen.factual(model, h, u, y, stages[0]); opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True); opt.step()
        for k in range(8):
            h, u, y = screen.sample(train, rng, stages[0]['batch'], probs[excitation], device)
            prediction = screen.factual(model, h, u, y, stages[0])
            values = seq_sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET], steps)
            pre, sign = seq_regularizer(values, oc.TAU_PRIOR, steps); reg = pre + sign
            grads = torch.autograd.grad(reg, tuple(model.parameters()), allow_unused=True)
            gradnorm = float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
            if not gradnorm > 0: raise ValueError('sequence regularizer has no effective gradient')
            records.append(dict(excitation=excitation, batch=k, prediction=float(prediction.detach()),
                                regularizer=float(reg.detach()), pre=float(pre.detach()), sign=float(sign.detach()),
                                gradient_norm=gradnorm))
    groups = [screen.calibrated_lambda([r for r in records if r['excitation'] == e]) for e in (False, True)]
    return dict(lambda_s=min(groups), per_sampler=groups, records=records, steps=list(steps), time_s=time.perf_counter() - start)


def adapt_seq(model, train, probs, old, lam, lags, seed, device, log_path, steps, n_steps=oc.ADAPT_STEPS):
    """onset_common.adapt with the sequence penalty (w_pre = w_sign = lam); head-only, same optimiser and sampler."""
    screen.seed_all(seed)
    model.train()
    for name, param in model.named_parameters():
        param.requires_grad_(name.startswith('head.'))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=5e-5)
    rng = np.random.default_rng(seed); stage = dict(old['training']['stages'][2]); start = time.perf_counter()
    with Path(log_path).open('w', encoding='utf-8') as log:
        for step in range(1, n_steps + 1):
            h, u, y = screen.sample(train, rng, 128, probs[False], device)
            factual = screen.factual(model, h, u, y, stage)
            sens = seq_sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET], steps); pre, sign = seq_regularizer(sens, lags, steps)
            loss = factual + lam * pre + lam * sign
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
            log.write(json.dumps(dict(step=step, prediction_loss=float(factual.detach()), L_pre=float(pre.detach()),
                                      L_sign=float(sign.detach()), total_loss=float(loss.detach()), grad_norm=float(grad))) + '\n')
    model.eval()
    return time.perf_counter() - start


def screening_bindings():
    """{(seed, variant): binding} for R0/COISC/L0 of the 40 screening seeds."""
    out = {}
    for p in sorted(SCREEN_STUDY.glob('seed_*/SEED_COMPLETE.json')):
        for b in read(p)['bindings']:
            if b['variant'] in ('R0', 'COISC', 'L0'):
                out[(int(b['seed']), b['variant'])] = dict(path=b['path'], sha256=b['sha256'])
    return out


def seq_bindings(root=STUDY):
    out = {}
    for p in sorted(Path(root).glob('seed_*/SEED_COMPLETE.json')):
        r = read(p)
        out[(int(r['seed']), VARIANT)] = dict(path=r['path'], sha256=r['sha256'])
    return out


def frozen_protocol():
    """Return (protocol, sha) only if frozen, code and bound files unchanged, and authorized for that exact hash."""
    digest = sha(PROTOCOL)
    assert not VERIFY_HASHES or PROTOCOL_SHA_FILE.read_text(encoding='utf-8').split()[0] == digest, 'protocol changed after freeze'
    protocol = read(PROTOCOL)
    assert protocol['status'].startswith('FROZEN'), 'protocol not frozen'
    for name, h in protocol['code_sha256'].items():
        assert not VERIFY_HASHES or sha(HERE / name) == h, f'code changed after freeze: {name}'
    for b in protocol['bound_files']:
        assert not VERIFY_HASHES or sha(private_path(b['path'])) == b['sha256'], f'bound file changed after freeze: {b["path"]}'
    auth = read(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    return protocol, digest
