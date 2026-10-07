"""Shared pieces for the post-hoc B5 Jacobian-norm baseline (POSTHOC_B5_JACOBIAN_PROTOCOL)."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np

from _paths import PKG as HERE, PROTOCOLS, OUTPUT_ROOT, VERIFY_HASHES  # noqa: E402
import onset_common as oc  # noqa: E402  (imports screen, which sets CUBLAS_WORKSPACE_CONFIG)
from onset_common import screen, torch  # noqa: E402

OUT = OUTPUT_ROOT / 'posthoc_jacobian_b5'
PROTO = PROTOCOLS / 'posthoc_jacobian_b5'
DRAFT = PROTO / 'POSTHOC_B5_JACOBIAN_PROTOCOL_DRAFT.json'
PROTOCOL = PROTO / 'POSTHOC_B5_JACOBIAN_PROTOCOL.json'
PROTOCOL_SHA_FILE = PROTO / 'POSTHOC_B5_JACOBIAN_PROTOCOL.sha256'
AUTHORIZATION = OUT / 'EXECUTION_AUTHORIZATION.json'
SEEDS = oc.E1_SEEDS


def jacobian_term(values):
    """Mean squared input sensitivity over all 12 x 7 forecast cells: no lag masks, no tolerance."""
    return values.square().mean()


def coisc_terms(w_pre, w_sign, lags):
    def terms(values):
        pre, sign = oc.regularizer(values, lags)
        return dict(L_pre=(w_pre, pre), L_sign=(w_sign, sign))
    return terms


def jacobian_terms(w_jac):
    def terms(values):
        return dict(L_jac=(w_jac, jacobian_term(values)))
    return terms


def calibrate(train, stats, probs, stages, device):
    """onset_common.calibrate_both pilots and probes, scoring the prior-mask COISC term and the Jacobian term."""
    start = time.perf_counter(); records = {'prior': [], 'jacobian': []}
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
            values = screen.sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET])
            pre, sign = oc.regularizer(values, oc.TAU_PRIOR)
            for name, reg, extra in (('prior', pre + sign, dict(pre=float(pre.detach()), sign=float(sign.detach()))),
                                     ('jacobian', jacobian_term(values), {})):
                grads = torch.autograd.grad(reg, tuple(model.parameters()), allow_unused=True, retain_graph=True)
                gradnorm = float(torch.sqrt(sum(g.square().sum() for g in grads if g is not None)))
                if not gradnorm > 0: raise ValueError(f'{name} regularizer has no effective gradient')
                records[name].append(dict(excitation=excitation, batch=k, prediction=float(prediction.detach()),
                                          regularizer=float(reg.detach()), **extra, gradient_norm=gradnorm))
    out = {}
    for name, rec in records.items():
        groups = [screen.calibrated_lambda([r for r in rec if r['excitation'] == e]) for e in (False, True)]
        out[name] = dict(lambda_s=min(groups), per_sampler=groups, records=rec)
    out['time_s'] = time.perf_counter() - start
    return out


def adapt(model, train, probs, old, terms_fn, seed, device, log_path, steps=oc.ADAPT_STEPS):
    """onset_common.adapt with the regularizer supplied as named (weight, value) terms added in order."""
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
            terms = terms_fn(screen.sensitivity(model, h[:screen.SUBSET], u[:screen.SUBSET]))
            loss = factual
            for w, v in terms.values():
                loss = loss + w * v
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True); optimizer.step()
            log.write(json.dumps(dict(step=step, prediction_loss=float(factual.detach()),
                                      **{k: float(v.detach()) for k, (w, v) in terms.items()},
                                      total_loss=float(loss.detach()), grad_norm=float(grad))) + '\n')
    model.eval()
    return time.perf_counter() - start


def frozen_protocol():
    """Return (protocol, sha) only if frozen, code unchanged, and authorized by the author for that exact hash."""
    digest = oc.sha(PROTOCOL)
    assert not VERIFY_HASHES or PROTOCOL_SHA_FILE.read_text(encoding='utf-8').split()[0] == digest, 'protocol changed after freeze'
    protocol = oc.read_json(PROTOCOL)
    assert protocol['status'].startswith('FROZEN'), 'protocol not frozen'
    for name, h in protocol['code_sha256'].items():
        assert not VERIFY_HASHES or oc.sha(HERE / name) == h, f'code changed after freeze: {name}'
    for b in protocol['shared_code_sha256']:
        assert not VERIFY_HASHES or oc.sha(oc.private_path(b['path'])) == b['sha256'], f'shared code changed after freeze: {b["path"]}'
    auth = oc.read_json(AUTHORIZATION)
    assert auth.get('authorized_by_author') is True and auth.get('protocol_sha256') == digest, 'no valid author authorization'
    return protocol, digest
