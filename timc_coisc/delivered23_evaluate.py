"""Thin population adapter. Frozen statistical functions remain unchanged.

Only 29/23/6 population literals and cohort labels in the inherited orchestration
are bound to 23/17/6. No metric, gate, median, CI or loss expression is changed.
"""
import hashlib
import inspect
from pathlib import Path
import sys
import types
import numpy as np

import os
import frozen_evaluate as frozen

EXPECTED_FROZEN_SHA='63b9350249f560a24fcf1466094696513166aeaf2df712eeca3d68d6b7773448'
if os.environ.get('TIMC_VERIFY_HASHES')=='1' and hashlib.sha256(Path(frozen.__file__).read_bytes()).hexdigest()!=EXPECTED_FROZEN_SHA:
    raise RuntimeError('INTEGRITY_FAILURE: original evaluator changed')

SEEDS=frozen.SEEDS
POINTS=frozen.POINTS
HORIZONS=frozen.HORIZONS


def membership_order(membership):
    if len(membership)!=23: raise ValueError('expected all23 locked sources')
    ids=[r['source_id'] for r in membership]
    if len(set(ids))!=23 or any(not isinstance(s,str) or not s for s in ids):
        raise ValueError('duplicate or invalid source identity')
    ext=sorted(r['source_id'] for r in membership if r['role']=='frozen_external')
    final=sorted(r['source_id'] for r in membership if r['role']=='frozen_final')
    if len(ext)!=17 or len(final)!=6: raise ValueError('locked17/6 roles/counts mismatch')
    return ext+final


def make_draw_table(membership):
    order=membership_order(membership)
    rng=np.random.default_rng(frozen.BOOTSTRAP_SEED)
    indices=np.c_[rng.integers(0,17,size=(frozen.REPLICATES,17)),rng.integers(17,23,size=(frozen.REPLICATES,6))].astype('<i8')
    counts=np.zeros((frozen.REPLICATES,23),dtype=np.int64)
    np.add.at(counts,(np.repeat(np.arange(frozen.REPLICATES),23),indices.ravel()),1)
    return dict(source_order=order,indices=indices,counts=counts,
        sha256=hashlib.sha256(indices.tobytes(order='C')).hexdigest(),seed=frozen.BOOTSTRAP_SEED)


# Clone globals, leaving original module and files intact.
_ns=vars(frozen).copy()
_ns.update(membership_order=membership_order,make_draw_table=make_draw_table)
validate_pairs=types.FunctionType(frozen.validate_pairs.__code__,_ns,'validate_pairs',frozen.validate_pairs.__defaults__)
_ns['validate_pairs']=validate_pairs
_source=inspect.getsource(frozen._evaluate_valid)
_old="cohorts={'all29':np.arange(29),'frozen_external':np.arange(23),'frozen_final':np.arange(23,29)}"
_new="cohorts={'all23':np.arange(23),'frozen_external':np.arange(17),'frozen_final':np.arange(17,23)}"
assert _source.count(_old)==1
_source=_source.replace(_old,_new)
assert _source.count('n_sources=29')==1
_source=_source.replace('n_sources=29','n_sources=23').replace('all29','all23')
exec(compile(_source,__file__+'::population_binding','exec'),_ns)
evaluate=types.FunctionType(frozen.evaluate.__code__,_ns,'evaluate',frozen.evaluate.__defaults__)
ADAPTED_ORCHESTRATION_SHA256=hashlib.sha256(_source.encode()).hexdigest()
# Direct aliases, not reimplementations.
SourceMedian=frozen.SourceMedian
weighted_psm=frozen.weighted_psm
source_summary=frozen.source_summary
decide_gates=frozen.decide_gates
overall_decision=frozen.overall_decision
percentile_ci=frozen.percentile_ci
draw_means=frozen.draw_means
effect_row=frozen.effect_row
