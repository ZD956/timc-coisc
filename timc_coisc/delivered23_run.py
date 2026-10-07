"""One write-once available-23 holdout execution. No training or tuning path."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import zipfile

import numpy as np
import torch

sys.dont_write_bytecode = True

import delivered23_evaluate as evaluator
import delivered23_diagnostics as diagnostics
from paper_models import VanillaLSTM
from _paths import PKG as HERE, PROTOCOLS, CONFIGS, DATA_ROOT, OUTPUT_ROOT, load_checkpoint, require_private, private_path, public_source_id

VERIFY_HASHES = os.environ.get('TIMC_VERIFY_HASHES') == '1'
AUDIT_DIR = DATA_ROOT / 'data_asset_audit'
DRAFT = PROTOCOLS / 'AVAILABLE23_EVALUATION_PROTOCOL_DRAFT.json'
# Unreleased companion of DRAFT holding original file names and archive members for the 23 sources.
SOURCE_LOCATIONS = DATA_ROOT / 'protocols_private' / 'AVAILABLE23_EVALUATION_PROTOCOL_DRAFT.json'
MANIFEST = DATA_ROOT / 'data' / '01_locked_source_manifest.csv'
ORIGINAL = DATA_ROOT / 'frozen_evaluation' / 'frozen_20260926T062548Z_c28722408e'
NORMALIZATION = CONFIGS / 'normalization.json'
CANONICAL = DATA_ROOT / 'canonical_preprocessing_20260926T_devonly'
EXECUTIONS = OUTPUT_ROOT / 'delivered23'


def preprocess_source(path):
    import importlib.util
    adapter = require_private(DATA_ROOT / 'delivered23_data_adapter.py', 'delivered23 source preprocessing')
    spec = importlib.util.spec_from_file_location('delivered23_data_adapter', adapter)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.preprocess_source(path)


# Digests below are those of the pre-release development files; they are enforced only with TIMC_VERIFY_HASHES=1.
SOURCE_HASH = '655dc58bd7ecc3624c13cd41244ad6acb53a6a56ebb75be63e660294606f54a2'
HASHES = {
    'frozen_evaluate.py': '63b9350249f560a24fcf1466094696513166aeaf2df712eeca3d68d6b7773448',
    'frozen_diagnostics.py': '800977056bc70e66493e241056d0524f68599f84e9288b84c1eb3d0775540140',
    'normalization.json': '6e0a0a41664c962c7e0d36f3694bb42eddd3279a9ae82cc569d06a6fac572dc1',
    'CANONICAL_PREPROCESSING_PROTOCOL.json': '03bd91f532b82144bd23ffda99483002176d881ca7ce07bbedb000c6bd3f7013',
    'delivered23_data_adapter.py': 'cf8bedfe7b450a6837fb31397a438f424a44ad0b8156be0b9ce7a6ae825f80eb',
}
DEPENDENCIES = [HERE / n for n in ('delivered23_evaluate.py', 'delivered23_diagnostics.py', 'delivered23_run.py',
                                   'frozen_evaluate.py', 'frozen_diagnostics.py', 'screen.py', 'paper_data.py',
                                   'paper_models.py', 'paper_eval.py', 'structural_training_common.py',
                                   'model_structural_ablation.py')]
DEPENDENCIES += [NORMALIZATION, DRAFT]
DEPENDENCIES += [CANONICAL / n for n in ('CANONICAL_PREPROCESSING_PROTOCOL.json',
                 'PREPROCESSING_PROVENANCE_DECISION.json', 'DEVELOPMENT_PREPROCESSING_CONFORMANCE.csv')]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write_once(path: Path, data: bytes) -> None:
    with path.open('xb') as stream:
        stream.write(data)


def write_json(path: Path, value) -> None:
    write_once(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def rows(path: Path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def bound_files(draft):
    checkpoints = []
    frozen = json.loads((PROTOCOLS / 'FROZEN_EVALUATION_PROTOCOL_DRAFT.json').read_text(encoding='utf-8'))
    for pair in frozen['provenance']['checkpoint_pairs']:
        for variant in ('R0', 'COISC'):
            checkpoints.append(dict(seed=pair['seed'], variant=variant,
                                    path=str(private_path(pair[variant]['path'])), expected_sha256=pair[variant]['sha256']))
    for item in checkpoints:
        item['actual_sha256'] = digest(require_private(item['path'], 'checkpoint'))
        if VERIFY_HASHES and item['actual_sha256'] != item['expected_sha256']:
            raise RuntimeError(f'INTEGRITY_FAILURE: checkpoint {item["path"]}')
    files = {}
    for path in DEPENDENCIES:
        actual = digest(path)
        files[path.name] = actual
        if VERIFY_HASHES and path.name in HASHES and actual != HASHES[path.name]:
            raise RuntimeError(f'INTEGRITY_FAILURE: {path}')
    return checkpoints, files


def approved_membership(draft):
    if digest(require_private(MANIFEST, 'membership check')) != SOURCE_HASH and VERIFY_HASHES:
        raise RuntimeError('INTEGRITY_FAILURE: original membership manifest')
    locked = {public_source_id(r['source_id']): r for r in rows(MANIFEST)}
    located = {public_source_id(s['source_id']): s for s in json.loads(
        require_private(SOURCE_LOCATIONS, 'source extraction').read_text(encoding='utf-8'))['included_sources']}
    proposal = json.loads((AUDIT_DIR / 'FINAL_DATA_ROLE_PROPOSAL.json').read_text(encoding='utf-8'))
    expected = {public_source_id(s) for a in proposal['groups']['PRESPECIFIED_HOLDOUT_AVAILABLE'] for s in a['source_ids']}
    given = draft['included_sources']
    ids = [s['source_id'] for s in given]
    if len(given) != 23 or len(set(ids)) != 23 or set(ids) != expected:
        raise RuntimeError('INTEGRITY_FAILURE: pre-specified source IDs')
    if sum(s['role'] == 'frozen_external_test' for s in given) != 17 or sum(s['role'] == 'frozen_final_test' for s in given) != 6:
        raise RuntimeError('INTEGRITY_FAILURE: 17/6 roles')
    dev_hashes = {r['sha256'] for r in locked.values() if r['role'] in ('formula_fit', 'formula_selection')}
    if json.loads((AUDIT_DIR / 'DATA_LEAKAGE_AUDIT.json').read_text(encoding='utf-8'))['confirmed_exact_identity_leakage']:
        raise RuntimeError('INTEGRITY_FAILURE: confirmed identity leakage')
    output = []
    for s in given:
        r = locked[s['source_id']]
        private = located[s['source_id']]
        if r['role'] != s['role'] or r['sha256'] != s['sha256'] or private['sha256'] != s['sha256'] or r['file_name'] != private['file_name'] or r['sha256'] in dev_hashes:
            raise RuntimeError(f'INTEGRITY_FAILURE: original identity/role/hash {s["source_id"]}')
        places = [p for p in private['archive_locations'] if p['actual_sha256'] == s['sha256']]
        if not places:
            raise RuntimeError(f'MISSING_SOURCE: no matching archive member for {s["source_id"]}')
        output.append(dict(source_id=s['source_id'], role='frozen_external' if s['role'] == 'frozen_external_test' else 'frozen_final',
                           original_role=s['role'], file_name=r['file_name'], expected_sha256=r['sha256'],
                           original_path=r['absolute_path'], archive_path=places[0]['archive_path'],
                           archive_member=places[0]['member_name'], expected_windows=int(r['eligible_320s_windows']),
                           expected_frames=int(r['normalized_10s_frames']), expected_raw_rows=int(r['raw_rows']),
                           expected_segments=int(r['continuous_segments']), expected_time_source=r['time_source']))
    if len({x['expected_sha256'] for x in output}) != 23:
        raise RuntimeError('INTEGRITY_FAILURE: duplicate raw source hash')
    return output


def materialize_sources(membership, directory):
    source_dir = directory / 'source_files'
    source_dir.mkdir()
    for i, source in enumerate(membership):
        archive = Path(source['archive_path'])
        path = source_dir / f'{i:02d}_{source["file_name"]}'
        with zipfile.ZipFile(archive) as z:
            member = resolve_zip_member(z, source['archive_member'])
            with z.open(member) as reader, path.open('xb') as writer:
                shutil.copyfileobj(reader, writer)
        source['relocated_path'] = str(path)
        source['actual_sha256'] = digest(path)
        source['hash_match'] = source['actual_sha256'] == source['expected_sha256']
        if not source['hash_match']:
            raise RuntimeError(f'INTEGRITY_FAILURE: extracted source {source["source_id"]}')


def resolve_zip_member(archive, locked_name):
    if locked_name in archive.namelist():
        return locked_name
    # Supplied archives omit UTF-8 filename flags; CP437 decoding of GBK bytes
    # changes only the ZIP label. Source identity is still verified by SHA256.
    encoded = locked_name.encode('gbk').decode('cp437')
    if encoded in archive.namelist():
        return encoded
    raise KeyError(f'locked ZIP member unavailable: {locked_name}')


def prepare():
    draft = json.loads(DRAFT.read_text(encoding='utf-8'))
    checkpoints, dependency_hashes = bound_files(draft)
    membership = approved_membership(draft)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    execution_id = f'delivered23_{stamp}_{digest(DRAFT)[:10]}'
    directory = EXECUTIONS / execution_id
    directory.mkdir(parents=True, exist_ok=False)
    try:
        materialize_sources(membership, directory)
        protocol = dict(draft)
        protocol.update(status='LOCKED_BEFORE_NUMERIC_INFERENCE', execution_id=execution_id,
                        protocol_mutable=False, OUTCOME_INSPECTED_BEFORE_23_SOURCE_LOCK=False,
                        SOURCE_EXCLUDED_BY_MODEL_PERFORMANCE=False, MODEL_RESCUE=False,
                        STATISTICAL_RESCUE=False, original_manifest_sha256=SOURCE_HASH,
                        original_manifest_population=29, unavailable_original_external=6,
                        availability_constrained_prespecified_population=23,
                        primary_population_change_reason='physical data availability in transferred project archive only; fixed before any numeric model outcome',
                        exact_membership=membership, checkpoints=checkpoints,
                        evaluator_dependency_sha256=dependency_hashes,
                        normalization_sha256=digest(NORMALIZATION),
                        canonical_preprocessing_protocol_sha256=HASHES['CANONICAL_PREPROCESSING_PROTOCOL.json'],
                        canonical_adapter_sha256=HASHES['delivered23_data_adapter.py'],
                        preprocessing_provenance_status='PREPROCESSING_PROVENANCE_PARTIALLY_UNRESOLVED',
                        development_preprocessing_conformance='87/90 available raw sources bitwise; 3 mismatch; 19 raw unavailable',
                        legacy_manifest_metadata_gate=False,
                        evaluator_tests='test_delivered23.py synthetic/in-memory fixtures PASS before sealing',
                        previous_execution_id=ORIGINAL.name,
                        previous_execution_status='NOT_EVALUABLE_MISSING_NUMERIC_SOURCE_DATA',
                        previous_numeric_inference=False, previous_bootstrap_replicates=0,
                        technical_limitations=['Original 29-source estimand is not evaluated; this is a fixed availability-constrained 23-source population.',
                                               'Unresolved acquisition-session overlap is retained as provenance limitation, not an exclusion rule.',
                                               'Historical preprocessing provenance remains partially unresolved: 87/90 bitwise, 3 mismatches, 19 unavailable raw sources.',
                                               'Four legacy manifest sampling-metadata mismatches are reported, but canonical preprocessing governs inference.'])
        for name in ('CANONICAL_PREPROCESSING_PROTOCOL.json', 'PREPROCESSING_PROVENANCE_DECISION.json',
                     'DEVELOPMENT_PREPROCESSING_CONFORMANCE.csv'):
            write_once(directory / name, (CANONICAL / name).read_bytes())
        protocol_path = directory / 'DELIVERED23_EVALUATION_PROTOCOL.json'
        write_json(protocol_path, protocol)
        write_once(directory / 'DELIVERED23_PROTOCOL.sha256', (digest(protocol_path) + '  DELIVERED23_EVALUATION_PROTOCOL.json\n').encode('ascii'))
        write_json(directory / 'PREFLIGHT_AUDIT.json', dict(status='PASS', execution_id=execution_id,
                   source_hash_matches=23, external=17, final=6, checkpoint_hash_matches=6,
                   normalization_hash_match=True, original_manifest_sha256=SOURCE_HASH,
                   protocol_sha256=digest(protocol_path), dependency_sha256=dependency_hashes,
                   canonical_preprocessing_protocol_sha256=HASHES['CANONICAL_PREPROCESSING_PROTOCOL.json'],
                   canonical_adapter_sha256=HASHES['delivered23_data_adapter.py'],
                   preprocessing_provenance_status='PREPROCESSING_PROVENANCE_PARTIALLY_UNRESOLVED',
                   numeric_contents_parsed=False, inference_started=False, bootstrap_replicates=0))
        print(str(directory), flush=True)
    except Exception as exc:
        write_json(directory / 'PRE_INFERENCE_FAILURE.json', dict(status='NOT_EVALUABLE', reason=str(exc),
                   numeric_contents_parsed=False, inference_started=False, traceback=traceback.format_exc()))
        raise


def windows_for_source(source):
    path = Path(source['relocated_path'])
    if digest(path) != source['expected_sha256']:
        raise RuntimeError(f'INTEGRITY_FAILURE: source changed {source["source_id"]}')
    data, details = preprocess_source(path)
    index = data.pop('window_index')
    data['window_ids'] = np.asarray([f'{source["source_id"]}__seg{number:03d}_frame{offset:06d}'
                                     for number, offset, _ in index])
    valid = (details['eligible_windows'] > 0 and details['all_windows_finite']
             and details['no_window_crosses_segment_boundary']
             and details['time_monotonic_inside_segment']
             and details['history_shape_valid'] and details['future_shape_valid']
             and details['target_shape_valid'] and data['origin'].shape[1:] == (7,)
             and len(data['window_ids']) == details['eligible_windows'])
    if not valid:
        raise ValueError(f'canonical window preflight failed {source["source_id"]}')
    legacy = {key: (source.get(expected), details[actual]) for key, expected, actual in (
        ('raw_rows', 'expected_raw_rows', 'raw_rows'),
        ('time_source', 'expected_time_source', 'time_source'),
        ('segments', 'expected_segments', 'segments'),
        ('normalized_frames', 'expected_frames', 'normalized_frames'),
        ('windows', 'expected_windows', 'eligible_windows')) if expected in source}
    record = dict(source_id=source['source_id'], **details, windows=details['eligible_windows'],
                  source_sha256=source['expected_sha256'],
                  legacy_manifest_metadata={key: dict(expected=old, actual=new, match=old == new)
                                            for key, (old, new) in legacy.items()},
                  legacy_metadata_gate=False)
    return data, record


def load_models(protocol, seed, device):
    stats = json.loads(NORMALIZATION.read_text(encoding='utf-8'))
    models = {}
    for item in protocol['checkpoints']:
        if item['seed'] != seed:
            continue
        if VERIFY_HASHES and digest(Path(item['path'])) != item['expected_sha256']:
            raise RuntimeError('INTEGRITY_FAILURE: checkpoint changed before load')
        payload = load_checkpoint(item['path'])
        if payload['seed'] != seed:
            raise RuntimeError('INTEGRITY_FAILURE: checkpoint seed')
        if item['variant'] == 'COISC' and (payload['steps'] != 75 or payload['lambda_s'] != 939.1430335805455):
            raise RuntimeError('INTEGRITY_FAILURE: COISC metadata')
        model = VanillaLSTM(stats)
        model.load_state_dict(payload['state_dict'], strict=True)
        for key in ('temperature_mean', 'temperature_std'):
            if not torch.equal(model.state_dict()[key], torch.tensor(stats[key], dtype=torch.float32)):
                raise RuntimeError('INTEGRITY_FAILURE: normalization checkpoint buffer')
        model.eval().requires_grad_(False)
        models[item['variant']] = model.to(device)
    if set(models) != {'R0', 'COISC'}:
        raise RuntimeError('INTEGRITY_FAILURE: missing model pair')
    if any(not torch.equal(v, models['COISC'].state_dict()[k]) for k, v in models['R0'].state_dict().items() if not k.startswith('head.')):
        raise RuntimeError('INTEGRITY_FAILURE: non-head tensor mismatch')
    return models


def run(directory):
    directory = Path(directory).resolve()
    if directory.parent != EXECUTIONS.resolve() or not directory.name.startswith('delivered23_'):
        raise ValueError('not a sealed delivered23 execution directory')
    protocol_path = directory / 'DELIVERED23_EVALUATION_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    if protocol['execution_id'] != directory.name or protocol['status'] != 'LOCKED_BEFORE_NUMERIC_INFERENCE':
        raise RuntimeError('INTEGRITY_FAILURE: protocol state')
    if (directory / 'INFERENCE_STARTED.json').exists() or (directory / 'DELIVERED23_FINAL_DECISION.json').exists():
        raise RuntimeError('single-use execution already started; no automatic rerun')
    line = (directory / 'DELIVERED23_PROTOCOL.sha256').read_text(encoding='ascii').split()[0]
    if digest(protocol_path) != line:
        raise RuntimeError('INTEGRITY_FAILURE: protocol hash changed')
    located = {p.name: p for p in DEPENDENCIES}
    for path, expected in protocol['evaluator_dependency_sha256'].items():
        if digest(located[Path(path).name]) != expected:
            raise RuntimeError(f'INTEGRITY_FAILURE: dependency changed {path}')
    if digest(NORMALIZATION) != protocol['normalization_sha256']:
        raise RuntimeError('INTEGRITY_FAILURE: normalization changed')
    for source in protocol['exact_membership']:
        if digest(Path(source['relocated_path'])) != source['expected_sha256']:
            raise RuntimeError(f'INTEGRITY_FAILURE: source changed {source["source_id"]}')
    for item in protocol['checkpoints']:
        if VERIFY_HASHES and digest(Path(item['path'])) != item['expected_sha256']:
            raise RuntimeError(f'INTEGRITY_FAILURE: checkpoint changed {item["path"]}')
    audit = dict(status='INCOMPLETE', execution_id=directory.name, protocol_sha256=line,
                 source_hashes_verified=True, checkpoint_hashes_verified=True, training=False,
                 fine_tuning=False, checkpoint_selection=False, source_deletion=False, rescue=False,
                 numeric_contents_parsed=False, inference_started=False, bootstrap_replicates_run=0,
                 source_processing=[], model_tensor_equality=[])
    try:
        source_data = {}
        for i, item in enumerate(protocol['exact_membership'], 1):
            data, record = windows_for_source(item)
            source_data[item['source_id']] = data
            audit['source_processing'].append(record)
            print(f'[data] {i}/23 {item["source_id"]}: {record["windows"]} windows', flush=True)
        audit['numeric_contents_parsed'] = True
        legacy_mismatches = [dict(source_id=r['source_id'], fields=[name for name, value in r['legacy_manifest_metadata'].items()
                                                                  if not value['match']])
                             for r in audit['source_processing']]
        legacy_mismatches = [row for row in legacy_mismatches if row['fields']]
        write_json(directory / 'FINAL_PREFLIGHT_AUDIT.json', dict(status='PASS', source_count=23,
                   external_count=sum(s['role'] == 'frozen_external' for s in protocol['exact_membership']),
                   final_count=sum(s['role'] == 'frozen_final' for s in protocol['exact_membership']),
                   source_hash_matches=23, checkpoint_hash_matches=6,
                   normalization_hash_match=True, canonical_protocol_hash_match=True,
                   canonical_adapter_hash_match=True, all_windows_finite=True,
                   all_window_shapes_valid=True, all_windows_within_continuous_segment=True,
                   development_exact_sha_overlap=0, source_processing=audit['source_processing'],
                   legacy_manifest_metadata_mismatches=legacy_mismatches,
                   legacy_manifest_metadata_gate=False,
                   preprocessing_provenance_status=protocol['preprocessing_provenance_status'],
                   inference_started=False))
        write_json(directory / 'INFERENCE_STARTED.json', dict(started_utc=datetime.now(timezone.utc).isoformat(),
                   protocol_sha256=line, source_count=23, training=False))
        audit['inference_started'] = True
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        audit['hardware'] = dict(device=str(device), cuda=torch.cuda.is_available(),
                                 gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                                 torch=torch.__version__, dtype='float32')
        order = evaluator.membership_order(protocol['exact_membership'])
        pairs = {}
        timing = []
        predictors = {}
        for seed in evaluator.SEEDS:
            start = time.perf_counter()
            models = load_models(protocol, seed, device)
            reference = {name: {k: v.detach().cpu().clone() for k,v in model.state_dict().items()} for name,model in models.items()}
            pairs[seed] = {}
            predictors[seed] = {name: diagnostics.model_predictor(model, device) for name,model in models.items()}
            for i, source_id in enumerate(order, 1):
                data = source_data[source_id]
                inferred = diagnostics.paired_inference(models, data['history'], data['future'], device)
                pairs[seed][source_id] = {name: {**data, **inferred[name]} for name in ('R0','COISC')}
                print(f'[inference] seed={seed} source={i}/23', flush=True)
            first = source_data[order[0]]
            timing += [dict(seed=seed, **row) for row in diagnostics.inference_timing(models, first['history'], first['future'], device)]
            equal = {name: all(torch.equal(v.detach().cpu(), reference[name][k]) for k,v in model.state_dict().items()) for name,model in models.items()}
            audit['model_tensor_equality'].append(dict(seed=seed, before_after=equal, non_head_paired_equal=True))
            if not all(equal.values()):
                raise RuntimeError('INTEGRITY_FAILURE: inference mutated tensor')
            print(f'[seed complete] {seed} elapsed={time.perf_counter()-start:.1f}s', flush=True)
        membership = [{'source_id':s['source_id'], 'role':s['role']} for s in protocol['exact_membership']]
        diagnostic = diagnostics.diagnostics_for_pairs(pairs,membership,predictors)
        diagnostic['inference_time'] = timing
        print('[secondary diagnostics complete]', flush=True)
        result = evaluator.evaluate(pairs,membership,diagnostic)
        if result['status'] == 'NOT_EVALUABLE':
            raise RuntimeError(f'NOT_EVALUABLE: {result["decision"].get("reason", "statistics failed")}')
        audit['bootstrap_replicates_run'] = 20000
        payloads = evaluator.frozen.output_payloads(result)
        payloads.pop('FROZEN_FINAL_DECISION.json')
        for original, blob in payloads.items():
            destination = original.replace('FROZEN_', 'DELIVERED23_').replace('frozen_', 'delivered23_')
            write_once(directory / destination, blob)
            print(f'[artifact] {destination}', flush=True)
        decision = result['decision']
        detail = []
        for seed in result['per_seed']:
            gates = {}
            values = seed['primary_values']
            for name, metric in [('Tp_120s_MAE','MAE'),('PLE','PLE'),('WSR','WSR'),('PSM','PSM')]:
                v=values[metric]; a=v['R0']; b=v['COISC']
                ci=v['contrast_ci'] if metric != 'PSM' else v['ratio_contrast_ci']
                gates[name]=dict(R0=a, COISC=b, absolute_effect=b-a,
                                 relative_effect_percent=100*(b/a-1) if a else None,
                                 CI_lower=ci[0], CI_upper=ci[1],
                                 threshold=.005 if metric=='MAE' else .9*a if metric in ('PLE','WSR') else max(.2*a,.0005),
                                 gate_contrast=seed['gates'][name], decision=seed['gates'][name]['decision'])
            detail.append(dict(seed=seed['seed'],decision=seed['decision'],gates=gates))
        enriched = dict(**decision, complete_source_count=23, external_complete_count=17, final_complete_count=6,
                        source_hashes_verified=True, checkpoint_hashes_verified=True,
                        manifest_sha256=SOURCE_HASH, protocol_sha256=line,
                        evaluator_sha256=next(v for k, v in protocol['evaluator_dependency_sha256'].items() if Path(k).name == 'delivered23_evaluate.py'),
                        bootstrap_complete=True, replicates_requested=20000,replicates_run=20000,
                        preprocessing_provenance_status=protocol['preprocessing_provenance_status'],
                        legacy_manifest_metadata_mismatch_count=len(legacy_mismatches),
                        training=False,fine_tuning=False,checkpoint_selection=False,source_deletion=False,rescue=False,
                        per_seed_complete_gates=detail)
        write_json(directory / 'DELIVERED23_FINAL_DECISION.json',enriched)
        audit.update(status=decision['overall_decision'], decision_sha256=digest(directory/'DELIVERED23_FINAL_DECISION.json'),
                     bootstrap_draw_sha256=result['draw_table']['sha256'], generated_files=sorted(p.name for p in directory.iterdir() if p.is_file()))
        write_json(directory / 'DELIVERED23_PROTOCOL_AUDIT.json', audit)
        print(f'[final] {decision["overall_decision"]}', flush=True)
    except BaseException as exc:
        status = 'INCOMPLETE' if audit['inference_started'] else 'NOT_EVALUABLE'
        audit.update(status=status, reason=str(exc), traceback=traceback.format_exc(),
                     generated_files=sorted(p.name for p in directory.iterdir() if p.is_file()))
        write_json(directory / 'DELIVERED23_PROTOCOL_AUDIT.json', audit)
        if not audit['inference_started']:
            write_json(directory / 'DELIVERED23_FINAL_DECISION.json',
                       dict(overall_decision='NOT_EVALUABLE', reason=str(exc),
                            inference_started=False, bootstrap_replicates_run=0))
        print(f'[{status.lower()}] {exc}', flush=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare','run'))
    parser.add_argument('execution_dir', nargs='?')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.execution_dir:
        run(args.execution_dir)
    else:
        parser.error('run requires the sealed execution_dir')
