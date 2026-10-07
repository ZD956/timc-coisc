"""Population-only binding for the frozen secondary diagnostic implementation."""
import inspect
import types

import delivered23_evaluate as adapted
import frozen_diagnostics as original

_source = inspect.getsource(original.diagnostics_for_pairs)
assert _source.count('from frozen_evaluate import SEEDS,validate_pairs') == 1
_source = _source.replace('from frozen_evaluate import SEEDS,validate_pairs',
                          'from delivered23_evaluate import SEEDS,validate_pairs')
assert _source.count('all29') == 2
_source = _source.replace('all29', 'all23')
_namespace = vars(original).copy()
exec(compile(_source, __file__ + '::population_binding', 'exec'), _namespace)
diagnostics_for_pairs = _namespace['diagnostics_for_pairs']
regime_rows = original.regime_rows
event_rows = original.event_rows
free_run_rows = original.free_run_rows
paired_inference = original.paired_inference
inference_timing = original.inference_timing
model_predictor = original.model_predictor
