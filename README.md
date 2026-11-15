# COISC: staged evaluation of lag-aware sensitivity adaptation of a learned thermal surrogate

Analysis code, frozen evaluation protocols and summary results for the article

> J. Zhou and B. Liu, "Lag-aware sensitivity adaptation of an LSTM thermal surrogate for predictive control: A preregistered multi-seed evaluation", submitted to *Tehnički vjesnik – Technical Gazette*.

Version 1.1 adds the five preregistered studies of that article: the deployment screening on 40 new training seeds (stage S3), the closed-loop confirmation (S4), and three baselines on the S4 bench: a first-move rule (S5), a sequence-sensitivity variant of the penalty (S6) and the R0 controller with a larger move penalty (S7). It also adds the post-hoc trajectory analyses of Section 5.5. Version 1.0 accompanied an earlier manuscript on the same work ("Does lag-aware sensitivity adaptation of a learned thermal surrogate improve prediction or control? A staged evaluation"); its section and table numbers are kept in the descriptions below and refer to that manuscript. Nothing released in 1.0 was changed in 1.1.

The field data are proprietary and are not included. Trained model weights are not released either. Without the data, you can still inspect the code and protocols, run the data-free tests, recompute the S3-S7 decisions and the post-hoc numbers, and redraw every figure from the released summary results.

## Contents

```
timc_coisc/     method, evaluation and analysis code (one flat package)
protocols/      frozen protocols, locks and gate reports as used in the study (anonymised);
                the later post-hoc studies have one subdirectory each
configs/        global normalisation statistics
results/        summary results behind every table and figure (anonymised)
figures/        make_figures.py, which redraws Figures 1-7 from results/
ORIGINAL_HASHES.json  SHA-256 of the original, pre-anonymisation protocol files
RELEASE_SHA256.txt    SHA-256 of every file in this release
```

The code uses the paper's notation throughout. u_h (`uh`, `UH`) is the heater input. T_p (`tp`, `Tp`) is the unmeasured primary target. T1-T6 are the measured sensors, and aux1-aux5 are background channels. Sources appear as pseudonyms: S001-S139 for the locked manifest, and S01-S12 for the control scenarios. Seed 1/2/3 in the article correspond to protocol seeds 20260922/20260923/20260924.

## Environment

Python 3.9 with the pinned packages in `requirements.txt`:

```
pip install -r requirements.txt
```

Training and fine-tuning (`screen.py`, `formal_coisc.py`, `ablation_run.py`) expect a CUDA GPU. Everything else runs on a CPU.

## What runs without the private data

```
python figures/make_figures_tg.py [OUTPUT_DIR]       # Figures 1-3 of the Technical Gazette article from results/
python timc_coisc/reproduce_tg_results.py            # recompute the S3-S7 decisions from results/
python timc_coisc/posthoc_tg.py                      # recompute the post-hoc numbers of Section 5.5 from results/
python figures/make_figures.py [OUTPUT_DIR]          # Figures 1-7 of the earlier manuscript (PNG + PDF) from results/
python timc_coisc/test_control_mpc.py                # data-free MPC unit tests
python timc_coisc/formal_coisc.py --integrity-only   # protocol hash and frozen COISC settings
```

Both figure scripts write into `figures/` by default, and their output is pixel-identical to the figures in the respective article. `reproduce_tg_results.py` reruns the frozen S3-S7 analysis functions on the released per-seed, first-move and per-run tables and checks every number of `SCREENING_RESULTS.json`, `CLOSEDLOOP_RESULTS.json`, `GUARD_RESULTS.json`, `SEQ_RESULTS.json` and `DU_RESULTS.json`; for S7 it also reapplies the calibration rule for the move weight. `posthoc_tg.py` prints the post-hoc reversal, saturation and smoothness figures quoted in Section 5.5. `test_control_mpc.py` rewrites `timc_coisc/CONTROL_MPC_UNIT_TEST_REPORT_V2.json`.

Every other entry point stops with `FileNotFoundError` and names the private input it needs.

## Running with the private data

Two environment variables point the code at the private inputs and the run outputs.

| Variable | Default | Meaning |
|---|---|---|
| `TIMC_DATA_ROOT` | `./private_data` | private data, plant checkpoints and private protocol companions |
| `TIMC_OUTPUT_ROOT` | `./outputs` | outputs of the training, evaluation and control runs |
| `TIMC_VERIFY_HASHES` | unset | set to `1` to enforce the pre-release digests recorded in the protocols |

Protocol entries of the form `<private>/x` resolve under `TIMC_DATA_ROOT`, and `<outputs>/x` under `TIMC_OUTPUT_ROOT`.

Expected private layout:

```
$TIMC_DATA_ROOT/
  data/01_locked_source_manifest.csv
  data/field_windows_development_109.npz
  data_asset_audit/
  protocols_private/AVAILABLE23_EVALUATION_PROTOCOL_DRAFT.json   (source locations)
  frozen_evaluation/frozen_20260926T062548Z_c28722408e/
  canonical_preprocessing_20260926T_devonly/
  delivered23_data_adapter.py
  plants/<plant>_seed<seed>/selected.pt
  v10/SELECTED_native_rollout_g4state_tpsafe_v10_candidate.pt(h)
  source_pseudonyms.csv          (optional; otherwise derived from the manifest)
```

Run outputs:

```
$TIMC_OUTPUT_ROOT/
  screen/<mode>/                       screen.py --mode {smoke,development}
  formal_coisc/seed_<seed>/            formal_coisc.py
  delivered23/<execution_id>/          delivered23_run.py prepare | run EXECUTION_DIR
  delivered23_posthoc/                 posthoc_analyze.py
  posthoc_ablation/                    ablation_run.py, holdout_ablation.py, lag_reanalysis.py
  control/<execution_id>/              run_formal_control_v3a.py EXECUTION_DIR, analyze_formal_control_v3.py EXECUTION_DIR
  posthoc_mil/                         test_de_mpc.py, run_posthoc_mil.py {gates,freeze,execute}, analyze_posthoc_mil.py
  posthoc_mechanism/                   mech_analysis.py, explore_first_action_split.py, explore_mil_weighting_strata.py
  posthoc_onset_aligned/               onset_train.py {gates,smoke,e1,e2}, onset_holdout.py {gate,run},
                                       onset_mil.py {gates,execute}, onset_analyze_am1.py analyze MIL_EXECUTION_DIR
  posthoc_jacobian_b5/                 b5_train.py {gates,smoke,run}, b5_holdout.py {gate,run}
  posthoc_jacobian_b5m/                b5m.py {gates,gates_holdout,smoke,freeze,train,holdout}
  screening/                           scr_train.py {gate,smoke,run}, scr_holdout.py {gate,smoke,run},
                                       scr_first_action.py {gate,smoke,run}, scr_analyze.py {gate,analyze}, scr_freeze.py
  closedloop/                          cl_mil.py {gate,smoke,execute}, cl_analyze.py {gate,smoke,analyze}, cl_freeze.py;
                                       posthoc_tg.py export and derive_reversal.py read its raw_runs/
  guard/                               gd_mil.py {gate,smoke,execute}, gd_analyze.py {gate,smoke,analyze}, gd_freeze.py
  sequence/                            sq_train.py {gate,smoke,run}, sq_eval.py {gate,smoke,holdout,first_action},
                                       sq_mil.py {gate,smoke,execute}, sq_analyze.py {gate,smoke,analyze}, sq_freeze.py
  move_penalty/                        du_calibrate.py {run,decide}, du_mil.py {gate,smoke,execute},
                                       du_analyze.py {gate,smoke,analyze}, du_freeze.py
```

The stages run in the order of the article: development screening and formal COISC (Section 5.2), the frozen holdout (5.3), model-in-the-loop MPC (5.4), and the post-hoc checks (5.5). The holdout and control executions are write-once. Each refuses to run into an existing execution directory, or against a protocol whose hash has changed. The onset-aligned and Jacobian studies write their gate reports and frozen protocol into `protocols/<study>/`, and the remaining outputs into `$TIMC_OUTPUT_ROOT/<study>/`. The mechanism study only reads its frozen protocol from `protocols/posthoc_mechanism/`. The onset-aligned and Jacobian training and execution steps also require `$TIMC_OUTPUT_ROOT/<study>/EXECUTION_AUTHORIZATION.json`, a record of the author's go-ahead that names the SHA-256 of the frozen protocol. It is not part of the release. `onset_freeze.py` and `b5_freeze.py` (and `b5m.py freeze`) produce the frozen protocols. The screening (S3) and closed-loop (S4) studies follow the same pattern with `protocols/screening/` and `protocols/closedloop/`, outputs under `$TIMC_OUTPUT_ROOT/screening/` and `$TIMC_OUTPUT_ROOT/closedloop/`, and `scr_freeze.py` and `cl_freeze.py`. They reuse the onset-aligned code and its outputs under `$TIMC_OUTPUT_ROOT/posthoc_onset_aligned/` (the seed-20261001 checkpoints and MIL execution used by their reproduction gates), and S4 reads the S3 checkpoints. Run order: `scr_train.py run`, `scr_holdout.py run` (separate process), `scr_first_action.py run 6`, `scr_analyze.py analyze`, then `cl_mil.py execute 6` and `cl_analyze.py analyze`. The three baselines reuse the S3 checkpoints and the S4 runs and add one arm each, with protocols in `protocols/guard/`, `protocols/sequence/` and `protocols/move_penalty/` and outputs under the same names in `$TIMC_OUTPUT_ROOT/`: S5 `gd_mil.py execute 6`, `gd_analyze.py analyze`; S6 `sq_train.py run`, `sq_eval.py holdout` (separate process), `sq_eval.py first_action 6`, `sq_mil.py execute 6`, `sq_analyze.py analyze`; S7 `du_calibrate.py run 6` and `du_calibrate.py decide` on development seeds 1-3 before freezing, then `du_mil.py execute 6`, `du_analyze.py analyze`. S7 needs the formal R0 and COISC checkpoints of seeds 1-3 under `$TIMC_OUTPUT_ROOT/formal_coisc/`. The scripts that trained the plant family and prepared the control study are not part of this release. The protocols refer to them under `<private>/unreleased_code/`.

## Anonymisation and hashes

The released protocols and results were sanitised for publication. Private paths became `<private>/` or `<outputs>/`, source names became pseudonyms, and channel names became the paper's symbols. All numerical values are unchanged.

Because of this sanitising, a released protocol file no longer has the digest recorded at freeze time. `ORIGINAL_HASHES.json` lists the SHA-256 of each original file, keyed by the release path below `protocols/` (and `configs/`). Digests of data arrays and checkpoints inside the protocols (for example `history_sha256`) are unchanged. They are always checked.

`TIMC_VERIFY_HASHES=1` additionally enforces the pre-release digests of protocol files and code. Use it only together with the original, unsanitised protocol set.

Checkpoints written before the renaming are loaded through `_paths.load_checkpoint`, which maps legacy normalisation and state-dict keys onto the released names.

## Results

`results/` contains the summary tables and JSON decisions behind Tables 2-4 and Figures 1-7.

- `development_coisc/`: development gates (Section 6.2)
- `holdout23/`: frozen holdout (Section 6.3)
- `holdout23_posthoc/`: post-hoc holdout attribution (Section 6.4)
- `posthoc_ablation/`: ablations and the lag-prior reanalysis (Sections 6.4, 6.5)
- `control/`: model-in-the-loop MPC (Section 6.6)
- `posthoc_mil/`: post-hoc MIL checks (Section 6.7)
- `posthoc_mechanism/`: MIL bench diagnostics and the exploratory strata and first-action analyses (Supplementary S3, Tables S4-S6)
- `posthoc_onset_aligned/`: onset-aligned re-adaptation COISC-ON, its holdout, the ten further seeds and the MIL check (Section 6.5, Supplementary S6). `COISC_onset` is COISC-ON in the article; `COISC_onset_lamprior` is the variant with the original $\lambda$ (Table S11)
- `posthoc_jacobian_b5/`, `posthoc_jacobian_b5m/`: generic Jacobian baseline at the calibrated and the matched weight (Section 6.4, Supplementary S7); `JAC` and `JAC_M` are JAC and JAC-M in the article
- `figures_data/`: equal-source-weighted T_p sensitivity quantiles (Figure 2; Figure 1 of the Technical Gazette article) and the 12 closed-loop traces in Figure 7
- `screening/`: stage S3 of the Technical Gazette article (Section 5.4, Figure 2). Seeds 20261101-20261140 are seeds 14-53 in the article. Per-seed R0 holdout and development scores, the first move of R0, COISC, COISC_onset and L0 in the 12 scenarios, the per-seed training records and the decision `SCREENING_RESULTS.json`. `RESUME_LOG.json` records the interruption and same-protocol resume
- `closedloop/`: stage S4 (Section 5.5, Table 3, Figure 3a-c). `CLOSEDLOOP_RUN_METRICS.csv` has one row per run (8964 runs), `CLOSEDLOOP_RESULTS.json` the decision, and `EXECUTION_AUDIT.json` and `PAUSE_LOG.json` the three author-requested pauses and two resumes. Plant indices refer to `protocols/CONTROL_PLANT_FAMILY_LOCK.json`
- `guard/`: stage S5, the first-move rule `R0_GUARD` on the six R0-reversed pairs (Sections 4.4, 5.5). `GUARD_RUN_CELLS.csv` holds the per-run tracking MAE and reversal counts of all four arms (576 runs) and `GUARD_RESULTS.json` the decision
- `sequence/`: stage S6, the sequence variant `COISC_seq` (Sections 4.4, 5.5). Per-seed training records, holdout readouts, first moves, `SEQ_RUN_CELLS.csv` (11952 runs, all four arms) and the decision `SEQ_RESULTS.json`
- `move_penalty/`: stage S7, the retuned baseline `R0_DU` (Sections 4.4, 5.5). `calibration/CALIBRATION_RUN_TV.csv` has the 1296 calibration runs that fixed the move weight, `DU_RUN_CELLS.csv` the 11520 panel-B runs of all four arms, `DU_RESULTS.json` the decision, and `EXECUTION_AUDIT.json` and `PAUSE_LOG.json` the three author-requested pauses and one unexpected stop, each followed by a same-protocol resume
- `posthoc_tg/`: per-run reversal, saturation and smoothness counts of the 8964 S4 runs behind the post-hoc paragraphs of Section 5.5; `figures_data/derived_reversal_seed.csv` holds the seed-level reversal fractions of Figure 3d

Some columns were removed because they would describe the private sources. The source-group, domain, scene and machine descriptors are gone from `POSTHOC_SOURCE_FAILURE_TABLE.csv`. `POSTHOC_SOURCE_GROUP_DECOMPOSITION.csv` keeps only the source-role grouping. Neither change affects any value reported in the article.

## License

MIT, see `LICENSE`. Contact: Junnan Zhou, zhoujn1@mails.neu.edu.cn.
