# COISC: staged evaluation of lag-aware sensitivity adaptation of a learned thermal surrogate

Analysis code, frozen evaluation protocols and summary results for the article

> J. Zhou and B. Liu, "Does lag-aware sensitivity adaptation of a learned thermal surrogate improve prediction or control? A staged evaluation", submitted to *Transactions of the Institute of Measurement and Control*.

The field data are proprietary and are not included. Trained model weights are not released either. Without the data, you can still inspect the code and protocols, run the data-free tests, and redraw every figure in the article from the released summary results.

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
python figures/make_figures.py [OUTPUT_DIR]          # Figures 1-7 (PNG + PDF) from results/
python timc_coisc/test_control_mpc.py                # data-free MPC unit tests
python timc_coisc/formal_coisc.py --integrity-only   # protocol hash and frozen COISC settings
```

`make_figures.py` writes into `figures/` by default. Its output is pixel-identical to the figures in the article. `test_control_mpc.py` rewrites `timc_coisc/CONTROL_MPC_UNIT_TEST_REPORT_V2.json`.

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
```

The stages run in the order of the article: development screening and formal COISC (Section 5.2), the frozen holdout (5.3), model-in-the-loop MPC (5.4), and the post-hoc checks (5.5). The holdout and control executions are write-once. Each refuses to run into an existing execution directory, or against a protocol whose hash has changed. The onset-aligned and Jacobian studies write their gate reports and frozen protocol into `protocols/<study>/`, and the remaining outputs into `$TIMC_OUTPUT_ROOT/<study>/`. The mechanism study only reads its frozen protocol from `protocols/posthoc_mechanism/`. The onset-aligned and Jacobian training and execution steps also require `$TIMC_OUTPUT_ROOT/<study>/EXECUTION_AUTHORIZATION.json`, a record of the author's go-ahead that names the SHA-256 of the frozen protocol. It is not part of the release. `onset_freeze.py` and `b5_freeze.py` (and `b5m.py freeze`) produce the frozen protocols. The scripts that trained the plant family and prepared the control study are not part of this release. The protocols refer to them under `<private>/unreleased_code/`.

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
- `figures_data/`: equal-source-weighted T_p sensitivity quantiles (Figure 2) and the 12 closed-loop traces in Figure 7

Some columns were removed because they would describe the private sources. The source-group, domain, scene and machine descriptors are gone from `POSTHOC_SOURCE_FAILURE_TABLE.csv`. `POSTHOC_SOURCE_GROUP_DECOMPOSITION.csv` keeps only the source-role grouping. Neither change affects any value reported in the article.

## License

MIT, see `LICENSE`. Contact: Junnan Zhou, zhoujn1@mails.neu.edu.cn.
