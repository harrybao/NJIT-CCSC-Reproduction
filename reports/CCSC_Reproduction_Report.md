# Reproducing the NJIT CCSC Deep-Learning Tools: Methods, Results and Lessons for Reliable Scientific ML

**Author:** Xiangsheng (Harry) Bao (Master of Data Science candidate, University of Sydney)
**Date:** 13 October 2026
**Scope:** All tools of the Community Coordinated Software Center (CCSC) that could be identified (19 repositories
under `github.com/ccsc-tools`), plus the related SolarDB tool MagNet.
**Repository:** every number below is produced by a script in `scripts/` and logged in `results/`.

---

## 1. Executive Summary

**What was asked.** "For each tool, click the Citation button, download the tool to reproduce all results and let me know how the tool works."

**What was done.**
* I identified 19 CCSC tools and recorded their publications, code, data, weights and environments (`tool_inventory.md`).
* I built four isolated Python environments on a 4-core CPU machine without a GPU.
* For each tool I ran the authors' released test/prediction path with the authors' pretrained models where available, and compared the outputs with the authors' stored outputs and reported numbers.
* Where a released evaluation looked questionable, I added independent, leakage-aware checks: chronological hold-outs, grouped splits, persistence baselines, and label-free evaluation.

**Access limitations.** The CCSC/CCDC pages (`nature.njit.edu`), the group's file server (`web.njit.edu`), arXiv, Zenodo and the journal sites were unreachable from the environment used. The Citation buttons therefore could not be clicked. Citations were taken from each repository's README/notebook and confirmed through bibliographic search metadata, and paper tables could not be read in full. This is the main reason several comparisons are against **the authors' stored tool outputs** rather than against paper tables.

**Status overview.**

| Status | Tools |
|---|---|
| **Successfully reproduced** (scope: released model + released test data → authors' stored outputs/numbers) | SEP-prediction (12/12 metric rows exact), RNN-CME-prediction (10/10 identical predictions), SolarUnet (masks byte-identical after a file-ordering fix), MIST (TSS 0.9062 = README 0.90), SINet (F10.7 exact; F30 within 2 %), Kp-prediction (within MC noise), DeepGeoCME (identical confusion matrix, TSS 0.757) |
| **Partially reproduced** | LSTM-flare-prediction (Table 2 retrained, 10 seeds; Table 5 protocol re-run; demo ≥M5.0 label inversion found), FlareML (demo inference only), SolarFlareNet (stored outputs reproduced, but 8/9 shipped models show no skill), TSInet (default runs; CLI bug), CMETNet (numerical ensemble only; test-set selection bias quantified), SYMHnet (stored storm outputs reproduced at tool level; the released test path is effectively one-step-ahead) |
| **Attempted but unsuccessful** | Dst-prediction: the stored outputs are reproduced, but the released `predict()` substitutes 0.9 × the true label for most predictions. The raw model does not reproduce the reported skill (R² 0.06–0.38). |
| **Blocked by missing data or compute** | FibrilNet and SDNN (weights and inputs only on web.njit.edu), SolarCNN (weight file missing from the repo), DeepHMI (1,000-step diffusion ≈ 11+ CPU-hours; authors' outputs re-scored instead), MagNet (nature.njit.edu) |
| **Not attempted** | LLMFlare: needs gated Llama-3.1-8B weights, a paid Gemini API key and a GPU. I did not incur costs without authorisation. |

**Key insights for reliable scientific ML.**
1. **Validation design dominates.** The pretrained MIST transformer reaches TSS 0.906 on its released random split of overlapping 24-hour windows (95.6 % of test windows have a 23-hour-overlapping training neighbour). On the chronologically and active-region-independent 2015–2018 data it reaches **TSS 0.197** (AUC 0.997 → 0.833). On that same independent test set, the 2019 LSTM retrained on the published split reaches TSS 0.73–0.88 depending on threshold policy.
2. **The 2019 LSTM data split is exemplary.** It is chronological (2010–13 / 2014 / 2015–18) with zero shared active regions. Using it, I measured how much choosing the threshold on the test set inflates TSS: +0.01 for ≥C and +0.14 for ≥M5.0.
3. **Several released evaluation paths do not measure what they appear to measure.**
   * Dst: label-dependent post-processing.
   * SYMHnet: identical 1 h and 2 h data with a next-row target.
   * SolarFlareNet: bundled test rows also present in training, and a position-based feature selection shifted by one column.
   * CMETNet: best-of-10 model chosen on the test year.
   * FlareML: display-side metric adjustment and test-batch scaling.
4. **These are observations about the released software packages, not about the published papers**, whose full texts and original experimental code I could not access. I would value the authors' guidance on whether the papers' evaluations differ from the released tools.

---

## 2. Tools and Publications

Full 16-field records are in `tool_inventory.md`; commit hashes are in `tools/upstream_commits.csv`.

| Tool | Publication | DOI / identifier |
|---|---|---|
| FlareML | Abduallah et al., *DeepSun: machine-learning-as-a-service for solar flare prediction*, RAA 21:160 (2021) | 10.1088/1674-4527/21/7/160; Zenodo 10.5281/zenodo.5634114 |
| LSTM-flare-prediction | Liu, Liu, Wang & Wang, *Predicting Solar Flares Using a Long Short-term Memory Network*, ApJ 877:121 (2019) | 10.3847/1538-4357/ab1b3c; arXiv:1905.07095 |
| RNN-CME-prediction | Liu, Liu, Wang & Wang, *Predicting CMEs Using SDO/HMI Vector Magnetic Data Products and RNNs*, ApJ 890:12 (2020) | 10.3847/1538-4357/ab6850; arXiv:2002.10953 |
| SolarFlareNet | Abduallah, Wang, Wang & Xu, *Operational prediction of solar flares using a transformer-based framework*, Sci. Rep. 13:13665 (2023) | 10.1038/s41598-023-40884-1; Zenodo 10.5281/zenodo.10080716 |
| SEP-prediction | Abduallah et al., *Predicting SEPs Using SDO/HMI Vector Magnetic Data Products and a BiLSTM Network*, ApJS 260:16 (2022) | 10.3847/1538-4365/ac5f56; Zenodo 7516609 |
| CMETNet | Alobaid et al., *Predicting CME arrival time through data integration and ensemble learning*, Front. Astron. Space Sci. 9:1013345 (2022) | 10.3389/fspas.2022.1013345; Zenodo 7516684 |
| Dst-prediction | Abduallah et al., *Forecasting the Disturbance Storm Time Index with Bayesian Deep Learning*, FLAIRS-35 (2022) | 10.32473/flairs.v35i.130564; Zenodo 7529705 |
| Kp-prediction | Abduallah, Wang, Xu & Wang, *A Transformer-Based Framework for Geomagnetic Activity Prediction*, ISMIS 2022, LNCS 13515 | Zenodo 7514451 |
| TSInet | Abduallah et al., *Reconstruction of Total Solar Irradiance by Deep Learning*, FLAIRS-34 (2021) | 10.32473/flairs.v34i1.128356; Zenodo 7516668 |
| SYMHnet | Abduallah et al., *Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With UQ*, Space Weather 22, e2023SW003824 (2024) | 10.1029/2023SW003824; Zenodo 10.5281/zenodo.10602518 |
| SolarUnet | Jiang et al., *Identifying and Tracking Solar Magnetic Flux Elements with Deep Learning*, ApJS 250:5 (2020) | 10.3847/1538-4365/aba4aa |
| FibrilNet | Jiang et al., *Tracing Hα Fibrils through Bayesian Deep Learning*, ApJS 256:20 (2021) | 10.3847/1538-4365/ac14b7; Zenodo 7521568 |
| SDNN | Jiang et al., *Inferring LOS Velocities and Doppler Widths from Stokes Profiles of GST/NIRIS Using Stacked DNNs*, ApJ 939:66 (2022) | 10.3847/1538-4357/ac927e; Zenodo 7521580 |
| SolarCNN | Xu et al., *Super-Resolution of SOHO/MDI Magnetograms … Attention-Aided CNN*, Sol. Phys. 299:36 (2024) | 10.1007/s11207-024-02283-1; Zenodo 10.5281/zenodo.15250172 |
| DeepHMI | Xu, Xu, Wang, Li & Wang, *Improving the Spatial Resolution of SDO/HMI … Using GST/NIRIS Data*, A&A (2025) | 10.1051/0004-6361/202453581 |
| MIST | Chaudhary, Abduallah & Wang (2025), README notebook; publication not confirmed | Zenodo 16780646 (cited in LLMFlare README) |
| SINet | Wang Z. et al., *Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning*, JGR Space Physics (2026) | 10.1029/2025JA034868 |
| LLMFlare | no citation given in the repository | — |
| DeepGeoCME | Yan, Abduallah & Wang, *Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations* (2026) | arXiv:2605.24748; Zenodo 22955090 |
| MagNet (SolarDB) | Jiang et al., Sol. Phys. 298:87 (2023); Yesmanchyk et al., *Out-of-Sample Validation of MagNet* (2026) | 10.1007/s11207-023-02180-z; arXiv:2601.15926 |

---

## 3. Methodology

**3.1 Identifying the tools.**
* `nature.njit.edu` was denied by the environment's egress policy, so I reconstructed the CCSC catalogue from the public GitHub account `ccsc-tools`, whose display name is "Community Coordinated Software Center". It holds 22 repositories: 19 tools, a 2021 website stub, a profile repo, and a repo of Zenodo badges for 13 tools.
* A search-engine snippet confirms that the CCSC page states the notebooks "are now available on GitHub and Zenodo".
* Whether the live page lists exactly these 19 tools is unverified.

**3.2 Reproduction protocol (per tool).**
1. Read the README and notebook and record the citation, data, weights and environment.
2. Make a working copy of the upstream clone; the upstream clone is never modified.
3. Create an isolated environment with the closest available versions (§4).
4. Run the **released** test/prediction entry point with the **released** weights and data, unchanged, and log stdout.
5. Compare with the authors' stored outputs (result CSVs, logs, notebook outputs) and with numbers stated in READMEs.
6. Audit the evaluation: data provenance, train/test overlap, threshold and model selection, label use at inference, horizon definition, baselines.
7. Where the released evaluation was not out-of-sample, run an **independent** evaluation that changes only the evaluation, never the model, and label it as an extension, not a reproduction.

**3.3 Integrity rules.**
* Nothing is reported that was not executed; every number has a log or JSON.
* "Reproduced" always states its scope.
* Compatibility edits are kept minimal and listed. Examples: `Model(inputs=…)` for Keras 2, keyword arguments for scikit-learn ≥ 1.0, a `torch.load(weights_only)` shim for torch 1.12, sorted `os.listdir`, and module globals set to bypass a CLI bug.
* No upstream repository was modified, no paid service was used, and no e-mail was sent.

**3.4 Metrics.**
* Classification: recall, precision, ACC, BACC, HSS and **TSS** (insensitive to class ratio, the standard for flare forecasting), plus AUC.
* Regression: RMSE, MAE, R², Pearson r, and skill relative to **persistence**.
* Images: SSIM, PSNR, PCC, IoU and RMSE (G).
* Definitions are in `scripts/common_metrics.py`.

---

## 4. Experimental Setup

* **Hardware:** cloud container, 4 vCPU, 15 GB RAM, **no GPU**, ~30 GB writable disk; Linux 6.18.
* **Network:** GitHub, raw.githubusercontent.com and PyPI were reachable. `nature.njit.edu`, `web.njit.edu`, arXiv, Zenodo, Google Drive, `download.pytorch.org`, journal sites and the Wayback Machine were not.
* **Environments** (frozen in `tools/environments/`):
  * `tf210`: Python 3.9.25, TensorFlow 2.10.1 (CPU), Keras 2.10, TFP 0.17, scikit-learn 1.2.1, xgboost 1.5. Used for the TensorFlow tools (papers used TF 1.12–2.11).
  * `sk024`: Python 3.9, scikit-learn 0.24.2, sklearn-extensions 0.0.2. Used for FlareML; the pickles were written by scikit-learn 0.23.2.
  * `torch112`: Python 3.9, PyTorch 1.12.1 (CPU), timm 1.0.9. Used for MIST, SINet, DeepHMI and DeepGeoCME (MIST/DeepGeoCME request torch 2.3/≥2.6).
  * `np2`: Python 3.11, NumPy 2.0. Used only to read MIST's scaler pickle.
* **Upstream code:** commit hashes in `tools/upstream_commits.csv`.
* **Re-running:** each `results/<tool>/` folder has the logs. Each script in `scripts/<tool>/` documents its usage (run from a working copy of the upstream repository).

---

## 5. Reproduction Results

### 5.1 Flare forecasting

**LSTM-flare-prediction (Liu et al. 2019).**
* *Released demo:* the pretrained models were run on the three single-AR demo test sets.
  * For **≥M5.0**, the test script reads softmax column 0 as "positive". The label encoder orders the classes ['N','X'], so predictions are inverted: TSS −0.671 as released. With column 1 the TSS is 0.686, and agreement with the authors' stored output is **98.9 %**.
  * The ≥C and ≥M stored outputs agree 93 % and 88 % with the shipped `.h5` files, so they were probably produced by earlier model snapshots.
  * `results/LSTM-flare-prediction/pretrained_orientation_check.json`
* *Paper split (Table 2), retrained with the released TF2 code and unchanged hyper-parameters* (7 epochs, batch 256, balanced class weights), **10 seeds** each; test = 2015-01 → 2018-05:

| Class (test n, positives) | Threshold policy | Recall | Precision | BACC | HSS | **TSS** |
|---|---|---|---|---|---|---|
| ≥M5.0 (44,689, 180) | tool's fixed 0.75 | 0.778 | 0.068 | 0.867 | 0.117 | 0.734 ± 0.164 |
| | chosen on 2014 validation | 0.904 | 0.048 | 0.913 | 0.083 | 0.825 ± 0.104 |
| | chosen on test (paper convention) | 0.964 | 0.047 | 0.939 | 0.082 | **0.877 ± 0.035** |
| ≥M (44,689, 1,278) | fixed 0.60 | 0.810 | 0.233 | 0.865 | 0.331 | 0.730 ± 0.031 |
| | validation | 0.864 | 0.173 | 0.869 | 0.251 | 0.739 ± 0.022 |
| | test | 0.846 | 0.206 | 0.874 | 0.298 | **0.748 ± 0.021** |
| ≥C (44,689, 8,732) | fixed 0.50 | 0.722 | 0.555 | 0.790 | 0.521 | 0.580 ± 0.011 |
| | validation | 0.748 | 0.528 | 0.792 | 0.505 | 0.584 ± 0.009 |
| | test | 0.757 | 0.524 | 0.795 | 0.504 | **0.589 ± 0.009** |

  Mean (std) over seeds; full table in `results/LSTM-flare-prediction/table2_summary.md`.
  * Split audit: train / validation / test HARPs are disjoint (0 shared ARs) and chronological (`table2_split_audit.json`).
* *Paper Table 5 protocol (≥C; 10×10 CV, original `LSTMpredict.py` with the 4-line TF2 compatibility patch `scripts/LSTM-flare-prediction/LSTMpredict_tf2_compat.patch`).* 100 trainings, 86 CPU-minutes:

| Metric | Authors' `output.txt` (TF 1.12, Keras 2.2.4) | Re-run (TF 2.10) |
|---|---|---|
| Recall | 0.773 (0.027) | 0.741 (0.023) |
| Precision | 0.541 (0.030) | 0.531 (0.024) |
| ACC | 0.826 (0.015) | 0.821 (0.012) |
| BACC | 0.806 (0.004) | 0.790 (0.005) |
| HSS | 0.526 (0.021) | 0.505 (0.018) |
| **TSS** | **0.612 (0.009)** | **0.581 (0.011)** |

  The re-run is 0.031 TSS lower, a consistent but small gap. Patching in Keras 2.2.4's default `recurrent_activation='hard_sigmoid'` did not change the result (Table 2 ≥C TSS 0.589 ± 0.009, 5 seeds), so the remaining gap is attributed to other framework or version differences or unrecorded run-to-run variation (`table5_comparison.json`).
* *Code notes:* the history-masking check tests `row` instead of `prev_row`, so incomplete history rows are never masked. Table 5 selects the threshold that maximises mean TSS on the test folds.

**MIST.**
* The released `test.py` reproduces **TSS 0.9062** (TP 493, FN 49, FP 56, TN 16,313), matching the README's 0.90.
* Data provenance: `processed_solar_dataset.csv` equals the LSTM Table 2 ≥M training file (2010-05 → 2014-01) to within 4.4 × 10⁻¹⁶ on every checked feature, with identical labels. Windows are rows *p…p+23* labelled by row *p+24* (100 % agreement). The 80/20 split is random over windows; the scaler is fitted on all 84,577 rows.
* **Overlap:** 16,172 of 16,911 test windows (95.6 %) have an adjacent window, differing by one hour, outside the test set, i.e. in training.
* **Out-of-sample check** (same weights and preprocessing; windows built from the Table 2 ≥M validation and test files):

| Evaluation set | n windows | positives | AUC | TSS @0.5 | TSS @ threshold chosen on 2014 |
|---|---|---|---|---|---|
| Released random-split test | 16,911 | 542 | 0.997 | **0.906** | — |
| 2014 (validation file) | 26,449 | 1,347 | 0.835 | 0.208 | — |
| 2015–2018 (test file) | 44,665 | 1,278 | 0.833 | **0.197** | 0.364 |

  Windows lying entirely inside one AR give similar values (TSS 0.164–0.186). `results/MIST/out_of_sample_eval.json`.

**SolarFlareNet.**
* The 9 shipped checkpoints load completely (95/95 tensors each), and the released tests reproduce the authors' stored result CSVs (≥ 99.98 % prediction agreement for 8/9).
* However, those **8 models have no skill** on the bundled 2022 test files (AUC 0.41–0.53; TSS −0.04 to 0.05), in both our run and the authors' stored files.
* The ninth (C/24 h) was retrained after the stored results were produced (shipped log, 2023-11-05). It gives TSS 0.792 and AUC 0.969. However, 84 % of the 24 h test rows (same HARP and timestamp) are also in `training_data_*_24.csv`; on the 2,581 evaluated rows *not* in training, TSS = 0.697.
* The loader selects features by position, using another dataset's column order, and its `start_feature=5` skips the first feature. The 9 features fed are TOTPOT, ABSNJZH, USFLUX, AREA_ACR, MEANPOT, R_VALUE, MEANGAM, MEANJZH, MEANJZD, rather than the 9 named in the code.
* `results/SolarFlareNet/`.

**FlareML.**
* The released pretrained ensemble reproduces the shipped log on the 40-sample demo: ENS accuracy 0.850, RF 0.825, ELM 0.725, MLP 0.300 (MLP predicts "C" for every sample, as in the authors' log).
* 31/40 demo test rows match training-sample rows on all 13 features (≤ 0.6 %, same class).
* Two display-side adjustments exist: `check_pm_precision` replaces RF's displayed per-class score by ENS − U(0.01, 0.07) whenever RF ≥ ENS, and `log_cv_report` edits FP/FN cells when they are 0.
* *Extension* on the 845-flare table, RF with the shipped hyper-parameters, 5 seeds: random 10-fold macro-TSS **0.338 ± 0.007**; AR-grouped 10-fold **0.246 ± 0.022**; chronological (< 2014 → 2014–16) 0.363 ± 0.009 (few X-class test events).

### 5.2 CME and SEP

* **SEP-prediction:** all 12 stored metric rows (FC_S/F_S × 12–72 h) are reproduced **exactly**, e.g. FC_S 24 h TSS 0.9782, F_S 24 h TSS 0.8689.
  * The test period is Apr–Sep 2017 with 3 ARs, and all positives come from 2 ARs (12665, 12673).
  * The calibrated Brier skill (BSSC) uses an isotonic regression fitted on the test set.
* **RNN-CME-prediction:** for all 10 GRU/LSTM × 12–60 h models, predicted labels are **identical** to the stored outputs (TSS 0.817–1.000). The test set covers 2 ARs (12497, 12529).
* **CMETNet** (numerical COMB ensemble; CNN not run).
  * The released script trains each regressor 10 times on random 200-event subsamples and keeps the round with the lowest MAE **on the hold-out year**.
  * Re-implementing the loop for every year with ≥ 3 events (same estimators and hyper-parameters): [[CMETNET]]
  * `pretrained_models_results.py` plots hard-coded MAE/PPMCC values; it does not compute them.
* **DeepGeoCME:** release assets verified by SHA-256. Our confusion matrix on the 33 test events is **identical** to the shipped figure (TN 14, FP 2, FN 2, TP 15): TSS 0.757, recall 0.882, precision 0.882, Brier score 0.080.

### 5.3 Geomagnetic indices, radio flux, irradiance

* **Dst-prediction (DSTT).**
  * `DSTTModel.predict()` maps each prediction *p* with true value *a* through `t(a, p, 0.9)`. *p* is kept only if it equals *a* or differs from it by ≥ 90 %; otherwise it becomes 0.9·*a*.
  * The authors' stored predictions equal 0.9 × label exactly for 88–90 % of hours. `DSTT_test.test()` also loads the 6 h weights for every horizon.
  * The loader's test-exclusion filter (`isin` on partial strings) removes 0 rows, so all 1,103 test hours are also in the training frame.
  * Label-free evaluation (50-pass MC mean, correct per-horizon weights) on Oct–Nov 2021:

| h | Released path R² (RMSE) | Raw model R² (RMSE nT) | Persistence R² (RMSE nT) |
|---|---|---|---|
| 1 | 0.942 (3.97) | 0.063 (15.98) | 0.934 (4.24) |
| 2 | 0.942 (3.97) | 0.076 (15.88) | 0.836 (6.70) |
| 3 | 0.938 (4.11) | 0.340 (13.43) | 0.750 (8.27) |
| 4 | 0.940 (4.05) | 0.380 (13.02) | 0.674 (9.45) |
| 5 | 0.938 (4.11) | 0.127 (15.46) | 0.599 (10.47) |
| 6 | 0.932 (4.30) | 0.338 (13.47) | 0.525 (11.41) |

* **Kp-prediction (KpNet).**
  * Re-run metrics match the stored `default_results` within the stochastic variation of the variational layers (e.g. 1 h: R² 0.629 vs 0.625; r = 0.801 in both).
  * Train 2010–2021 vs test Jan–Mar 2022, with no shared timestamps.
  * Persistence is better at 1–2 h (R² 0.82 / 0.65 vs 0.63 / 0.49); KpNet is better from 3 h onward (e.g. 6 h: 0.26 vs 0.08).
* **SYMHnet.**
  * The released test path reproduces the authors' stored storm outputs at tool level (same series; RMSE of the same order, MC noise).
  * The released data folders for 1 h and 2 h are **byte-identical** (same MD5), and `SYM_H == SYMH_ORIG` (no shift). The test windowing (`lag=1`, `forecast_horizon=1`) targets the **next row**, i.e. +1 min or +5 min.
  * Results with two persistence references: [[SYMH]]
* **SINet.**
  * F10.7: RMSE 3.8801 / 13.7411 / 15.0921 / 16.3241 (1 / 27 / 45 / 60 d), identical to the notebook to 4 decimals.
  * F30: 2.0466 / 7.7862 / 9.1215 / 10.1475, vs notebook 2.0586 / 7.6491 / 9.1584 / 9.9921 (within 2 %).
* **TSInet.**
  * `process_args()` lacks a `global` declaration, so `-n NRLTSI2/SATIRE-S` silently reconstructs TCTE; I used a three-line driver that sets the module globals.
  * 30-day default reconstructions: RMSE 0.71–0.79 W m⁻², vs 0.62–0.72 in the authors' stored files (stochastic periodic re-fitting).
  * A naive "repeat the last seed value" baseline gives 0.09–0.23 W m⁻² on the same windows.

### 5.4 Image-based tools

* **SolarUnet.** The pipeline iterates `os.listdir()` unsorted; on Linux the order was frame_1, frame_3, frame_2, which mis-pairs masks with FITS frames. With sorted listing, the 3 predicted masks are **byte-identical** to the authors' and the post-processed and tracking images agree on ≥ 99.99 % of pixels.
* **SolarCNN.**
  * The SavedModel lacks `variables.data-*` (on Zenodo / Google Drive, unreachable), so it could not be run.
  * Re-scoring the 3 shipped predictions against HMI: they have a different amplitude scale (std 65 G vs 197 G), consistent with `test.py` omitting the training normalisation (clip ±2000 G, ÷2000).
  * After the same oracle linear calibration of input and prediction, the predictions are **not better than the MDI input**: SSIM 0.763–0.777 vs 0.770–0.785; PSNR 31.5–32.3 vs 31.5–32.3 dB.
* **DeepHMI.** One U-Net pass takes ≈ 67 s on this CPU (1,000 steps × 2 components ≈ 37 h), so the run is blocked by compute. Re-scoring the authors' shipped output on the single test pair against GST/NIRIS:
  * Bz: SSIM 0.810 vs 0.733 (HMI input), RMSE 190 vs 220 G, PCC 0.972 vs 0.965.
  * Bt: SSIM 0.659 vs 0.683, RMSE 206 vs 223 G, PCC 0.896 vs 0.906.
* **FibrilNet / SDNN:** blocked (artefacts on web.njit.edu).

---

## 6. Comparison with Reported Results

| Tool | Reported (source) | Reproduced here | Agreement |
|---|---|---|---|
| LSTM ≥C Table 5 | TSS 0.612 ± 0.009 (`output.txt`) | TSS 0.581 ± 0.011 (TF 2.10, same code and protocol) | close (Δ = −0.031) |
| LSTM Table 2 (paper split) | paper tables not readable here | TSS 0.877 / 0.748 / 0.589 (≥M5.0 / ≥M / ≥C, test-chosen threshold, 10 seeds) | numbers for the authors to compare |
| MIST | TSS 0.90 (README) | 0.906 (released split); 0.197 (2015–18) | released number exact; not robust out-of-sample |
| SolarFlareNet | paper TSS not readable; stored result CSVs | stored CSVs reproduced; 8/9 AUC ≈ 0.5 | stored outputs reproduced; skill not |
| FlareML | demo log | identical | exact (demo) |
| RNN-CME | stored outputs | identical labels | exact |
| SEP | `default_results` (TSS 0.86–0.99) | identical | exact |
| CMETNet | MAE 9.75 h, r 0.83 (README) | [[CMETNET_SHORT]] | partial (no CNN) |
| Dst | stored results, R² ≈ 0.94 | released path R² ≈ 0.94; raw model R² 0.06–0.38 | not supported by raw model |
| Kp | stored results | within MC noise | reproduced |
| SYMHnet | README FSS 0.343 / 0.553 (storm 37, 5-min) | [[SYMH_SHORT]] | see §5.3 |
| SINet | notebook RMSE | F10.7 exact; F30 within 2 % | reproduced |
| TSInet | stored 30-day files | similar RMSE; naive baseline better | reproduced at tool level |
| SolarUnet | stored masks | byte-identical (sorted listing) | exact |
| DeepGeoCME | shipped confusion matrix | identical | exact |
| SolarCNN / DeepHMI | paper metrics not readable | shipped outputs re-scored | — |

---

## 7. Limitations and Challenges

1. **No access to papers or to the CCSC/CCDC pages.** Comparisons are mostly against the authors' stored outputs and README numbers. Paper tables (e.g. LSTM Tables 3–6, SolarFlareNet, SYMHnet) should be compared by someone with the full texts.
2. **Version drift.** Tools were written for TF 1.12–2.11 and PyTorch 1.12–2.6; I ran TF 2.10 and PyTorch 1.12 on CPU. Most outputs were nevertheless exact, which indicates numerical stability.
3. **Stochastic models** (variational/MC-dropout layers in Dst/Kp/SYMHnet; diffusion in DeepHMI; random subsampling in CMETNet; re-fitting in TSInet) cannot be reproduced bit-for-bit, so I compare metrics.
4. **Compute.** There is no GPU: DeepHMI's sampler (≈ 37 CPU-hours), SolarFlareNet retraining (5 × LSTM-400 stacks per model) and FibrilNet training were out of scope.
5. **Small evaluation sets in demos** (1–3 images; 2–3 ARs for SEP/RNN-CME; 33 events for DeepGeoCME) make many reproduced numbers statistically fragile.
6. **Scope of claims.** My findings concern the **released** code, data and weights. The original experimental code behind each paper may differ, and I have not concluded otherwise.

---

## 8. Understanding of Each Tool (how they work)

* **FlareML.** A tabular 4-class classifier. Thirteen SHARP parameters at the start of a flare day feed RF, MLP and ELM models (an ELM uses a random hidden layer with a closed-form output layer), combined by majority vote. Skill is reported as per-class one-vs-rest TSS.
* **LSTM-flare.** A sequence classifier. An AR's last 10 hourly SHARP + flare-history vectors go into an LSTM with temporal attention, which outputs the probability of a ≥X-class flare within 24 h. Class imbalance is handled with weighted cross-entropy; a threshold then turns the probability into an alert.
* **RNN-CME.** The same family (GRU/LSTM + attention) conditioned on an M/X flare having occurred. It predicts whether the eruption carries a CME, a key step towards geoeffectiveness.
* **SolarFlareNet.** A deeper transformer-style sequence model (Conv1D → LSTM → 4 encoder blocks) for 24/48/72 h horizons. It has a calibration-based probabilistic output and is run operationally.
* **SEP-prediction.** A BiLSTM + attention over SHARP sequences, predicting whether a flaring AR yields an SEP event, with isotonic probability calibration.
* **CMETNet.** A regression ensemble (RF, SVR, XGBoost, GPR medianed) on CME and solar-wind features, fused with a CNN on LASCO C2 images, predicting Sun-to-Earth transit time.
* **DSTT / KpNet / SYMHnet.** Solar-wind-driven geomagnetic-index forecasters. Convolution + recurrent + attention feature extractors feed Bayesian output layers (TFP variational layers or MC dropout); repeated stochastic passes give the mean prediction plus aleatoric and epistemic spread. SYMHnet adds a graph layer that treats solar-wind parameters as nodes.
* **SINet.** TimesNet-style forecasting. An FFT detects dominant periods (e.g. the 27-day solar rotation), the 1-D series is reshaped to 2-D by period, and Inception convolutions are applied, giving multi-day F10.7/F30 forecasts.
* **TSInet.** An autoregressive Conv-LSTM-attention generator seeded with 7 observed TSI values. It rolls forward with periodic self-training on its own outputs.
* **SolarUnet / FibrilNet.** U-Net segmentation of BBSO/GST images, trained on labels from classical threshold tools (SWAMIS, a threshold-based fibril tracer). Post-processing tracks features and fits fibril orientations; FibrilNet adds MC-dropout uncertainty maps.
* **SDNN.** A stacked 1-D CNN that approximates Milne–Eddington inversion of NIRIS Stokes profiles (I, Q, U, V). It replaces slow iterative fitting with a learned mapping.
* **SolarCNN / DeepHMI / MagNet.** Cross-instrument image translation:
  * SolarCNN: MDI → HMI quality, using attention CNNs.
  * DeepHMI: HMI → GST resolution, using a conditional diffusion model.
  * MagNet: MDI LOS + Hα → transverse field, trained with HMI vectors.
  The scientific purpose is to homogenise multi-decade archives across instruments.
* **MIST.** A patch-based time-series transformer: 24 h of 16 SHARP features are cut into 2 h patches, embedded and encoded, and output a ≥M flare probability.
* **LLMFlare.** An explanation layer: SHAP attributions for MIST predictions are turned into narratives by a LoRA-tuned Llama-3.1-8B, and the narratives are judged by an LLM.
* **DeepGeoCME.** A multimodal image fusion classifier (ResNet-50 + EfficientNet-B0 + attention) that predicts whether a CME will be geoeffective.

---

## 9. Potential Research Directions

1. **A leakage and evaluation audit toolkit for heliophysics ML.** The checks in §5 could become a reusable test suite run alongside every CCSC tool: AR/time overlap, window adjacency, threshold or model selection on test, label access at inference, horizon consistency, and persistence baselines.
2. **AI-ready split manifests and provenance for CCDC/SolarDB datasets.** Ship each dataset with grouped chronological splits, a data dictionary (units, missing-value codes), checksums and lineage. The undocumented MIST ← LSTM Table 2 relationship shows why. This fits the aims of NSF 26-512.
3. **External, cross-instrument validation and reliability stratification** (the question in my first e-mail). For MagNet, SolarCNN and DeepHMI, quantify error by instrument (MDI, HMI, IVM, GST), seeing and data quality, field strength and coverage. The goal is to flag where reconstructed fields should be interpreted with caution across the historical archive.
4. **Calibrated uncertainty.** Test whether the Bayesian intervals of DSTT, KpNet and SYMHnet achieve nominal coverage without ad-hoc scale factors, and report proper scoring rules.
5. **Event-level uncertainty for small test sets.** Use bootstrap-over-ARs confidence intervals for TSS, so that results on 2–3 ARs are interpretable.

---

## 10. References

See §2 for the 19 tool publications. Additional references:
* Bloomfield D. S., Higgins P. A., McAteer R. T. J., Gallagher P. T. 2012, *Toward Reliable Benchmarking of Solar Flare Forecasting Methods*, ApJL 747:L41.
* Nita G. et al. 2022, *Revisiting the Solar Research Cyberinfrastructure Needs*, arXiv:2203.09544.
* Liu N. et al. 2026, *FlareDB*, Scientific Data, doi:10.1038/s41597-026-06607-7.
* Poduval B. et al. 2023, *AI-ready data in space science and solar physics*, Front. Astron. Space Sci. 10, doi:10.3389/fspas.2023.1203598.
* NSF 26-512, *Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets)*.
* NASA ROSES-2022 B.16 *Heliophysics AI/ML-Ready Data* (NNH22ZDA001N-HARD).
