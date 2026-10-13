# CCSC Tool Inventory

Inventory of the deep-learning tools in the NJIT **Community Coordinated Software Center (CCSC)**,
<https://nature.njit.edu/solardb/ccsc>. Compiled 2026-10-13.

## 0. How this list was built (read first)

* **The CCSC page itself could not be opened from the cloud environment used for this work.**
  `nature.njit.edu` is denied by the environment's egress policy (HTTP 403 on CONNECT). The
  web-fetch tool also could not resolve it (DNS `ENOTFOUND`), and neither could
  `web.njit.edu`, `arxiv.org`, `zenodo.org`, `nature.com`, `iopscience.iop.org` or the Wayback Machine.
  So the **Citation buttons could not be clicked**.
* What *was* verified:
  * A search-engine snippet confirms the page title "**CCSC: Community Coordinated Software Center**". It also confirms the
    page text saying the tools' "Jupyter notebooks are now available on GitHub and Zenodo for interested researchers to view and execute".
  * The GitHub account **`ccsc-tools`** ("Community Coordinated Software Center") is public. Its profile
    lists **22 repositories**. All 22 were cloned (commit hashes in `tools/upstream_commits.csv`):
    19 tool repositories, the website stub `ccsc` (2021 landing page, no tool list), `deepsunutils`,
    and `zenodo_icons`. `zenodo_icons` holds the DOI badges for **13 tools** (CMETNet, Dst, FibrilNet, FlareML, Kp,
    LSTM-flare, RNN-CME, SDNN, SEP, SolarFlareNet, SolarUnet, SYMHnet, TSInet).
  * Citation data below comes from each repository's README/notebook "Citation"/"References" section
    (✅ verified in the repository) or from bibliographic search results (🔎 search-verified metadata; the full text
    was not readable here).
* **Not verified:** whether the live CCSC page lists exactly these 19 tools, in this order, with these
  citations. The six repositories created after Feb 2024 (MIST, SolarCNN, DeepHMI, SINet, LLMFlare, DeepGeoCME)
  have no Zenodo icon in `zenodo_icons` and may or may not appear on the page. **MagNet** (the tool the
  applicant's e-mail discussed) is *not* in `ccsc-tools`. It is distributed via `nature.njit.edu/solardb/magnet`
  (blocked) and is listed at the end as a related SolarDB tool.

Legend: ✅ verified from the repository/code · 🔎 bibliographic metadata from search results (not full text) · ❓ unknown/unverified.
Status labels follow the five categories requested: **Successfully Reproduced**, **Partially Reproduced**,
**Attempted but Unsuccessful**, **Not Attempted**, **Blocked by Missing Data or Compute Resources**.
"Reproduced" here always means *the scope stated in the cell*. Re-running a shipped demo is **not** treated as reproducing a paper.

## 1. Summary table

| # | Tool (repo) | Task | Model | Paper (venue, year) | Pretrained weights in repo? | Data in repo? | GPU needed? | Reproduction status (scope) |
|---|---|---|---|---|---|---|---|---|
| 1 | FlareML | 4-class flare class (B/C/M/X) from 13 SHARP params | RF, MLP, ELM, majority-vote ENS | Abduallah et al., RAA 21:160 (2021) 🔎 | ✅ | ✅ (845 samples + demo split) | No | **Partially** – demo inference reproduced; demo test set overlaps training |
| 2 | LSTM-flare-prediction | ≥M5.0/≥M/≥C flare in 24 h | LSTM + attention | Liu et al., ApJ 877:121 (2019) ✅ | ✅ (demo only) | ✅ (full Table 2/Table 5 data) | No | **Partially** – Table 2 retrained (10 seeds); Table 5 code re-run; ≥M5.0 demo test script inverts labels |
| 3 | RNN-CME-prediction | Does an M/X-flaring AR produce a CME (12–60 h) | GRU / LSTM + attention | Liu et al., ApJ 890:12 (2020) ✅ | ✅ | ✅ (2 test ARs) | No | **Successfully reproduced** (shipped-model outputs, 10/10 identical) |
| 4 | SolarFlareNet | ≥M5.0/≥M/≥C flare in 24/48/72 h | Conv1D + LSTM + 4 transformer-encoder blocks | Abduallah et al., Sci. Rep. 13:13665 (2023) ✅🔎 | ✅ | ✅ (2010–2022) | Optional | **Partially** – stored outputs reproduced; 8/9 shipped models show no skill; bundled test set overlaps training |
| 5 | SEP-prediction | SEP event given M/X flare (± CME), 12–72 h | BiLSTM + attention | Abduallah et al., ApJS 260:16 (2022) 🔎 | ✅ | ✅ | No | **Successfully reproduced** (12/12 stored metric rows exact) |
| 6 | CMETNet | CME Sun→Earth transit time (h) | RF, SVR, XGBoost, GPR + CNN on LASCO C2 | Alobaid et al., Front. Astron. Space Sci. 9:1013345 (2022) 🔎 | ❌ (trains in script) | ✅ (363 ICMEs, 351 FITS) | CNN: optional | **Partially** – numerical ensemble re-trained per hold-out year; CNN not run; test-set model selection quantified |
| 7 | Dst-prediction (DSTT) | Dst index 1–6 h ahead + uncertainty | Conv1D + LSTM + multi-head attention + variational (TFP) layers | Abduallah et al., FLAIRS-35 (2022) 🔎 | ✅ | ✅ | No | **Attempted but unsuccessful** – stored outputs match the released test path, but that path replaces predictions with 0.9×label; raw model R² 0.06–0.38 |
| 8 | Kp-prediction (KpNet) | Kp index 1–9 h ahead + uncertainty | Conv1D + BiLSTM + transformer + variational layers | Abduallah et al., ISMIS 2022, LNCS 13515 🔎 | ✅ | ✅ | No | **Successfully reproduced** (tool-level, within MC noise) |
| 9 | TSInet | Total solar irradiance reconstruction | Conv1D + LSTM + self-attention (autoregressive, periodic refit) | Abduallah et al., FLAIRS-34 (2021) 🔎 | ✅ | ✅ | No | **Partially** – default 30-day runs reproduced (stochastic); CLI bug; naive baseline beats it |
| 10 | SYMHnet | SYM-H index 1–2 h ahead, 1/5-min data | GNN + BiLSTM + Bayesian (MC dropout) | Abduallah et al., Space Weather 22 e2023SW003824 (2024) 🔎 | ✅ | ✅ (42 storms) | Optional | **Partially** – stored storm outputs re-run (see report); shipped test path is effectively 1-step-ahead |
| 11 | SolarUnet | Magnetic flux element segmentation + tracking (BBSO/GST) | U-Net + tracking post-processing | Jiang et al., ApJS 250:5 (2020) ✅ | ✅ | ✅ (3 demo frames) | No (inference) | **Successfully reproduced** (demo; masks byte-identical after fixing file ordering) |
| 12 | FibrilNet | Hα fibril tracing + uncertainty | Bayesian U-Net (MC dropout) | Jiang et al., ApJS 256:20 (2021) ✅ | ❌ downloaded at run time from web.njit.edu | ✅ (5 test, 242 train) | Training: yes | **Blocked** – model host unreachable |
| 13 | SDNN | Stokes inversion (LOS velocity, Doppler width) GST/NIRIS | Stacked 1-D CNN/DNN | Jiang et al., ApJ 939:66 (2022) ✅ | ❌ (web.njit.edu) | ❌ input .fts also on web.njit.edu | No | **Blocked** – model and input data unreachable |
| 14 | SolarCNN | Super-resolution of SOHO/MDI LOS magnetograms (HMI as truth) | Attention-aided CNN | Xu et al., Sol. Phys. 299:36 (2024) ✅ | ❌ (`variables.data` missing; Zenodo/Google Drive) | ✅ 3 samples | No | **Blocked** – weights missing; shipped outputs re-scored |
| 15 | DeepHMI | Super-resolution of HMI Bz/Bt with GST/NIRIS as truth | Conditional diffusion model (U-Net denoiser) | Xu et al., A&A (2025), doi:10.1051/0004-6361/202453581 ✅ | ✅ | ✅ 1 sample | Strongly recommended | see report (CPU run of 1,000-step sampler) |
| 16 | MIST | ≥M flare in 24 h | Patch time-series transformer (MTST) | ❓ Chaudhary, Abduallah, Wang (2025; Zenodo 16780646 per LLMFlare README) | ✅ | ✅ test windows only (`X_train` absent) | No | **Successfully reproduced** reported TSS 0.906; **out-of-sample TSS 0.20** |
| 17 | SINet | Daily F10.7 & F30 forecasts, 1–60 days | TimesNet-style (TimesBlock + Inception) | Wang Z. et al., JGR Space Physics (2026), doi:10.1029/2025JA034868 ✅ | ✅ | ✅ | No | **Successfully reproduced** (F10.7 notebook numbers exact; F30 see report) |
| 18 | LLMFlare | Natural-language explanations of flare predictions | MIST + SHAP + Llama-3.1-8B LoRA (r=32), Gemini-2.5-Pro judge | ❓ (no citation in README) | ✅ LoRA adapter only | ✅ | Yes (A100 for fine-tuning) | **Not attempted / Blocked** – gated Llama weights, paid Gemini API, GPU |
| 19 | DeepGeoCME | Is a CME geoeffective? (SOHO + SDO) | Fusion: ResNet-50 + EfficientNet-B0 + attention | Yan, Abduallah, Wang, arXiv:2605.24748 (2026) ✅ | ✅ (GitHub release, SHA-256 verified) | ✅ (33 test events) | No (inference) | see report |
| – | MagNet (SolarDB, not in ccsc-tools) | Transverse field (Bx, By) for SOHO/MDI from LOS + Hα | CNN (see paper) | Jiang et al., Sol. Phys. 298:87 (2023), doi:10.1007/s11207-023-02180-z 🔎; out-of-sample validation arXiv:2601.15926 🔎 | ❓ on nature.njit.edu | ❓ on nature.njit.edu | ❓ | **Blocked** – distribution host unreachable |

(The DeepGeoCME and DeepHMI cells are finalised in `reports/CCSC_Reproduction_Report.md`, which is the
authoritative status document.)

## 2. Per-tool records (16 fields)

Fields: 1 name · 2 research question · 3 paper title · 4 citation · 5 paper link/DOI · 6 source code · 7 dataset ·
8 algorithm · 9 Python & dependencies · 10 pretrained weights · 11 notebook · 12 GPU · 13 data size & download needs ·
14 reported results · 15 reproducibility & difficulties · 16 current status.

### 2.1 FlareML
1. **FlareML** (backend of the DeepSun MLaaS service, `nature.njit.edu/spacesoft/DeepSun/`).
2. Given 13 SHARP magnetic parameters of an active region (AR) at 00:00 on a flare day, classify the day's flare as B, C, M or X.
3. "DeepSun: machine-learning-as-a-service for solar flare prediction".
4. Abduallah Y., Wang J. T. L., Nie Y., Liu C., Wang H. 2021, *Research in Astronomy and Astrophysics* 21(7):160 🔎 (author list from search; README lists Abduallah, Wang, Wang). Notebook citation: "Abduallah, Wang & Wang, Predicting Solar Flares with Machine Learning, github.com/ccsc-tools/FlareML" ✅.
5. doi:10.1088/1674-4527/21/7/160 ✅; arXiv:2009.04238 🔎; Zenodo doi:10.5281/zenodo.5634114 ✅.
6. github.com/ccsc-tools/FlareML (commit 44ffd7a) ✅; newer: github.com/deepsuncode/Machine-learning-as-a-service.
7. `data/original_data/flaringar_original_data.csv` (845 flares, 2010-05-03 → 2016-12-27, 472 ARs), a 435-row training sample, and a 40-row demo test set ✅.
8. Random forest, MLP, extreme learning machine (`sklearn_extensions`), and a majority vote of the three (ties go to RF) ✅.
9. Python 3.9, scikit-learn 0.24.2, sklearn-extensions 0.0.2, numpy 1.19.5. The pickles were written by scikit-learn 0.23.2 ✅.
10. Yes: `models/default_model_{rf,mlp,elm}.sav` ✅.
11. Yes: two notebooks, Binder-enabled ✅.
12. No.
13. < 1 MB data; 49 MB models.
14. The repo's notebook/log reports only the demo bar charts. Paper numbers could not be read (full text blocked) ❓.
15. Demo inference reproduces the shipped log exactly (ENS acc 0.850, RF 0.825, ELM 0.725; MLP collapses to class C, acc 0.300, as in the authors' log).
    **Caveats:** (a) `check_pm_precision()` replaces the *displayed* RF per-class score by "ENS − U(0.01, 0.07)" whenever RF ≥ ENS.
    (b) `log_cv_report()` alters confusion-matrix cells when FP or FN = 0.
    (c) MLP/ELM inputs are standardised with a scaler fitted on the *test batch*.
    (d) **31 of the 40 demo test rows also appear in the training sample** (all 13 features within 0.6 %, same class), so the demo score is not out-of-sample.
16. **Partially reproduced** (demo inference; paper experiment not re-run).

### 2.2 LSTM-flare-prediction
1. **LSTM-flare-prediction** ("FlarePredict").
2. Will an AR produce a ≥M5.0 / ≥M / ≥C flare within the next 24 h?
3. "Predicting Solar Flares Using a Long Short-term Memory Network".
4. Liu H., Liu C., Wang J. T. L., Wang H. 2019, *ApJ* 877:121 ✅.
5. doi:10.3847/1538-4357/ab1b3c ✅; arXiv:1905.07095 ✅; Zenodo record 5652615 🔎; web.njit.edu/~wangj/LSTMpredict/.
6. github.com/ccsc-tools/LSTM-flare-prediction (12fc8b9) ✅ (TF2 port); original paper code in `Table5_dataset_and_source_code/LSTMpredict.py` (TF 1.12) ✅.
7. `Table2_dataset/{C,M,M5}`: train 84,577 rows (2010-05 → 2014-01), validation 26,473 rows (2014), test 44,689 rows (2015 → 2018-05), with **no shared HARP/NOAA ARs**.
   `Table5_dataset_and_source_code/C` keeps the incomplete samples. "Collection of Flares" lists 4,203 B, 6,768 C, 704 M and 49 X flares ✅.
8. One LSTM layer (10 units) → temporal attention → Dense(200) → Dense(500) → softmax. Inputs are 10-step series of 14 (≥C), 22 (≥M) or 20 (≥M5.0) of 40 features (25 SHARP + 15 flare-history) ✅.
9. TF/Keras 2.10, scikit-learn 1.2 (paper: TF 1.12, Keras 2.2.4) ✅.
10. Yes, but only for the small "sample_run" demo (one test AR per class) ✅.
11. Yes (`ccsc_FlarePredict.ipynb`; Binder) ✅.
12. No (one training run takes about 26–45 s on 2 CPU threads).
13. ≈ 1 GB repository; all data included.
14. Table 5 reference output for ≥C (`output.txt`): recall 0.773, precision 0.541, ACC 0.826, BACC 0.806, HSS 0.526, **TSS 0.612 (±0.009)** ✅. Other paper tables: ❓ (full text blocked).
15. See report. Key points:
    * The released `flarepredict_test.py` treats softmax column 0 as "positive". For ≥M5.0 the LabelEncoder order is ['N','X'], so **≥M5.0 predictions are inverted**: TSS −0.671. Using column 1 gives TSS 0.686 and 98.9 % agreement with the authors' stored output.
    * The paper's Table 5 picks the decision threshold that maximises average TSS **on the test folds**.
    * The previous-timestep missing-value check tests `row` instead of `prev_row`, so incomplete history rows are never masked.
16. **Partially reproduced.** Retraining on Table 2 (10 seeds): TSS 0.877 ± 0.035 (≥M5.0), 0.748 ± 0.021 (≥M), 0.589 ± 0.009 (≥C) with the test-optimised threshold; lower with a leak-free validation threshold. The Table 5 protocol was re-run (see report).

### 2.3 RNN-CME-prediction
1. **RNN-CME-prediction** ("CMEPredict").
2. Given that an AR produces an M- or X-class flare, will that flare be associated with a CME within 12/24/36/48/60 h?
3. "Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks".
4. Liu H., Liu C., Wang J. T. L., Wang H. 2020, *ApJ* 890:12 ✅.
5. doi:10.3847/1538-4357/ab6850 ✅; arXiv:2002.10953 ✅.
6. github.com/ccsc-tools/RNN-CME-prediction (04be43f) ✅.
7. 18 SHARP parameters; labels from GOES flares (NCEI) plus flare–CME associations (DONKI). The test set covers **only 2 ARs (12497, 12529)** ✅.
8. GRU or LSTM (10 units) + attention + dense layers; per-model fixed thresholds of 0.40–0.50 ✅.
9. TF/Keras 2.8–2.11 ✅.
10. Yes (10 × `.h5`) ✅.
11. Yes ✅.
12. No.
13. ≈ 290 MB.
14. Stored outputs: TSS 0.82–1.00 on the 2-AR test set ✅.
15. Predicted labels are **identical** (10/10 settings) to the stored outputs. The very small, 2-AR test set means the TSS has a large sampling uncertainty.
16. **Successfully reproduced** (shipped-model inference). The paper's cross-validation was not attempted.

### 2.4 SolarFlareNet
1. **SolarFlareNet** (operational on the Web; also runs at NASA CCMC per a 2026 arXiv comparison 🔎).
2. Will an AR produce a ≥M5.0 / ≥M / ≥C flare within 24 / 48 / 72 h?
3. "Operational prediction of solar flares using a transformer-based framework".
4. Abduallah Y., Wang J. T. L., Wang H., Xu Y. 2023, *Scientific Reports* 13:13665 ✅🔎.
5. doi:10.1038/s41598-023-40884-1 ✅; Zenodo doi:10.5281/zenodo.10080716 ✅.
6. github.com/ccsc-tools/SolarFlareNet (1bf04b8) ✅.
7. SHARP time series; training files 2010–2022-09, testing files 2022-01 → 2022-09 (100 HARPs) ✅.
8. BatchNorm → Conv1D(32) → LSTM(400) → 4 × [multi-head attention + Conv1D + LSTM(400)] → Dense(200) → Dense(500) → softmax ✅.
9. TF 2.8 / Keras 2.6 (ran on TF 2.10) ✅.
10. Yes (9 checkpoints, 74 MB each) ✅.
11. Yes ✅.
12. Optional.
13. ≈ 1.9 GB.
14. Paper metrics ❓ (full text blocked). Shipped `result/` CSVs ✅.
15. Re-run outputs agree ≥ 99.98 % with the stored result CSVs for 8/9 settings. **Those 8 shipped models have AUC 0.40–0.53 (no skill) on the bundled test files.**
    The C/24 h weights were retrained after the stored results were produced (see the shipped 2023-11-05 log) and give TSS 0.79. However, **84 % of the bundled 24 h test rows also occur (same HARP and timestamp) in the training file**; on the non-overlapping rows TSS is 0.697.
    The loader selects features by position using another dataset's column order (off by one: TOTUSJH is never used). The 9 features actually fed are TOTPOT, ABSNJZH, USFLUX, AREA_ACR, MEANPOT, R_VALUE, MEANGAM, MEANJZH, MEANJZD.
16. **Partially reproduced** (stored outputs reproduced; paper-level skill not reproducible with the shipped artefacts).

### 2.5 SEP-prediction
1. **SEP-prediction**.
2. Will an AR that produces an M/X flare (FC_S: one with a CME; F_S: regardless of CME) produce a solar energetic particle event within 12–72 h?
3. "Predicting Solar Energetic Particles Using SDO/HMI Vector Magnetic Data Products and a Bidirectional LSTM Network".
4. Abduallah Y., Jordanova V. K., Liu H., Li Q., Wang J. T. L., Wang H. 2022, *ApJS* 260:16 🔎.
5. doi:10.3847/1538-4365/ac5f56 🔎; arXiv:2203.14393 🔎; Zenodo record 7516609 ✅.
6. github.com/ccsc-tools/SEP-prediction (119ea98) ✅.
7. SHARP parameters; training 2011–2016, test Apr–Sep 2017. The test set has **3 ARs, of which 2 (12665, 12673) produce all positives** ✅.
8. BiLSTM + attention + dense layers; isotonic-regression calibration ✅.
9. TF 2.10 ✅.
10. Yes (`default_models`, `models`) ✅.
11. Yes ✅.
12. No.
13. ≈ 1.4 GB.
14. Stored metrics (`default_results`): TSS 0.86–0.99 ✅.
15. **All 12 stored metric rows reproduced exactly.** Caveats: the effective sample is 2 SEP-producing ARs, and the calibrated Brier scores (BSC/BSSC) use an isotonic regression fitted on the test set itself.
16. **Successfully reproduced** (tool-level).

### 2.6 CMETNet
1. **CMETNet**.
2. How many hours will a geoeffective CME take to travel from the Sun to Earth?
3. "Predicting CME arrival time through data integration and ensemble learning".
4. Alobaid K. A., Abduallah Y., Wang J. T. L., Wang H., Jiang H., Xu Y., Yurchyshyn V., Zhang H., Cavus H., Jing J. 2022, *Frontiers in Astronomy and Space Sciences* 9:1013345 🔎.
5. doi:10.3389/fspas.2022.1013345 🔎; Zenodo record 7516684 ✅.
6. github.com/ccsc-tools/CMETNet (cfc7e0f) ✅.
7. `ICME_list.csv`: 363 ICMEs, 1996–2021, with CME and solar-wind features; 351 LASCO C2 FITS images ✅.
8. Median ensemble of RF (800 trees), SVR (RBF), XGBoost and GPR, combined with a CNN on C2 images ✅.
9. TF 2.11, scikit-learn 1.2, xgboost 1.5 ✅.
10. No (the scripts train). `pretrained_models_results.py` only plots **hard-coded** numbers ✅.
11. Yes ✅.
12. CNN: optional.
13. ≈ 1 GB.
14. Paper/README: MAE 9.75 h, Pearson r 0.83 (CMETNet); COMB MAE 10.32 h ✅ (hard-coded in the plotting script).
15. The released training script **selects, among 10 random 200-event training subsamples, the model with the lowest MAE on the held-out test year**. The quantified optimism is in the report.
16. **Partially reproduced** (numerical ensemble only).

### 2.7 Dst-prediction (DSTT)
1. **Dst Transformer (DSTT)**.
2. Forecast the Dst geomagnetic index 1–6 h ahead from 7 solar-wind parameters, with aleatoric and epistemic uncertainty.
3. "Forecasting the Disturbance Storm Time Index with Bayesian Deep Learning" (arXiv title: "A Deep Learning Approach to Dst Index Prediction").
4. Abduallah Y., Wang J. T. L., Bose P., Zhang G., Gerges F., Wang H. 2022, *Proc. FLAIRS-35* 🔎.
5. doi:10.32473/flairs.v35i.130564 🔎; arXiv:2205.02447 🔎; Zenodo record 7529705 ✅.
6. github.com/ccsc-tools/Dst-prediction (e5ae919) ✅.
7. OMNIWeb hourly solar wind + Dst; test window Oct–Nov 2021 (1,103 h) ✅.
8. Conv1D → LSTM(650) → multi-head attention → TFP variational dense layers ✅.
9. TF 2.10 + TFP 0.17 ✅.
10. Yes (1–6 h) ✅.
11. Yes ✅.
12. No.
13. ≈ 730 MB.
14. Stored `default_results`: RMSE ≈ 3.3–3.9 nT, r ≈ 0.98 ✅. Paper ❓.
15. **The released `DSTTModel.predict()` passes predictions through `DSTT_utils.t()`.** Whenever a prediction is not ≥ 90 % away from the true value, it is replaced by **0.9 × true label**; 88–90 % of the stored "predictions" equal 0.9 × label exactly.
    In addition, `DSTT_test.test()` loads the 6 h weights for every horizon. The loader also appends the Oct–Nov 2021 test rows to the training frame (an `isin` filter on partial strings never matches).
    The raw network (MC mean of 50 passes, correct weights) gives R² 0.06–0.38 (RMSE 13–16 nT), worse than persistence at every horizon (R² 0.93 → 0.53).
16. **Attempted but unsuccessful** (the reported skill is not supported by the raw model).

### 2.8 Kp-prediction (KpNet)
1. **KpNet**.
2. Forecast the Kp index 1–9 h ahead with uncertainty.
3. "A Transformer-Based Framework for Geomagnetic Activity Prediction".
4. Abduallah Y., Wang J. T. L., Xu C., Wang H. 2022, *ISMIS 2022*, LNCS 13515, pp. 325–335 🔎.
5. Zenodo record 7514451 ✅.
6. github.com/ccsc-tools/Kp-prediction (8e473da) ✅.
7. Hourly solar wind + Kp; training 2010–2021, test Jan–Mar 2022 (no shared timestamps) ✅.
8. Conv1D + BiLSTM + transformer encoder + TFP variational layers ✅.
9. TF 2.10 + TFP 0.17 ✅.
10. Yes (1–9 h) ✅.
11. Yes ✅.
12. No.
13. ≈ 710 MB.
14. Stored results: e.g. 1 h R² 0.625, r 0.801 ✅.
15. Reproduced within the model's stochastic noise; `predict()` does not use labels. Notes: persistence is better at 1–2 h. The reported "prediction" in `uncertainty()` is the minimum over MC samples, and the uncertainty magnitudes carry ad-hoc scale factors.
16. **Successfully reproduced** (tool-level).

### 2.9 TSInet
1. **TSInet**.
2. Reconstruct total solar irradiance (TSI) beyond the range of physics-based models.
3. "Reconstruction of Total Solar Irradiance by Deep Learning" (arXiv: "Deep Learning Based Reconstruction of Total Solar Irradiance").
4. Abduallah Y., Wang J. T. L., Shen Y., Alobaid K. A., Criscuoli S., Wang H. 2021, *Proc. FLAIRS-34* 🔎.
5. doi:10.32473/flairs.v34i1.128356 🔎; arXiv:2107.11042 🔎; Zenodo record 7516668 ✅.
6. github.com/ccsc-tools/TSInet-irradiance-reconstruction (8eb42cc) ✅.
7. Train: SORCE TSI (2003–2020); test: TCTE, NRLTSI2, SATIRE-S, SATIRE-M ✅.
8. Conv1D ×3 → max-pool → LSTM → self-attention → dense; autoregressive generation with periodic re-fitting on its own predictions ✅.
9. TF 2.10, keras-self-attention 0.51 ✅.
10. Yes ✅.
11. Yes ✅.
12. No.
13. 30 MB.
14. Stored 30-day outputs (RMSE 0.62–0.72 W m⁻² vs the reference series, computed by us) ✅.
15. RMSE reproduced (0.70–0.79). Caveats:
    * The CLI ignores `-n` (a missing `global`), so it always reconstructs TCTE.
    * Rescaling uses the min/max of the target file.
    * A "repeat the last observed value" baseline achieves 0.09–0.23 W m⁻².
16. **Partially reproduced.**

### 2.10 SYMHnet
1. **SYMHnet**.
2. Forecast the SYM-H index 1 or 2 h ahead from 1-min or 5-min solar-wind data, with uncertainty.
3. "Prediction of the SYM-H Index Using a Bayesian Deep Learning Method With Uncertainty Quantification".
4. Abduallah Y., Alobaid K. A., Wang J. T. L., Wang H., Jordanova V. K., Yurchyshyn V., Cavus H., Jing J. 2024, *Space Weather* 22(2), e2023SW003824 🔎.
5. doi:10.1029/2023SW003824 🔎; arXiv:2402.17196 🔎; Zenodo doi:10.5281/zenodo.10602518 ✅.
6. github.com/ccsc-tools/SYMHnet (f4583d9) ✅.
7. 42 storms (1998–2018); storms 1–25 for training, 26–42 for testing ✅.
8. Graph neural network + BiLSTM, MC-dropout uncertainty ✅ (README).
9. TF 2.8 (ran on 2.10) ✅.
10. Yes ✅.
11. Yes ✅.
12. Optional.
13. ≈ 3.7 GB.
14. README: FSS 0.343 (1 h) / 0.553 (2 h) vs GBM 0.074 / 0.087 for storm 37 (5-min) ✅.
15. The 1 h and 2 h data folders are **byte-identical**, and SYM_H equals SYMH_ORIG (no shift). The test windowing (`lag=1`, `forecast_horizon=1`) targets the **next row**, i.e. the released test path predicts 1 or 5 minutes ahead, not 1–2 h. The output filename uses `start_hour`, so multi-horizon calls overwrite the 1 h file.
16. **Partially reproduced** (numbers in the report).

### 2.11 SolarUnet
1. **SolarUnet**.
2. Identify and track signed magnetic flux elements in BBSO/GST vector magnetograms (an alternative to SWAMIS).
3. "Identifying and Tracking Solar Magnetic Flux Elements with Deep Learning".
4. Jiang H., Wang J., Liu C., Jing J., Liu H., Wang J. T. L., Wang H. 2020, *ApJS* 250:5 ✅.
5. doi:10.3847/1538-4365/aba4aa ✅; arXiv:2008.12080 ✅.
6. github.com/ccsc-tools/SolarUnet-magnetic-tracking (551dcd4) ✅.
7. BBSO/GST frames (3 demo frames, training images/labels from SWAMIS) ✅.
8. U-Net (720×720) + post-processing + forward/backward tracking ✅.
9. TF 2.10, OpenCV, astropy ✅.
10. Yes (90 MB) ✅.
11. Yes ✅.
12. No (inference).
13. ≈ 370 MB.
14. Stored demo outputs ✅.
15. Masks are byte-identical once file ordering is fixed. The tool iterates `os.listdir()` without sorting; on Linux this returned frame_1, frame_3, frame_2 and mis-paired masks with FITS frames.
16. **Successfully reproduced** (demo).

### 2.12 FibrilNet
1. **FibrilNet**.
2. Trace chromospheric fibrils in Hα images with per-pixel uncertainty.
3. "Tracing Hα Fibrils through Bayesian Deep Learning".
4. Jiang H. et al. 2021, *ApJS* 256:20 ✅.
5. doi:10.3847/1538-4365/ac14b7 ✅; arXiv:2107.07886 ✅; Zenodo record 7521568 ✅.
6. github.com/ccsc-tools/FibrilNet (ba1764b) ✅.
7. 5 test images (AR 12665, 2017-07-13), 242 training pairs ✅.
8. Bayesian U-Net (MC dropout) + fibril fitting ✅.
9. TF ✅.
10. **No**: `utils.download_model()` fetches `web.njit.edu/~wangj/deepsuncode/FibrilNet/model.hdf5` ✅.
11. Yes ✅.
12. Training: yes (10 × 1,000 steps at 720×720).
13. 173 MB repository + model download.
14. Stored masks and uncertainty maps ✅.
15. The model host is unreachable from this environment.
16. **Blocked by missing data or compute resources.**

### 2.13 SDNN
1. **SDNN**.
2. Infer LOS velocity and Doppler width (plus Bx, By, Bz) from GST/NIRIS Stokes profiles, replacing slow Milne–Eddington inversion.
3. "Inferring Line-of-sight Velocities and Doppler Widths from Stokes Profiles of GST/NIRIS Using Stacked Deep Neural Networks".
4. Jiang H., Li Q., Xu Y., Hsu W., Ahn K., Cao W., Wang J. T. L., Wang H. 2022, *ApJ* 939:66 ✅.
5. doi:10.3847/1538-4357/ac927e ✅; arXiv:2210.04122 ✅; Zenodo record 7521580 ✅.
6. github.com/ccsc-tools/SDNN (dc44cec) ✅.
7. **Input `cals_170713_162012.fts` is downloaded from web.njit.edu**; only output FITS files ship ✅.
8. Stacked 1-D CNN/DNN (16 Conv1D layers) ✅.
9. TF ✅.
10. **No** (web.njit.edu) ✅.
11. Yes ✅.
12. No.
13. Unknown input size.
14. Stored output maps ✅.
15. Model and input are both unreachable.
16. **Blocked by missing data or compute resources.**

### 2.14 SolarCNN
1. **SolarCNN**.
2. Enhance (super-resolve) SOHO/MDI LOS magnetograms of ARs using SDO/HMI as ground truth, to give cycle-23 data HMI-like quality.
3. "Super-Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using SDO/HMI Data and an Attention-Aided Convolutional Neural Network".
4. Xu C., Wang J. T. L., Wang H., Jiang H., Li Q., Abduallah Y., Xu Y. 2024, *Solar Physics* 299:36 ✅.
5. doi:10.1007/s11207-024-02283-1 ✅; arXiv:2403.18302 🔎; Zenodo doi:10.5281/zenodo.15250172 ✅.
6. github.com/ccsc-tools/SolarCNN (52d4fa7) ✅.
7. 3 MDI/HMI 256×256 pairs + 3 stored predictions; full set on Zenodo ✅.
8. Attention-aided CNN; mixed SSIM/L1 loss ✅.
9. TF 2.6 + tensorflow-addons ✅.
10. **Incomplete**: the SavedModel lacks `variables.data-*` (Zenodo / Google Drive) ✅.
11. Yes ✅.
12. No.
13. 7.6 MB repository.
14. Paper: SSIM, PCC and PSNR improvements ❓ (numbers not readable).
15. The weights are missing. `test.py` omits the training normalisation (clip ±2000 G, ÷2000). The shipped predictions are on a different amplitude scale (std 65 G vs 197 G). After the same oracle linear calibration, they are not better than the input MDI image on the 3 samples.
16. **Blocked by missing data or compute resources** (author outputs re-scored).

### 2.15 DeepHMI
1. **DeepHMI**.
2. Super-resolve SDO/HMI transverse and LOS magnetograms to GST/NIRIS resolution.
3. "Improving the Spatial Resolution of SDO/HMI Transverse and Line-of-Sight Magnetograms Using GST/NIRIS Data with Machine Learning".
4. Xu C., Xu Y., Wang J. T. L., Li Q., Wang H. 2025, *Astronomy & Astrophysics* ✅.
5. doi:10.1051/0004-6361/202453581 ✅.
6. github.com/ccsc-tools/DeepHMI (92d3ebb) ✅.
7. One paired HMI/NIRIS test FITS (AR 12371, 2015-06-22) ✅.
8. Conditional denoising diffusion model (1,000 steps, U-Net with attention) ✅.
9. PyTorch 1.12.1, einops, astropy<5 ✅.
10. Yes (`model_bz`, `model_bt`, 28 MB each) ✅.
11. Yes ✅.
12. Strongly recommended (1,000 denoising steps at 720×720).
13. 134 MB.
14. Paper: better than bicubic on 4 metrics ❓ (numbers not readable).
15. The output is stochastic, so we compare metrics, not pixels.
16. See report.

### 2.16 MIST
1. **MIST** (the model class is called `MTST`, a multi-patch time-series transformer).
2. Will an AR produce a ≥M flare within 24 h?
3. ❓ The README notebook says "A New Deep Learning Method for Solar Flare Prediction" (Chaudhary, Abduallah, Wang; 2025-08-06). LLMFlare's README points to Zenodo record 16780646 and "the FLAIRS paper". A FLAIRS-39 (2026) paper by Chaudhary, Wang, Xu, Xu and Zhang exists 🔎, but its link to MIST is unverified.
4. ❓
5. Zenodo record 16780646 (per LLMFlare README) ❓.
6. github.com/ccsc-tools/MIST (7cc02f1) ✅.
7. `processed_solar_dataset.csv` = **exactly the LSTM-flare Table 2 ≥M training file** (2010-05 → 2014-01; 84,577 rows; 16 features). Windows of 24 rows are labelled by row p+24 and **split 80/20 at random**; the shipped StandardScaler was fitted on all 84,577 rows ✅ (all verified).
8. Patching (2 h patches) → linear embedding → 3-layer transformer encoder (d=128, 8 heads) → pooling → sigmoid ✅.
9. PyTorch 2.3 (ran on 1.12.1) ✅.
10. Yes ✅.
11. Yes (`README.ipynb`) ✅.
12. No.
13. 142 MB.
14. README: TSS 0.90 ✅.
15. TSS 0.9062 reproduced exactly. However, **95.6 % of test windows have a 23-of-24-hour-overlapping neighbour in the training set**. On the chronologically and AR-independent 2015–2018 Table 2 test file, the same model gives **TSS 0.197** (AUC 0.83), versus TSS ≈ 0.73 for the 2019 LSTM on the same data.
16. **Successfully reproduced** (the reported number); the out-of-sample evaluation reveals leakage.

### 2.17 SINet
1. **SINet** (Solar Index Network).
2. Predict daily F10.7 and F30 radio-flux indices 1–60 days ahead.
3. "Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning".
4. Wang Z., Abduallah Y., Wang J. T. L., Wang H., Xu Y., Yurchyshyn V., Oria V., Alobaid K. A., Bai X. 2026, *JGR Space Physics* ✅.
5. doi:10.1029/2025JA034868 ✅.
6. github.com/ccsc-tools/SINet (8017d8d) ✅.
7. NOAA F10.7; Toyokawa/Nobeyama F30; test 2009–2021 ✅.
8. TimesNet-style blocks (FFT period discovery + 2-D Inception convolutions) ✅.
9. PyTorch ≥1.12 ✅.
10. Yes (8 models) ✅.
11. Yes, with stored outputs ✅.
12. No.
13. 150 MB.
14. Notebook: F10.7 RMSE 3.880 / 13.741 / 15.092 / 16.324 for 1 / 27 / 45 / 60 d ✅.
15. F10.7 reproduced to 4 decimals.
16. **Successfully reproduced** (tool-level).

### 2.18 LLMFlare
1. **LLMFlare**.
2. Generate natural-language explanations, recommendations and counterfactuals for flare predictions.
3. ❓ (README gives none).
4. ❓
5. ❓
6. github.com/ccsc-tools/LLMFlare (2a2403b) ✅.
7. MIST test windows (`X_test.npy` byte-identical to MIST's), SHAP values, 50,733 synthetic narratives, 200 evaluation examples ✅.
8. MIST predictor → SHAP → LoRA-fine-tuned Llama-3.1-8B; LLM-as-a-judge with Gemini 2.5 Pro ✅.
9. Python 3.11, transformers, peft, trl, bitsandbytes ✅.
10. LoRA adapter only (base model gated on Hugging Face) ✅.
11. No notebook ✅.
12. Yes (A100 for fine-tuning; inference on CPU is impractical for 8B).
13. 416 MB repository + 16 GB base model.
14. README: faithfulness 8.04 ± 2.96 (fine-tuned) vs 1.85 ± 1.99 (zero-shot) ✅.
15. Needs a Hugging Face token with Llama access, **a paid Google Gemini API key**, and a GPU, which requires the applicant's authorisation. It inherits MIST's split issue.
16. **Not attempted** (blocked pending authorisation and GPU).

### 2.19 DeepGeoCME
1. **DeepGeoCME**.
2. Is a CME geoeffective? (binary)
3. "Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations".
4. Yan Z., Abduallah Y., Wang J. T. L. 2026, arXiv:2605.24748 ✅.
5. arXiv:2605.24748 ✅; Zenodo record 22955090 ✅.
6. github.com/ccsc-tools/DeepGeoCME (6daa000), upstream github.com/xysgszx/DeepGeoCME ✅.
7. Test tensors for 33 events (GitHub release, 144 MB) ✅.
8. Multimodal fusion: ResNet-50 and EfficientNet-B0 branches + attention; decision threshold 0.6 ✅.
9. PyTorch ≥2.6, timm (ran on 1.12.1 with a `torch.load` shim) ✅.
10. Yes (GitHub release, 199 MB; SHA-256 verified) ✅.
11. Yes (Colab) ✅.
12. No.
13. ≈ 350 MB downloads.
14. Shipped `confusion_matrix.png` ✅.
15. See report.
16. See report.

### Related SolarDB tool: MagNet
* Jiang H., Li Q., Liu N., Hu Z., Abduallah Y., Jing J., Xu Y., Wang J. T. L., Wang H. 2023, *Solar Physics* 298:87, doi:10.1007/s11207-023-02180-z; arXiv:2211.02278 🔎.
  It generates SOHO/MDI transverse fields (Bx, By) from MDI LOS + BBSO Hα, trained with SDO/HMI vector magnetograms.
* Yesmanchyk, Xu, Wang, Jiang, Xu, Wang 2026, "Out-of-Sample Validation of MagNet", arXiv:2601.15926 🔎. It validates on Mees/IVM data not used for training.
* Code, model and data are distributed at `nature.njit.edu/solardb/magnet` (search snippets show directory listings such as `/solardb/magnetdata/MagNet/Validate_HMI/`) and are **blocked** in this environment.
