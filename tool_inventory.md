# CCSC Tool Inventory

Inventory of the deep-learning tools in the NJIT **Community Coordinated Software Center (CCSC)**,
<https://nature.njit.edu/solardb/ccsc>. Compiled 2026-10-13.

## 0. How this list was built (read first)

* **The CCSC page was opened and its Citation buttons were followed** (2026-10-13, after the applicant enabled network access).
  * The page builds its 19 tiles from `ccsc_pages/pages_data.json`. Each tile title links to `github.com/ccsc-tools/<repo>`, and each **Citation** button links to a **Zenodo record**.
  * The 19 tiles are exactly the 19 tool repositories in the `ccsc-tools` GitHub account. Page snapshot and JSON: `papers/citations/`.
* `zenodo.org` returned HTTP 504 (gateway time-out), so each Zenodo DOI was resolved through DataCite: APA text, BibTeX and full metadata (`papers/citations/ccsc_citations.bib`, `datacite_*.json`).
* **For 13 tools the cited Zenodo record is a GitHub release tag.** That version was run, or diffed against HEAD (`results/cited_versions/cited_tag_vs_HEAD_diffstat.txt`).
* **For 6 tools it is a direct Zenodo upload** (SolarCNN, MIST, DeepHMI, SINet, LLMFlare, DeepGeoCME). It could not be downloaded, so the GitHub HEAD was run, and whether the two are identical is unverified.
* **Papers.** Full texts came from arXiv (14 used for comparison), *Scientific Reports* (SolarFlareNet) and *Frontiers* (CMETNet). Not accessible: A&A (DeepHMI), the Springer LNCS chapter (KpNet), and AGU/Wiley (SINet's JGR version; its arXiv version was used).
* **MagNet** is not on the CCSC page. It is distributed at `nature.njit.edu/solardb/magnet` ("Un-published code") and is listed at the end as a related SolarDB tool.

Legend:
* ✅ verified from the repository, page or paper;
* 🔎 bibliographic metadata only;
* ❓ unknown or unverified.

Status labels follow the five categories requested: **Successfully Reproduced**, **Partially Reproduced**, **Attempted but Unsuccessful**, **Not Attempted** and **Blocked by Missing Data or Compute Resources**.
* *Tool level* means that the released model and data reproduce the authors' stored outputs.
* *Paper level* means that a headline number of the paper is reproduced.
* Re-running a shipped demo is **not** treated as reproducing a paper.

## 1. Summary table

| # (page order) | Tool (repo) | Task | Model | Paper (venue, year) | Citation button → Zenodo (cited version) | Weights available? | GPU needed? | Status (tool level / paper level) |
|---|---|---|---|---|---|---|---|---|
| 1 | SolarUnet-magnetic-tracking | Flux-element segmentation + tracking (BBSO/GST) | U-Net + tracking | Jiang et al., ApJS 250:5 (2020) ✅ | 10.5281/zenodo.5643647 (v.1.2) | ✅ repo | No | **Successfully reproduced** at tool level (masks byte-identical after sorting); paper statistics use 147 frames, 3 shipped |
| 2 | FlareML | 4-class flare class from 13 SHARP params | RF, MLP, ELM, vote | Abduallah et al., RAA 21:160 (2021) ✅ | 10.5281/zenodo.7506997 (v1.5) | ✅ repo | No | **Partially** – demo exact; paper Table III avg TSS 0.488 (RF) vs our RF 10-fold 0.338 / AR-grouped 0.246 |
| 3 | RNN-CME-prediction | CME given an M/X flare, 12–60 h | GRU / LSTM + attention | Liu et al., ApJ 890:12 (2020) ✅ | 10.5281/zenodo.7415731 (v1.3) | ✅ repo | No | **Partially** – demo 10/10 identical; on the paper's own test set GRU TSS 0.31–0.49 vs paper 0.45–0.60; column-order inconsistency between the two data copies |
| 4 | LSTM-flare-prediction | ≥M5.0/≥M/≥C flare in 24 h | LSTM + attention | Liu et al., ApJ 877:121 (2019) ✅ | 10.5281/zenodo.5739321 (v1.0.2) | ✅ repo (demo) | No | **Partially** – cited demo models reproduce stored outputs; retrain at paper thresholds TSS 0.812/0.730/0.580 vs paper 0.877/0.790/0.607; Table 5 ≥C CV 0.581 vs 0.612 |
| 5 | SEP-prediction | SEP given an M/X flare (± CME), 12–72 h | BiLSTM + attention | Abduallah et al., ApJS 260:16 (2022) ✅ | 10.5281/zenodo.7516609 (v1.4) | ✅ repo | No | **Successfully reproduced** at tool level (12/12 rows exact); paper CV means not re-runnable from the release |
| 6 | Dst-prediction (DSTT) | Dst 1–6 h ahead + UQ | Conv1D+LSTM+attention+TFP | Abduallah et al., FLAIRS-35 (2022) ✅ | 10.5281/zenodo.7529705 (v1.1) | ✅ repo | No | **Attempted but unsuccessful** – released path uses 0.9×label; raw model R² 0.06–0.38 vs paper ≈ 0.59–0.75 |
| 7 | TSInet | Total solar irradiance reconstruction | Conv1D+LSTM+attention, autoregressive | Abduallah et al., FLAIRS-34 (2021) ✅ | 10.5281/zenodo.7516668 (v1.6) | ✅ repo | No | **Partially** – stored runs reproduced (stochastic); CLI bug; paper reports figures only |
| 8 | Kp-prediction (KpNet) | Kp 1–9 h ahead + UQ | Conv1D+BiLSTM+transformer+TFP | Abduallah et al., ISMIS 2022, LNCS 13515:325–335 ✅ | 10.5281/zenodo.7514451 (v1.1) | ✅ repo | No | **Successfully reproduced** at tool level (within MC noise); paper not accessible |
| 9 | CMETNet | CME transit time (h) | RF, SVR, XGB, GPR + CNN | Alobaid et al., Front. Astron. Space Sci. 9:1013345 (2022) ✅ | 10.5281/zenodo.7516684 (v.1.1) | trains in script | CNN optional | **Partially** – numerical ensemble only; COMB 14.2 h (2013–17) vs paper FW 10.32 h; test-year model selection |
| 10 | FibrilNet | Hα fibril tracing + uncertainty | SegNet-style Bayesian CNN (MC dropout) | Jiang et al., ApJS 256:20 (2021) ✅ | 10.5281/zenodo.7521568 (v1.6) | ✅ web.njit.edu (264 MB) | Training only | **Successfully reproduced** at tool level (IoU vs stored masks 0.901 = IoU between our own MC runs 0.900); paper Table 1 reference skeletons not shipped |
| 11 | SDNN | Stokes inversion (B, Doppler width, LOS velocity) | Stacked 1-D CNN | Jiang et al., ApJ 939:66 (2022) ✅ | 10.5281/zenodo.7521580 (v1.3) | ✅ web.njit.edu (149 MB) | No | **Successfully reproduced** at tool level (5/5 maps, r = 1.000000); paper's ME reference maps not shipped |
| 12 | SolarFlareNet | ≥M5.0/≥M/≥C flare in 24/48/72 h | Conv1D+LSTM+4 transformer blocks | Abduallah et al., Sci. Rep. 13:13665 (2023) ✅ | 10.5281/zenodo.10201356 (tag v1.3) | ✅ repo | Optional | **Partially** – stored CSVs reproduced; 8/9 models no skill vs paper TSS 0.71–0.84 |
| 13 | SYMHnet | SYM-H 1–2 h ahead, 1/5-min data | GNN + BiLSTM + MC dropout | Abduallah et al., Space Weather 22 e2023SW003824 (2024) ✅ | 10.5281/zenodo.10602518 (tag v1.0) | ✅ repo | Optional | **Partially** – v1.0 reproduces the paper's per-storm RMSE for storms 31/33/42 (~2 %), not 28/36/37/40; next-row target |
| 14 | SolarCNN | MDI → HMI-quality magnetograms | Attention-aided CNN | Xu et al., Sol. Phys. 299:36 (2024) ✅ | 10.5281/zenodo.15250172 (direct upload ❓) | ✅ Google Drive (README link) | No | **Partially** – shipped predictions exact; with training normalisation SSIM 0.812 → 0.892 (paper 0.833 → 0.904) on 3 of 76 samples |
| 15 | MIST | ≥M flare in 24 h | Patch time-series transformer | ❓ closest: Chaudhary et al., arXiv:2606.21896 (2026) | 10.5281/zenodo.16780646 (direct upload ❓) | ✅ repo | No | **Partially** – README TSS 0.906 exact; arXiv paper's numbers differ; out-of-sample TSS 0.197 |
| 16 | DeepHMI | HMI → GST/NIRIS resolution | Conditional diffusion model | Xu et al., A&A (2025) ✅ | 10.5281/zenodo.17059248 (direct upload ❓) | ✅ repo | Strongly | **Blocked (compute)** – ≈ 37 CPU-hours; shipped output re-scored |
| 17 | SINet | F10.7 & F30, 1–60 d | TimesNet-style | Wang Z. et al., JGR Space Physics (2026); arXiv:2604.10045 ✅ | 10.5281/zenodo.20274899 (direct upload ❓) | ✅ repo | No | **Successfully reproduced** (tool and paper: F10.7 RMSE 3.88 / 16.32 sfu) |
| 18 | LLMFlare | Natural-language flare explanations | MIST + SHAP + Llama-3.1-8B LoRA | ❓ none found | 10.5281/zenodo.20738748 (direct upload ❓) | LoRA adapter only | Yes | **Not attempted** – gated Llama, paid Gemini API, GPU (needs authorisation) |
| 19 | DeepGeoCME | Is a CME geoeffective? | ResNet-50 + EfficientNet-B0 + attention | Yan et al., arXiv:2605.24748 (2026) ✅ | 10.5281/zenodo.22955090 (direct upload, 1.0.0 ❓) | ✅ GitHub release | No | **Successfully reproduced** (tool and paper: TSS 0.757) |
| – | MagNet (SolarDB; not on the CCSC page) | MDI LOS + Hα → transverse field | 2 × ResNet-101/DANet heads | Jiang et al., Sol. Phys. 298:87 (2023) ✅ | — | ✅ nature.njit.edu (1.18 GB) | Recommended | **Blocked** – package lacks `layers/attention.py` (PAM, CAM) that the model file imports |

"direct upload ❓" means that the record's citation was resolved, but its files could not be downloaded (Zenodo 504), so whether they equal the GitHub HEAD that was run is unverified.

(`reports/CCSC_Reproduction_Report.md` is the authoritative status document.)

## 2. Per-tool records (16 fields)

Fields: 1 name · 2 research question · 3 paper title · 4 citation · 5 paper link/DOI · 6 source code · 7 dataset ·
8 algorithm · 9 Python & dependencies · 10 pretrained weights · 11 notebook · 12 GPU · 13 data size & download needs ·
14 reported results · 15 reproducibility & difficulties · 16 current status.

### 2.1 FlareML
1. **FlareML** (backend of the DeepSun MLaaS service, `nature.njit.edu/spacesoft/DeepSun/`).
2. Given 13 SHARP magnetic parameters of an active region (AR) at 00:00 on a flare day, classify the day's flare as B, C, M or X.
3. "DeepSun: machine-learning-as-a-service for solar flare prediction".
4. Paper: Abduallah Y., Wang J. T. L., Nie Y., Liu C., Wang H. 2021, *Research in Astronomy and Astrophysics* 21(7):160 ✅ (arXiv full text). **Citation button:** Abduallah, Y., Wang, J. T. L., & Wang, H. (2023). *Predicting Solar Flares with Machine Learning* (Version v1.5) [Computer software]. Zenodo. doi:10.5281/zenodo.7506997 ✅.
5. doi:10.1088/1674-4527/21/7/160 ✅; arXiv:2009.04238 ✅; the README badge points to an older record, 10.5281/zenodo.5634114.
6. github.com/ccsc-tools/FlareML (commit 44ffd7a) ✅; newer: github.com/deepsuncode/Machine-learning-as-a-service.
7. `data/original_data/flaringar_original_data.csv` (845 flares, 2010-05-03 → 2016-12-27, 472 ARs), a 435-row training sample, and a 40-row demo test set ✅.
8. Random forest, MLP, extreme learning machine (`sklearn_extensions`), and a majority vote of the three (ties go to RF) ✅.
9. Python 3.9, scikit-learn 0.24.2, sklearn-extensions 0.0.2, numpy 1.19.5. The pickles were written by scikit-learn 0.23.2 ✅.
10. Yes: `models/default_model_{rf,mlp,elm}.sav` ✅.
11. Yes: two notebooks, Binder-enabled ✅.
12. No.
13. < 1 MB data; 49 MB models.
14. Paper Table III (13 SHARP parameters): TSS B / C / M / X / average = ENS 0.745 / 0.380 / 0.551 / 0.362 / **0.507**; RF 0.708 / 0.378 / 0.537 / 0.330 / **0.488** ✅. The repo reports only the demo.
15. Demo inference reproduces the shipped log exactly (ENS acc 0.850, RF 0.825, ELM 0.725; MLP collapses to class C, acc 0.300, as in the authors' log).
    **Caveats:** (a) `check_pm_precision()` replaces the *displayed* RF per-class score by "ENS − U(0.01, 0.07)" whenever RF ≥ ENS.
    (b) `log_cv_report()` alters confusion-matrix cells when FP or FN = 0.
    (c) MLP/ELM inputs are standardised with a scaler fitted on the *test batch*.
    (d) **31 of the 40 demo test rows also appear in the training sample** (all 13 features within 0.6 %, same class), so the demo score is not out-of-sample.
16. **Partially reproduced.** Demo inference is exact (cited v1.5 differs from HEAD only in README). Extension: RF on the 845-flare table, random 10-fold macro-TSS 0.338 ± 0.007, AR-grouped 0.246 ± 0.022. This is below the paper's 0.488, but the paper's exact protocol is not in the release.

### 2.2 LSTM-flare-prediction
1. **LSTM-flare-prediction** ("FlarePredict").
2. Will an AR produce a ≥M5.0 / ≥M / ≥C flare within the next 24 h?
3. "Predicting Solar Flares Using a Long Short-term Memory Network".
4. Paper: Liu H., Liu C., Wang J. T. L., Wang H. 2019, *ApJ* 877:121 ✅. **Citation button:** Liu, H., Liu, C., Wang, J. T. L., & Wang, H. (2021). *Predicting Solar Flares Using a Long Short-term Memory Network* (Version v1.0.2) [Computer software]. Zenodo. doi:10.5281/zenodo.5739321 ✅.
5. doi:10.3847/1538-4357/ab1b3c ✅; arXiv:1905.07095 ✅; web.njit.edu/~wangj/LSTMpredict/.
6. github.com/ccsc-tools/LSTM-flare-prediction (12fc8b9) ✅ (TF2 port); original paper code in `Table5_dataset_and_source_code/LSTMpredict.py` (TF 1.12) ✅.
7. `Table2_dataset/{C,M,M5}`: train 84,577 rows (2010-05 → 2014-01), validation 26,473 rows (2014), test 44,689 rows (2015 → 2018-05), with **no shared HARP/NOAA ARs**.
   `Table5_dataset_and_source_code/C` keeps the incomplete samples. "Collection of Flares" lists 4,203 B, 6,768 C, 704 M and 49 X flares ✅.
8. One LSTM layer (10 units) → temporal attention → Dense(200) → Dense(500) → softmax. Inputs are 10-step series of 14 (≥C), 22 (≥M) or 20 (≥M5.0) of 40 features (25 SHARP + 15 flare-history) ✅.
9. TF/Keras 2.10, scikit-learn 1.2 (paper: TF 1.12, Keras 2.2.4) ✅.
10. Yes, but only for the small "sample_run" demo (one test AR per class) ✅.
11. Yes (`ccsc_FlarePredict.ipynb`; Binder) ✅.
12. No (one training run takes about 26–45 s on 2 CPU threads).
13. ≈ 1 GB repository; all data included.
14. Paper Table 3/4 (LSTM, single split, thresholds 0.50/0.60/0.50): TSS **0.877 / 0.790 / 0.607** (≥M5.0 / ≥M / ≥C). Table 5 (CV): 0.881 / 0.792 / **0.612**. The authors' `output.txt` equals the Table 5 ≥C column digit for digit ✅. Table 2 sample counts equal the shipped `Table2_dataset` exactly ✅.
15. See report. Key points:
    * The released `flarepredict_test.py` treats softmax column 0 as "positive". For ≥M5.0 the LabelEncoder order is ['N','X'], so **≥M5.0 predictions are inverted**: TSS −0.671. Using column 1 gives TSS 0.686 and 98.9 % agreement with the authors' stored output.
    * The paper's Table 5 picks the decision threshold that maximises average TSS **on the test folds**.
    * The previous-timestep missing-value check tests `row` instead of `prev_row`, so incomplete history rows are never masked.
16. **Partially reproduced.**
    * The cited v1.0.2 demo models reproduce the stored demo outputs (99.1–99.5 %; HEAD ships newer models).
    * Retraining on Table 2 (10 seeds) at the paper's thresholds gives TSS 0.812 ± 0.119 / 0.730 ± 0.031 / 0.580 ± 0.011, against the paper's 0.877 / 0.790 / 0.607. With the test-optimised threshold it gives 0.877 / 0.748 / 0.589.
    * The Table 5 protocol (10×10 CV, 100 trainings) under TF 2.10 gives 0.581 ± 0.011, against 0.612 ± 0.009.

### 2.3 RNN-CME-prediction
1. **RNN-CME-prediction** ("CMEPredict").
2. Given that an AR produces an M- or X-class flare, will that flare be associated with a CME within 12/24/36/48/60 h?
3. "Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks".
4. Paper: Liu H., Liu C., Wang J. T. L., Wang H. 2020, *ApJ* 890:12 ✅. **Citation button:** Liu, H., Liu, C., Wang, J. T. L., & Wang, H. (2022). *Predicting Coronal Mass Ejections Using SDO/HMI Vector Magnetic Data Products and Recurrent Neural Networks* (Version v1.3) [Computer software]. Zenodo. doi:10.5281/zenodo.7415731 ✅.
5. doi:10.3847/1538-4357/ab6850 ✅; arXiv:2002.10953 ✅.
6. github.com/ccsc-tools/RNN-CME-prediction (04be43f) ✅.
7. 18 SHARP parameters; labels from GOES flares (NCEI) plus flare–CME associations (DONKI). The demo test set covers **only 2 ARs (12497, 12529)**; `CME_data_samples/` holds the paper's full training and test sets (row counts = paper Table 1) ✅. The two copies store the 18 feature columns in **different orders** (see 15).
8. GRU or LSTM (10 units) + attention + dense layers; per-model fixed thresholds of 0.40–0.50 ✅.
9. TF/Keras 2.8–2.11 ✅.
10. Yes (10 × `.h5`) ✅.
11. Yes ✅.
12. No.
13. ≈ 290 MB.
14. Paper Table 4 (CV means): GRU TSS 0.452 / 0.588 / 0.602 / 0.588 / 0.600 (12–60 h), AUC 0.78–0.87; LSTM TSS 0.46–0.56 ✅. Stored demo outputs: TSS 0.82–1.00 on 2 ARs ✅.
15. Predicted labels are **identical** (10/10 settings) to the stored demo outputs.
    * On the paper's full test sets the released models give GRU TSS 0.31–0.49 (AUC 0.72–0.77) and LSTM 0.26–0.37 at the tool's thresholds. Missing padding rows explain at most 4.4 % of labels.
    * The columns of `CMEpredict/normalized_*.csv` are a permutation of those in `CME_data_samples/`. Only the latter labelling is physically consistent (e.g. TOTUSJH–USFLUX r = 0.92). The code's feature lists reproduce the paper's Table 2 ranking under the other labelling (`results/RNN-CME-prediction/column_label_check.json`).
    * The quality filters index non-existent columns and never trigger.
16. **Partially reproduced.** The tool level is exact (cited v1.3, run from the tag); the paper level is not reached.

### 2.4 SolarFlareNet
1. **SolarFlareNet** (operational on the Web; also runs at NASA CCMC per a 2026 arXiv comparison 🔎).
2. Will an AR produce a ≥M5.0 / ≥M / ≥C flare within 24 / 48 / 72 h?
3. "Operational prediction of solar flares using a transformer-based framework".
4. Paper: Abduallah Y., Wang J. T. L., Wang H., Xu Y. 2023, *Scientific Reports* 13:13665 ✅. **Citation button:** Abduallah, Y., Wang, J. T. L., Wang, H., & Xu, Y. (2023). *Operational Prediction of Solar Flares Using a Transformer-Based Framework* [Computer software]. Zenodo. doi:10.5281/zenodo.10201356 (GitHub tag v1.3) ✅.
5. doi:10.1038/s41598-023-40884-1 ✅ (open access); the README badge points to 10.5281/zenodo.10080716.
6. github.com/ccsc-tools/SolarFlareNet (1bf04b8) ✅.
7. SHARP time series; training files 2010–2022-09, testing files 2022-01 → 2022-09 (100 HARPs) ✅.
8. BatchNorm → Conv1D(32) → LSTM(400) → 4 × [multi-head attention + Conv1D + LSTM(400)] → Dense(200) → Dense(500) → softmax ✅.
9. TF 2.8 / Keras 2.6 (ran on TF 2.10) ✅.
10. Yes (9 checkpoints, 74 MB each) ✅.
11. Yes ✅.
12. Optional.
13. ≈ 1.9 GB.
14. Paper Table 1 (24 h, 10-fold CV): TSS **0.818 / 0.839 / 0.835** (≥M5.0 / ≥M / ≥C). Table 2: 48 h 0.736 / 0.728 / 0.719; 72 h 0.729 / 0.714 / 0.709 ✅. Shipped `result/` CSVs ✅.
15. Re-run outputs agree ≥ 99.98 % with the stored result CSVs for 8/9 settings. **Those 8 shipped models have AUC 0.40–0.53 (no skill) on the bundled test files.**
    The C/24 h weights were retrained after the stored results were produced (see the shipped 2023-11-05 log) and give TSS 0.79. However, **84 % of the bundled 24 h test rows also occur (same HARP and timestamp) in the training file**; on the non-overlapping rows TSS is 0.697.
    The loader selects features by position using another dataset's column order (off by one: TOTUSJH is never used). The 9 features actually fed are TOTPOT, ABSNJZH, USFLUX, AREA_ACR, MEANPOT, R_VALUE, MEANGAM, MEANJZH and MEANJZD. The paper's 9 are TOTUSJH, TOTUSJZ, USFLUX, MEANALP, R_VALUE, TOTPOT, SAVNCPP, AREA_ACR and ABSNJZH.
16. **Partially reproduced** (stored outputs reproduced; paper-level skill not reproducible with the shipped artefacts).

### 2.5 SEP-prediction
1. **SEP-prediction**.
2. Will an AR that produces an M/X flare (FC_S: one with a CME; F_S: regardless of CME) produce a solar energetic particle event within 12–72 h?
3. "Predicting Solar Energetic Particles Using SDO/HMI Vector Magnetic Data Products and a Bidirectional LSTM Network".
4. Paper: Abduallah Y., Jordanova V. K., Liu H., Li Q., Wang J. T. L., Wang H. 2022, *ApJS* 260:16 ✅. **Citation button:** Abduallah, Y., Jordanova, V. K., Liu, H., Li, Q., Wang, J. T. L., & Wang, H. (2022). *Predicting Solar Energetic Particles Using SDO/HMI Vector Magnetic Data Products and a Bidirectional LSTM Network* (Version v1.4) [Computer software]. Zenodo. doi:10.5281/zenodo.7516609 ✅.
5. doi:10.3847/1538-4365/ac5f56 ✅; arXiv:2203.14393 ✅.
6. github.com/ccsc-tools/SEP-prediction (119ea98) ✅.
7. SHARP parameters; training 2011–2016, test Apr–Sep 2017. The test set has **3 ARs, of which 2 (12665, 12673) produce all positives** ✅.
8. BiLSTM + attention + dense layers; isotonic-regression calibration ✅.
9. TF 2.10 ✅.
10. Yes (`default_models`, `models`) ✅.
11. Yes ✅.
12. No.
13. ≈ 1.4 GB.
14. Paper Tables 3–4 (CV means, biLSTM): TSS FC_S 0.737 / 0.515 / 0.567 / 0.592 / 0.590 / 0.703 (12–72 h); F_S 0.441 / 0.428 / 0.529 / 0.412 / 0.416 / 0.509 ✅. Stored demo metrics (`default_results`, 3 ARs): TSS 0.86–0.99 ✅.
15. **All 12 stored metric rows reproduced exactly.** Caveats: the effective sample is 2 SEP-producing ARs, and the calibrated Brier scores (BSC/BSSC) use an isotonic regression fitted on the test set itself.
16. **Successfully reproduced** at tool level. Paper level: the release ships neither the CV folds nor code, and its sample counts differ from the paper's Table 1 (FC_S 12 h: 4,043 vs 3,946), so the paper's numbers cannot be re-run.

### 2.6 CMETNet
1. **CMETNet**.
2. How many hours will a geoeffective CME take to travel from the Sun to Earth?
3. "Predicting CME arrival time through data integration and ensemble learning".
4. Paper: Alobaid K. A., Abduallah Y., Wang J. T. L., Wang H., Jiang H., Xu Y., Yurchyshyn V., Zhang H., Cavus H., Jing J. 2022, *Frontiers in Astronomy and Space Sciences* 9:1013345 ✅. **Citation button:** Alobaid, K. A., Abduallah, Y., Wang, J. T. L., & Wang, H. (2022). *Predicting CME Arrival Time through Data Integration and Ensemble Learning* (Version v.1.1) [Computer software]. Zenodo. doi:10.5281/zenodo.7516684 ✅.
5. doi:10.3389/fspas.2022.1013345 ✅ (open access).
6. github.com/ccsc-tools/CMETNet (cfc7e0f) ✅.
7. `ICME_list.csv`: 363 ICMEs, 1996–2021, with CME and solar-wind features; 351 LASCO C2 FITS images ✅.
8. Median ensemble of RF (800 trees), SVR (RBF), XGBoost and GPR, combined with a CNN on C2 images ✅.
9. TF 2.11, scikit-learn 1.2, xgboost 1.5 ✅.
10. No (the scripts train). `pretrained_models_results.py` only plots **hard-coded** numbers ✅.
11. Yes ✅.
12. CNN: optional.
13. ≈ 1 GB.
14. Paper: leave-one-year-out over 2013–2021 (9 runs). CMETNet MAE **9.75 h**, PPMCC 0.83; numerical-only FW 10.32 h ✅. The plotting script hard-codes these numbers.
15. The released training script **selects, among 10 random 200-event training subsamples, the model with the lowest MAE on the held-out test year**. The quantified optimism is in the report.
16. **Partially reproduced** (numerical ensemble only). Over 2013–2017, the paper's years that have ≥ 3 events in the release, COMB MAE is 14.2 h even with test-year selection (15.3 h with validation-year selection).

### 2.7 Dst-prediction (DSTT)
1. **Dst Transformer (DSTT)**.
2. Forecast the Dst geomagnetic index 1–6 h ahead from 7 solar-wind parameters, with aleatoric and epistemic uncertainty.
3. "Forecasting the Disturbance Storm Time Index with Bayesian Deep Learning" (arXiv title: "A Deep Learning Approach to Dst Index Prediction").
4. Paper: Abduallah Y., Wang J. T. L., Bose P., Zhang G., Gerges F., Wang H. 2022, *Proc. FLAIRS-35* ✅. **Citation button:** Abduallah, Y., Wang, J. T. L., Prianka Bose, Genwei Zhang, Gerges, F., & Wang, H. (2022). *Forecasting the Disturbance Storm Time Index with Bayesian Deep Learning* (Version v1.1) [Computer software]. Zenodo. doi:10.5281/zenodo.7529705 ✅.
5. doi:10.32473/flairs.v35i.130564 ✅; arXiv:2205.02447 ✅.
6. github.com/ccsc-tools/Dst-prediction (e5ae919) ✅.
7. OMNIWeb hourly solar wind + Dst; test window Oct–Nov 2021 (1,103 h) ✅.
8. Conv1D → LSTM(650) → multi-head attention → TFP variational dense layers ✅.
9. TF 2.10 + TFP 0.17 ✅.
10. Yes (1–6 h) ✅.
11. Yes ✅.
12. No.
13. ≈ 730 MB.
14. Paper Figure 3 (bar charts only; test 2021-10-01 → 11-15, 1,104 h): DSTT R² ≈ 0.75 → 0.59 and RMSE ≈ 5–6 nT for 1–6 h (read off the figure). Stored `default_results`: RMSE ≈ 3.3–3.9 nT, r ≈ 0.98 ✅.
15. **The released `DSTTModel.predict()` passes predictions through `DSTT_utils.t()`.** Whenever a prediction is not ≥ 90 % away from the true value, it is replaced by **0.9 × true label**; 88–90 % of the stored "predictions" equal 0.9 × label exactly.
    In addition, `DSTT_test.test()` loads the 6 h weights for every horizon. The loader also appends the Oct–Nov 2021 test rows to the training frame (an `isin` filter on partial strings never matches).
    The raw network (MC mean of 50 passes, correct weights) gives R² 0.06–0.38 (RMSE 13–16 nT), worse than persistence at every horizon (R² 0.93 → 0.53).
16. **Attempted but unsuccessful.** The raw model (R² 0.06–0.38) is far below the paper, and the released path is above it because of the label substitution. The paper's RMSE and R² are not mutually consistent with this test period (Dst std 16.5 nT).

### 2.8 Kp-prediction (KpNet)
1. **KpNet**.
2. Forecast the Kp index 1–9 h ahead with uncertainty.
3. "A Transformer-Based Framework for Geomagnetic Activity Prediction".
4. Paper: Abduallah Y., Wang J. T. L., Xu C., Wang H. 2022, *ISMIS 2022*, LNCS 13515, pp. 325–335 ✅ (Crossref). **Citation button:** Abduallah, Y., Wang, J. T. L., Chunhui Xu, & Wang, H. (2022). *A Transformer-Based Framework for Geomagnetic Activity Prediction* (Version v1.1) [Computer software]. Zenodo. doi:10.5281/zenodo.7514451 ✅.
5. doi:10.1007/978-3-031-16564-1_31 ✅ (full text not accessible).
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
16. **Successfully reproduced** at tool level. The paper (Springer LNCS) was not accessible.

### 2.9 TSInet
1. **TSInet**.
2. Reconstruct total solar irradiance (TSI) beyond the range of physics-based models.
3. "Reconstruction of Total Solar Irradiance by Deep Learning" (arXiv: "Deep Learning Based Reconstruction of Total Solar Irradiance").
4. Paper: Abduallah Y., Wang J. T. L., Shen Y., Alobaid K. A., Criscuoli S., Wang H. 2021, *Proc. FLAIRS-34* ✅. **Citation button:** Abduallah, Y., Wang, J. T. L., Yucong Shen, Alobaid, K. A., Criscuoli, S., & Wang, H. (2022). *Reconstruction of Total Solar Irradiance by Deep Learning* (Version v1.6) [Computer software]. Zenodo. doi:10.5281/zenodo.7516668 ✅.
5. doi:10.32473/flairs.v34i1.128356 ✅; arXiv:2107.11042 ✅ (results in Figs. 4–10 only).
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
4. Paper: Abduallah Y., Alobaid K. A., Wang J. T. L., Wang H., Jordanova V. K., Yurchyshyn V., Cavus H., Jing J. 2024, *Space Weather* 22(2), e2023SW003824 ✅. **Citation button:** Abduallah, Y., Alobail [sic], K. A., Wang, J. T. L., Wang, H., Jordanova, V. K., Yurchyshyn, V., Cavus, H., & Jing, J. (2024). *Prediction of the SYM-H Index Using a Bayesian Deep Learning Method with Uncertainty Quantification* [Computer software]. Zenodo. doi:10.5281/zenodo.10602518 (GitHub tag v1.0) ✅.
5. doi:10.1029/2023SW003824 ✅; arXiv:2402.17196 ✅.
6. github.com/ccsc-tools/SYMHnet (f4583d9) ✅.
7. 42 storms (1998–2018); storms 1–25 for training, 26–42 for testing ✅.
8. Graph neural network + BiLSTM, MC-dropout uncertainty ✅ (README).
9. TF 2.8 (ran on 2.10) ✅.
10. Yes ✅.
11. Yes ✅.
12. Optional.
13. ≈ 3.7 GB.
14. Paper Tables 8/9/12 (5-min, per test storm 26–42): e.g. storm 37 RMSE 10.320 / 10.923 (1 h / 2 h), R² 0.993; storm 31 7.923 / 9.737. FSS 0.343 / 0.553 vs GBM 0.074 / 0.087 for storm 37 ✅.
15. The 1 h and 2 h data folders are **byte-identical**, and SYM_H equals SYMH_ORIG (no shift). The test windowing (`lag=1`, `forecast_horizon=1`) targets the **next row**, i.e. the released test path predicts 1 or 5 minutes ahead, not 1–2 h. The output filename uses `start_hour`, so multi-horizon calls overwrite the 1 h file.
16. **Partially reproduced.** The cited v1.0 code (11 features; HEAD has 8) reproduces the paper's RMSE/R² for storms 31, 33 and 42 within about 2 %, gives about half the paper's RMSE for storms 28, 36 and 40, and a higher RMSE for 37. Under the next-row target, one-step persistence beats it on all 7 storms (`results/cited_versions/SYMHnet_v1.0/`).

### 2.11 SolarUnet
1. **SolarUnet**.
2. Identify and track signed magnetic flux elements in BBSO/GST vector magnetograms (an alternative to SWAMIS).
3. "Identifying and Tracking Solar Magnetic Flux Elements with Deep Learning".
4. Paper: Jiang H., Wang J., Liu C., Jing J., Liu H., Wang J. T. L., Wang H. 2020, *ApJS* 250:5 ✅. **Citation button:** Jiang, H., Wang, J., Liu, C., Jing, J., Liu, H., Wang, J. T. L., & Wang, H. (2021). *Identifying and Tracking Solar Magnetic Flux Elements with Deep Learning* (Version v.1.2) [Computer software]. Zenodo. doi:10.5281/zenodo.5643647 ✅.
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
16. **Successfully reproduced** at tool level (demo). The paper's Table 2 statistics use 147 test magnetograms; 3 are shipped.

### 2.12 FibrilNet
1. **FibrilNet**.
2. Trace chromospheric fibrils in Hα images with per-pixel uncertainty.
3. "Tracing Hα Fibrils through Bayesian Deep Learning".
4. Paper: Jiang H., Jing J., Wang J., Liu C., Li Q., Xu Y., Wang J. T. L., Wang H. 2021, *ApJS* 256:20 ✅. **Citation button:** Jiang, H., Jing, J., Wang, J., Liu, C., Li, Q., Xu, Y., Wang, J. T. L., & Wang, H. (2023). *Tracing Hα Fibrils through Bayesian Deep Learning* (Version v1.6) [Computer software]. Zenodo. doi:10.5281/zenodo.7521568 ✅.
5. doi:10.3847/1538-4365/ac14b7 ✅; arXiv:2107.07886 ✅.
6. github.com/ccsc-tools/FibrilNet (ba1764b) ✅.
7. 5 test images (AR 12665, 2017-07-13), 242 training pairs ✅.
8. SegNet-style encoder–decoder (max-pooling indices re-used for unpooling, skip concatenations); dropout active at test time (MC dropout) → mean mask + aleatoric/epistemic maps; polynomial fibril fitting ✅.
9. TF 1.11 / Keras 2.2.4 (ran on TF 2.10 with two name shims: `K.tf`, `np`) ✅.
10. Yes, from the authors' server: `utils.download_model()` fetches `web.njit.edu/~wangj/deepsuncode/FibrilNet/model.hdf5` (264 MB; SHA-256 in `results/downloaded_artefacts_sha256.txt`) ✅.
11. Yes ✅.
12. Training: yes (10 × 1,000 steps at 720×720).
13. 173 MB repository + model download.
14. Paper Table 1 (5 test images vs the Jing et al. 2011 tool): PS ≥ 95 %, FCFP ≈ 80 %, RI ≥ 91 % ✅. The repo stores masks and uncertainty maps ✅.
15. The shipped `test.py` calls `predict(model, image, T=2)`, a name defined nowhere (NameError). The notebook path `helper.predict` (T = 5) was run instead: 3 seeds, about 2.5 CPU-minutes each.
    * Ours vs stored masks: IoU 0.901 (0.885–0.911), pixel agreement 0.991.
    * Ours vs ours (different seeds): IoU 0.900 (0.886–0.911).
    * The stored masks behave like one more MC draw (`results/FibrilNet/compare_runs.json`).
    * The Jing et al. skeletons for the test images (paper Table 1) are not shipped.
16. **Successfully reproduced** at tool level (cited v1.6, run from the tag). Paper Table 1 is not re-runnable from the release.

### 2.13 SDNN
1. **SDNN**.
2. Infer LOS velocity and Doppler width (plus Bx, By, Bz) from GST/NIRIS Stokes profiles, replacing slow Milne–Eddington inversion.
3. "Inferring Line-of-sight Velocities and Doppler Widths from Stokes Profiles of GST/NIRIS Using Stacked Deep Neural Networks".
4. Paper: Jiang H., Li Q., Xu Y., Hsu W., Ahn K., Cao W., Wang J. T. L., Wang H. 2022, *ApJ* 939:66 ✅. **Citation button:** Jiang, H., Li, Q., Xu, Y., Hsu, W., Ahn, K., Cao, W., Wang, J. T. L., & Wang, H. (2023). *Stokes Inversion for GST/NIRIS Using Stacked Deep Neural Networks* (Version v1.3) [Computer software]. Zenodo. doi:10.5281/zenodo.7521580 ✅.
5. doi:10.3847/1538-4357/ac927e ✅; arXiv:2210.04122 ✅.
6. github.com/ccsc-tools/SDNN (dc44cec) ✅.
7. Input `cals_170713_162012.fts` (228 MB; 4 Stokes × 55 wavelengths × 720 × 720) is downloaded from web.njit.edu. It is the paper's test image D3 (AR 12665, 2017-07-13 16:20:12 UT). The repo ships only the output FITS files ✅.
8. Four per-Stokes 1-D CNN branches (Conv1D 64→256, max-pooling, Dense 240) → stacked → Conv1D 64→512 → Dense 2048 → 1024 → 5 outputs (|B|, inclination, azimuth, Doppler width, LOS velocity) per pixel ✅.
9. TF 1.11 / Keras 2.2.4 (ran on TF 2.10: the network was rebuilt with early-bound Lambda closures and the weights copied by layer name) ✅.
10. Yes, from the authors' server: `web.njit.edu/~wangj/deepsuncode/SDNN/pretrained_model.h5` (149 MB) ✅.
11. Yes ✅.
12. No.
13. 377 MB of downloads; inference takes about 16 CPU-minutes per 720×720 image (2 threads).
14. Paper Table 1, D3 (model trained on D1): LOS velocity MAE 0.289, PPMCC 0.912; Doppler width PPMCC 0.716, against Milne–Eddington ✅. The repo stores the output maps ✅.
15. Under Keras 2.10 the released `SDNN()` fails at `predict()`: its loop `Lambda`s close over `i` and `out`, which TF 1.x evaluated once at graph build. After an equivalent rebuild, **all 5 output maps match the stored ones** (max |Δ| ≤ 0.011 G for Bx/By/Bz, ≤ 3.5 × 10⁻⁶ for velocity/width; r = 1.000000). The ME maps needed for Table 1 are not shipped.
16. **Successfully reproduced** at tool level (cited v1.3, run from the tag). Paper Table 1 is not re-runnable from the release.

### 2.14 SolarCNN
1. **SolarCNN**.
2. Enhance (super-resolve) SOHO/MDI LOS magnetograms of ARs using SDO/HMI as ground truth, to give cycle-23 data HMI-like quality.
3. "Super-Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using SDO/HMI Data and an Attention-Aided Convolutional Neural Network".
4. Paper: Xu C., Wang J. T. L., Wang H., Jiang H., Li Q., Abduallah Y., Xu Y. 2024, *Solar Physics* 299:36 ✅. **Citation button:** Xu, C., Wang, J. T. L., Wang, H., Jiang, H., Li, Q., Abduallah, Y., & Xu, Y. (2025). *Super-Resolution of SOHO/MDI Magnetograms of Solar Active Regions Using SDO/HMI Data and an Attention-Aided Convolutional Neural Network*. Zenodo. doi:10.5281/zenodo.15250172 (direct upload; not downloadable here) ✅.
5. doi:10.1007/s11207-024-02283-1 ✅; arXiv:2403.18302 ✅.
6. github.com/ccsc-tools/SolarCNN (52d4fa7) ✅.
7. 3 MDI/HMI 256×256 pairs + 3 stored predictions; full set on Zenodo ✅.
8. Attention-aided CNN; mixed SSIM/L1 loss ✅.
9. TF 2.6 + tensorflow-addons ✅.
10. The repo copy lacks `variables.data-*`. The README's Google Drive link supplies `model_solarcnn.zip` (163 MB), whose `saved_model.pb` and `variables.index` are byte-identical to the repo's ✅.
11. Yes ✅.
12. No.
13. 7.6 MB repository.
14. Paper Table 3 (76 test samples): SolarCNN SSIM 0.9039, PCC 0.8842, PSNR 37.40 dB; 'None' 0.8334 / 0.8627 / 34.73 ✅.
15. With the Drive weights, `test.py` reproduces the 3 shipped predictions to 3 × 10⁻⁵ G. `test.py` omits the training normalisation (clip ±2000 G, ÷2000), so those predictions are not better than the input.
    * With the normalisation, on the 3 shipped pairs, in the training space: SSIM **0.892** vs 0.812 for the MDI input, PCC 0.954 vs 0.943, PSNR **36.77** vs 32.63 dB.
    * This is the same direction and size as the paper (`results/SolarCNN/eval_normalised.json`).
16. **Partially reproduced.** The tool level is exact; the paper's improvement is reproduced on 3 of its 76 test samples (the rest are on Zenodo).

### 2.15 DeepHMI
1. **DeepHMI**.
2. Super-resolve SDO/HMI transverse and LOS magnetograms to GST/NIRIS resolution.
3. "Improving the Spatial Resolution of SDO/HMI Transverse and Line-of-Sight Magnetograms Using GST/NIRIS Data with Machine Learning".
4. Paper: Xu C., Xu Y., Wang J. T. L., Li Q., Wang H. 2025, *Astronomy & Astrophysics* ✅ (full text 403). **Citation button:** Xu, C., Xu, Y., Wang, J. T. L., Li, Q., & Wang, H. (2025). *Improving the Spatial Resolution of SDO/HMI Transverse and Line-of-Sight Magnetograms Using GST/NIRIS Data with Machine Learning*. Zenodo. doi:10.5281/zenodo.17059248 (direct upload) ✅.
5. doi:10.1051/0004-6361/202453581 ✅.
6. github.com/ccsc-tools/DeepHMI (92d3ebb) ✅.
7. One paired HMI/NIRIS test FITS (AR 12371, 2015-06-22) ✅.
8. Conditional denoising diffusion model (1,000 steps, U-Net with attention) ✅.
9. PyTorch 1.12.1, einops, astropy<5 ✅.
10. Yes (`model_bz`, `model_bt`, 28 MB each) ✅.
11. Yes ✅.
12. Strongly recommended (1,000 denoising steps at 720×720).
13. 134 MB.
14. Paper numbers ❓ (A&A full text not accessible; no arXiv version found).
15. A CPU re-run is infeasible: about 67 s per U-Net pass, so ≈ 37 h for 2 components × 1,000 steps. The authors' shipped output was re-scored against the GST/NIRIS truth:
    * Bz: SSIM 0.810 vs 0.733 for the HMI input; RMSE 190 vs 220 G.
    * Bt: SSIM 0.659 vs 0.683; RMSE 206 vs 223 G.
    Only 1 test pair is available.
16. **Blocked by missing data or compute resources** (compute).

### 2.16 MIST
1. **MIST** (the model class is called `MTST`, a multi-patch time-series transformer).
2. Will an AR produce a ≥M flare within 24 h?
3. Software record: "A New Deep Learning Method for Solar Flare Prediction" ✅. Closest paper: "Prediction of Solar Flares Using Photospheric Magnetic Field Parameters with Deep Learning" (Chaudhary, Wang, Xu, Xu, Zhang; arXiv:2606.21896, 2026). It has the same design (16 SHARP features, T = 24, P = 2, D = 128, May 2010–May 2018, stratified random split), but its link to MIST is inferred ❓.
4. **Citation button:** Chaudhary, Y., Abdullah [sic], Y., & Wang, J. T. L. (2025). *A New Deep Learning Method for Solar Flare Prediction* [Computer software]. Zenodo. doi:10.5281/zenodo.16780646 (direct upload) ✅.
5. arXiv:2606.21896 ✅ (closest paper; its link to MIST is inferred).
6. github.com/ccsc-tools/MIST (7cc02f1) ✅.
7. `processed_solar_dataset.csv` = **exactly the LSTM-flare Table 2 ≥M training file** (2010-05 → 2014-01; 84,577 rows; 16 features). Windows of 24 rows are labelled by row p+24 and **split 80/20 at random**; the shipped StandardScaler was fitted on all 84,577 rows ✅ (all verified).
8. Patching (2 h patches) → linear embedding → 3-layer transformer encoder (d=128, 8 heads) → pooling → sigmoid ✅.
9. PyTorch 2.3 (ran on 1.12.1) ✅.
10. Yes ✅.
11. Yes (`README.ipynb`) ✅.
12. No.
13. 142 MB.
14. README: TSS 0.90 ✅. arXiv:2606.21896: test accuracy 95.18 %, recall 0.89, AUC 0.96 at threshold 0.5; 67,661 / 16,916 train/test samples; thresholds for feature selection tuned with Optuna "to maximize the true skill statistic on the test set" ✅.
15. TSS 0.9062 reproduced exactly. However, **95.6 % of test windows have a 23-of-24-hour-overlapping neighbour in the training set**. On the chronologically and AR-independent 2015–2018 Table 2 test file, the same model gives **TSS 0.197** (AUC 0.83), versus TSS ≈ 0.73 for the 2019 LSTM on the same data.
16. **Partially reproduced.** The README number is exact. The arXiv paper's numbers differ from the released model's on its released split (accuracy 99.38 %, recall 0.910, AUC 0.997; 67,642 / 16,911 windows). The out-of-sample evaluation reveals leakage.

### 2.17 SINet
1. **SINet** (Solar Index Network).
2. Predict daily F10.7 and F30 radio-flux indices 1–60 days ahead.
3. "Daily Predictions of F10.7 and F30 Solar Indices With Deep Learning".
4. Paper: Wang Z., Abduallah Y., Wang J. T. L., Wang H., Xu Y., Yurchyshyn V., Oria V., Alobaid K. A., Bai X. 2026, *JGR Space Physics* ✅. **Citation button:** Wang, Z., Abduallah, Y., & Wang, J. T. L. (2026). *Daily Predictions of F10.7 and F30 Solar Indices with Deep Learning*. Zenodo. doi:10.5281/zenodo.20274899 (direct upload) ✅.
5. doi:10.1029/2025JA034868 ✅ (403 here); arXiv:2604.10045 ✅.
6. github.com/ccsc-tools/SINet (8017d8d) ✅.
7. NOAA F10.7; Toyokawa/Nobeyama F30; test 2009–2021 ✅.
8. TimesNet-style blocks (FFT period discovery + 2-D Inception convolutions) ✅.
9. PyTorch ≥1.12 ✅.
10. Yes (8 models) ✅.
11. Yes, with stored outputs ✅.
12. No.
13. 150 MB.
14. Paper (arXiv): F10.7 RMSE 3.88 sfu (1 d) and 16.32 sfu (60 d); F30 9.99 sfu (60 d) ✅. Notebook: F10.7 RMSE 3.880 / 13.741 / 15.092 / 16.324 for 1 / 27 / 45 / 60 d ✅.
15. F10.7 is reproduced to 4 decimals, and F30 within 2 %. SINet beats persistence and 27-day recurrence at every horizon.
16. **Successfully reproduced**, at tool and paper level: F10.7 RMSE 3.8801 / 16.3241 sfu (1 / 60 d) vs the paper's 3.88 / 16.32 (arXiv:2604.10045); F30 60 d 10.1475 vs 9.99 (−1.6 %).

### 2.18 LLMFlare
1. **LLMFlare**.
2. Generate natural-language explanations, recommendations and counterfactuals for flare predictions.
3. Software record: "Explainable Solar Flare Prediction Using Large Language Models" ✅; no paper found.
4. **Citation button:** Chaudhary, Y., Abdullah [sic], Y., & Wang, J. T. L. (2026). *Explainable Solar Flare Prediction Using Large Language Models*. Zenodo. doi:10.5281/zenodo.20738748 (direct upload) ✅.
5. ❓ (no paper found)
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
4. Paper: Yan Z., Wang J. T. L., Wang H., Lee H., Jing J., Xu Y., Xu C., Yurchyshyn V. 2026, arXiv:2605.24748 ✅. **Citation button:** Yan, Z., Abduallah, Y., & Wang, J. T. L. (2026). *Deep Learning-Enabled Prediction of Geoeffective CMEs Using SOHO and SDO Observations* (Version 1.0.0). Zenodo. doi:10.5281/zenodo.22955090 (direct upload) ✅.
5. arXiv:2605.24748 ✅.
6. github.com/ccsc-tools/DeepGeoCME (6daa000), upstream github.com/xysgszx/DeepGeoCME ✅.
7. Test tensors for 33 events (GitHub release, 144 MB) ✅.
8. Multimodal fusion: ResNet-50 and EfficientNet-B0 branches + attention; decision threshold 0.6 ✅.
9. PyTorch ≥2.6, timm (ran on 1.12.1 with a `torch.load` shim) ✅.
10. Yes (GitHub release, 199 MB; SHA-256 verified) ✅.
11. Yes (Colab) ✅.
12. No.
13. ≈ 350 MB downloads.
14. Paper: fusion model TSS **0.757** on the 80:20 split (Fig. 7 confusion matrix); 5-fold CV TSS 0.703 ± 0.030 ✅. Shipped `confusion_matrix.png`: TN 14, FP 2, FN 2, TP 15 ✅.
15. The assets downloaded from the GitHub release, and their SHA-256 hashes match the manifest. Our run gives an identical confusion matrix (TSS 0.757, recall 0.882, precision 0.882, Brier score 0.080). The training split (`train.csv`) is not shipped, so the independence of the 33 test events could not be checked.
16. **Successfully reproduced**, at tool and paper level (headline TSS 0.757). The 5-fold CV needs training images that are not shipped.

### Related SolarDB tool: MagNet
* Paper: Jiang H., Li Q., Liu N., Hu Z., Abduallah Y., Jing J., Xu Y., Wang J. T. L., Wang H. 2023, *Solar Physics* 298:87, doi:10.1007/s11207-023-02180-z; arXiv:2211.02278 ✅.
  It generates SOHO/MDI transverse fields (Bx, By) from MDI LOS + BBSO Hα, trained with SDO/HMI vector magnetograms.
* Yesmanchyk A., Xu Y., Wang J. T. L., Jiang H., Xu C., Wang H. 2026, "Out-of-Sample Validation of MagNet", arXiv:2601.15926 ✅. It validates on data not used for training.
* Not on the CCSC page. It is distributed at `nature.njit.edu/solardb/magnet`, which is titled "Magnet Un-published code" ✅. The page offers `MagNet.zip` (1.08 GB) or the folders:
  * `model/model.h5` (1.18 GB, Keras 2.6, weights only, 809 layers);
  * `Validate_MDI/`, `Validate_HMI/` (12 cases each; ha, los, bx, by, cor);
  * `Predicted_BxBy/` (the authors' outputs);
  * `magnet.py`, `MagNet_model.py`, `utils.py`.
* Requirements (README): TensorFlow-GPU 2.6.0, numpy 1.19.5, astropy 5.0.
* **Status: Blocked (incomplete package).** `MagNet_model.py` imports `from layers.attention import PAM, CAM`, and no `layers/` module exists in the zip (central directory read by HTTP range requests) or the folder listing. The weights hold only a scalar `gamma` per PAM/CAM layer (`results/MagNet/package_check.json`). Running it would need a third-party attention module, which would no longer be the released code.
