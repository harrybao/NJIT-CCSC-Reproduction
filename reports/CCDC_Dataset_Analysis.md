# CCDC and AI/ML-Ready Solar Datasets: Data-Science Analysis

*Prepared by Xiangsheng (Harry) Bao, 13 October 2026. All quantitative statements are backed by scripts and
outputs in this repository (`scripts/datasets/`, `results/datasets/`, and the per-tool `results/` folders).*

## 1. Executive summary

* **Access.** The CCDC page (`https://nature.njit.edu/solardb/ccdc`) could **not** be opened from the cloud environment
  used for this work: the host is blocked by the environment's egress policy and fails DNS in the fetch tool. No search engine indexes the page.
  The list of CCDC datasets therefore remains **unverified**. This report does not guess at it.
* **What I could analyse rigorously:** the ML-ready datasets distributed with the CCSC tools (more than 12 tables
  and image sets, inspected directly), plus published descriptions of SolarDB datasets (FlareDB, the global Hα network, the
  BBSO archive, MagNet data) and of the partner dataset SolARED.
* **Main data-science findings** (details in §4–§5):
  1. **Provenance is implicit.** MIST's "processed_solar_dataset.csv" is, value for value, the 2010–2013 ≥M training file of the 2019 LSTM study. LLMFlare's test set is byte-identical to MIST's. Neither relationship is documented.
  2. **Split design is the dominant source of optimism.** The LSTM Table 2 split (chronological, AR-disjoint) is exemplary. Datasets split at random over overlapping windows give inflated skill: MIST TSS 0.906 falls to 0.197 on independent years. Bundled "test" files sometimes overlap training files: in SolarFlareNet, 84 % of the 24 h test rows also appear in training.
  3. **Missing values are encoded as 0** in the SHARP-based tables, so they cannot be distinguished from legitimate zeros (e.g., flare-history counts). Sentinel fill values (9999.99 etc.) remain in some solar-wind tables.
  4. **Formats are inconsistent.** Timestamps such as `2010-1-1-0`, `1/1/2011 1:34` and `2014-12-28T01:22:09.20Z` are mixed. Some labels are stored as text (`Positive`, `P`, `p`, `padding`, flare classes). Features are selected by column position in several loaders, which already caused one silent column shift (SolarFlareNet).
  5. **Persistence risk.** Several models and inputs are hosted only on `web.njit.edu` (FibrilNet, SDNN) or `nature.njit.edu` (MagNet), with no checksum. DeepGeoCME (GitHub release plus a SHA-256 manifest) and the Zenodo DOIs are the best practice in the collection.
* **Funding context.** NSF's *Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets)*, **NSF 26-512**, is open with a full-proposal deadline of **4 November 2026** (per search results; verify on nsf.gov).
  NASA Heliophysics ran **ROSES B.16 "Heliophysics AI/ML-Ready Data" (HARD)** in 2022–2023. No ROSES-24 awards were made, and the topic now falls under **B.5 RADIANT** (ROSES-25).
  These are public programmes, **not** grants held by Prof. Wang's group.

## 2. What is known about CCDC and SolarDB

| Item | Status | Source |
|---|---|---|
| CCDC page title / contents | ❓ unverified (page blocked, not indexed) | — |
| SolarDB ("A Cyberinfrastructure for Advancing Space Weather Research") maintained by NJIT's Institute for Space Weather Sciences, partially NSF-supported | 🔎 search snippet of `nature.njit.edu/solardb/` | search |
| SolarDB's three ML-ready datasets: (i) a flare database (M5.0+ flares of cycle 24, SDO images from 12 h before to 6 h after each flare), (ii) a global Hα network (8 observatories), (iii) a BBSO data archive | 🔎 | Nita et al. 2022, EarthCube white paper, arXiv:2203.09544 |
| **FlareDB**: 151 significant flares, May 2010–May 2025; AIA AR patches in HPC and CEA projections, aligned with HMI magnetogram patches; quick-look movies on Zenodo; web download | 🔎 | Liu N., Abduallah Y., Kapure T., Li Q., Wang H., Wang J. T. L., *Scientific Data* (2026), doi:10.1038/s41597-026-06607-7 |
| **MagNet data**: directory listings `/solardb/magnetdata/MagNet/Validate_HMI/`, `/Validate_MDI/` | 🔎 search snippet; blocked | Jiang et al. 2023, Sol. Phys. 298:87 |
| **SolARED** (partner): acoustic power, unsigned flux and continuum-intensity time series for 50 emerging ARs (2010–2023), 81 tiles, FITS download via `sun.njit.edu/sarportal` | 🔎 | Kasapis et al. 2026, arXiv:2601.13145 |
| Interactive Multi-Instrument Database of Solar Flares (`solarflare.njit.edu`) | 🔎 exists | search |

**Interpretation (hypothesis, to be confirmed on the live page):** by analogy with CCSC ("Community Coordinated
*Software* Center"), CCDC is probably a "Community Coordinated *Data* Center" collecting the AI-ready datasets behind the CCSC tools
(FlareDB, MagNet data, SolarDB archives). This is an inference, not a verified fact.

## 3. Inventory of ML-ready datasets actually inspected (bundled with CCSC tools)

Source of all numbers: `results/datasets/bundled_dataset_profile.json` and the per-tool audits.

| Dataset (file) | Origin of raw data | Format | Size | Coverage | Unit of sample | Labels | Split as shipped |
|---|---|---|---|---|---|---|---|
| LSTM Table 2 (≥C/≥M/≥M5.0) | SDO/HMI SHARP (JSOC) + GOES flares (NOAA NCEI) | CSV, 40 features + metadata | train 84,577 / val 26,473 / test 44,689 rows | 2010-05 → 2018-05 | AR × hour | flare class in next 24 h | **chronological, AR-disjoint** (0 shared HARPs) |
| LSTM Table 5 (≥C, incl. incomplete samples) | same | CSV (normalised) | 127,709 rows (train) | 2010-05 → 2014-01 | AR × hour | same | as Table 2 |
| SolarFlareNet | SHARP + GOES | CSV, 25 SHARP + metadata | train ≈ 180–222k, test ≈ 63–78k rows per class/window | 2010-05 → 2022-09 | HARP × 12 min/1 h | N / P / p ("p" skipped) | test (2022) **contained in training** for 22–84 % of rows |
| SEP | SHARP + GOES + DONKI (flare–CME–SEP links) | CSV | train 7,002 / test 801 rows (24 h, FC_S) | 2011 → 2017 | HARP × time | SEP yes/no | chronological; **test = 3 ARs (2 positive)** |
| RNN-CME | SHARP + GOES + DONKI | CSV (normalised) | 75,063 rows (24 h train) | 2010 → 2019 | HARP × 12 min | N / P / padding | test = **2 ARs** |
| FlareML | SHARP + GOES | CSV, 13 features | 845 flares | 2010-05 → 2016-12 | flare day | B/C/M/X | demo test (40) overlaps training sample (31/40) |
| MIST | = LSTM Table 2 ≥M train (verified) | CSV + NumPy windows | 84,577 rows → 84,553 windows | 2010-05 → 2014-01 | 24 h window | ≥M in next 24 h | **random 80/20 over overlapping windows** |
| CMETNet | ICME list with CME (LASCO-derived) and solar-wind features — exact catalogue sources per the paper, not verified here; LASCO C2 FITS | CSV + 351 FITS | 363 ICMEs, 672 MB images | 1996 → 2021 | CME event | transit time (h) | leave-one-year-out; best of 10 rounds chosen **on test year** |
| Dst / Kp | NASA OMNIWeb hourly solar wind + indices | CSV | ≈ 96–104k rows | 2010 → 2022 | hour | Dst (nT) / Kp | Kp: clean temporal; Dst: **test window also inside training frame** |
| SYMHnet | OMNI 1-min/5-min + SYM-H | CSV | 315k (1-min) / 72k (5-min) train rows per storm folder | 1998 → 2018, 42 storms | minute | SYM-H | by storm (1–25 train, 26–42 test); 1 h and 2 h folders byte-identical |
| TSInet | SORCE, TCTE (LASP), NRLTSI2, SATIRE-S/M | CSV | 6,017 (SORCE) … 50,038 rows | 2003 → 2020 (+ models) | day | TSI (W m⁻²) | train SORCE / test other series |
| SINet | NOAA F10.7, Toyokawa/Nobeyama F30 | CSV + preprocessing pickle | — | test 2009 → 2021 | day | index value | train data not shipped |
| SolarUnet / FibrilNet / SDNN | BBSO/GST (NIRIS, Hα), SWAMIS labels, ME inversion | PNG/FITS | 3 / 5 test images; 242 train pairs | 2017–2018 | image | masks / maps | demo only |
| SolarCNN / DeepHMI / DeepGeoCME | SOHO/MDI + SDO/HMI; HMI + GST/NIRIS; SOHO + SDO | FITS / PyTorch tensors | 3 pairs / 1 pair / 33 events | — | image / event | HMI / NIRIS / geoeffective | demo only (full sets on Zenodo) |

## 4. Answers to the questions in the brief

**Where do the data come from?** Mostly NASA/NOAA archives:
* SDO/HMI SHARP parameters from JSOC (Stanford), SDO/AIA images;
* GOES X-ray flare catalogues (NOAA NCEI); flare–CME–SEP associations from NASA DONKI;
* solar wind and geomagnetic indices from NASA OMNIWeb/SPDF; LASCO catalogues;
* TSI from LASP (SORCE, TCTE) and model reconstructions (NRLTSI2, SATIRE);
* ground-based BBSO/GST data from NJIT's own observatory, and the global Hα network.

**Formats.** CSV for tabular time series (one row per AR-time or time step); FITS for images and magnetograms; PNG for
demo masks; NumPy, Pickle and PyTorch tensors for derived windows and models. Model weights are HDF5, TF checkpoints or SavedModel, and `.pth`.

**Size.** From kilobytes (845-flare table) to several GB (SYMHnet storm folders, ≈ 3.7 GB; SolarFlareNet, ≈ 1.9 GB).
The full image datasets (SolarCNN, DeepHMI, FlareDB) are on Zenodo or SolarDB.

**Collection, cleaning, integration.** Typical pipeline, reconstructed from the code:
1. Download SHARP time series per HARP.
2. Attach NOAA AR numbers (`harp_to_noaa.csv`).
3. Join GOES flare events by AR and time to create look-ahead labels (e.g. "≥M flare within 24 h of t").
4. Derive flare-history features (counts and decayed sums of past flares).
5. Z-normalise.
6. Drop samples whose physical features contain 0 (the convention for "missing").
7. Build fixed-length sequences per AR, zero-padding the history.

For geomagnetic indices: merge OMNI solar-wind parameters with the index at a time offset, then remove sentinel fill values.

**Missing values and quality issues** (`results/datasets/bundled_dataset_profile.json`):
* **0 is the missing-value code** for SHARP features. In LSTM Table 2, 27 % of physical-feature cells are exactly 0, many of them legitimate zeros in flare-history counts. The `has_zero_record` filters therefore conflate "missing" and "zero", and one of them checks the wrong row (§ LSTM in the main report).
* **Sentinel values** remain in some solar-wind tables: 0.13 % of cells in the Dst training file equal 9999.99-type fill values.
* **Non-standard timestamps:**
  * `YYYY-M-D-H` without zero padding is not lexicographically sortable; NRLTSI2 dates failed to parse in TSInet and fell back to the current date.
  * Time-only uniqueness is not guaranteed: TSInet's NRLTSI2 output repeats the same date.
* **Label encodings vary:** `Positive/Negative`, `P/N/p`, `padding`, raw GOES class strings. Alphabetical label encoding silently flips the positive class for `['N','X']`, which inverted the LSTM ≥M5.0 demo.
* **Positional column selection:** SolarFlareNet's loader assumes another table's column order, so 9 unintended features are used.

**Metadata.** CSV headers are informative (SHARP keywords), and READMEs describe columns for the LSTM and RNN-CME datasets.
However, there are no machine-readable data dictionaries (units, missing codes, valid ranges), no declared split manifest, and no provenance
links between derived datasets (MIST ← LSTM Table 2; LLMFlare ← MIST). Several READMEs give software versions but not data versions or SHARP release keywords.

**How do the data support ML?** They are ready-made supervised learning tables (labels attached, normalised, windowed) with
pre-trained models and notebooks, which is excellent for onboarding. What is missing for *reliable* ML is the
evaluation scaffolding: recommended splits, baselines, and documented class balance and effective sample size.

**Leakage and train/test issues** (all verified with scripts):

| Dataset | Issue | Effect measured |
|---|---|---|
| MIST | random split over 24 h windows overlapping by 23 h; scaler fit on all rows | TSS 0.906 → **0.197** on 2015–2018 (AUC 0.997 → 0.833) |
| SolarFlareNet | 2022 test rows also in training files (22–84 %) | 8/9 shipped models have no skill on the test files; C/24 h TSS 0.79 overall vs 0.697 on non-duplicated rows |
| FlareML demo | 31/40 test rows in training sample | demo TSS not out-of-sample; AR-grouped CV lowers macro-TSS from 0.338 to 0.246 |
| LSTM Table 5 | threshold chosen on test folds | Table 2 retraining: ≥M5.0 TSS 0.734 (fixed) / 0.825 (validation-chosen) / 0.877 (test-chosen) |
| Dst | test window inside training frame; labels used in `predict()` | raw model R² 0.06–0.38 vs reported ≈ 0.94 |
| CMETNet | best of 10 rounds chosen on hold-out year | test-selected MAE 0.4–2.1 h lower than the average round |
| SEP / RNN-CME | test sets of 3 and 2 ARs | very wide uncertainty on TSS ≈ 0.86–1.00 |

## 5. AI-readiness scorecard of the bundled datasets

Criteria are adapted from the ESIP "AI-Ready Data" checklist and from Poduval et al. (2023, *Front. Astron. Space Sci.* 10,
doi:10.3389/fspas.2023.1203598). ● = good, ◐ = partial, ○ = missing.

| Criterion | LSTM T2 | SolarFlareNet | MIST | SEP | Dst/Kp | SYMHnet | CMETNet | Image sets |
|---|---|---|---|---|---|---|---|---|
| Persistent identifier (DOI) | ● Zenodo | ● | ◐ (Zenodo per other README) | ● | ● | ● | ● | ◐ |
| Data dictionary (units, missing codes) | ◐ README | ○ | ○ | ○ | ○ | ◐ ReadMe | ○ | ○ |
| Provenance (raw source, versions, code) | ◐ | ◐ | ○ (undocumented reuse) | ◐ | ◐ | ◐ | ◐ | ◐ |
| Explicit split manifest, grouped by time/AR | ● | ○ | ○ | ◐ | ◐ | ● (by storm) | ◐ | ○ |
| Leakage-free as shipped | ● | ○ | ○ | ● | Kp ● / Dst ○ | ◐ (horizon) | ◐ | n/a |
| Baselines provided (persistence/climatology) | ○ | ○ | ○ | ○ | ○ | ◐ (GBM) | ◐ | ○ |
| Integrity (checksums) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | DeepGeoCME ● |
| Hosting redundancy | ● GitHub | ● | ● | ● | ● | ● | ● | FibrilNet/SDNN/MagNet ○ |

## 6. NSF/NASA AI/ML-ready data programmes (public information)

* **NSF 26-512 – Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets).**
  * Supports making *existing* scientific datasets usable for AI: AI-based feature extraction, metadata generation, dataset integration, robust automated-analysis pipelines, augmentation and harmonisation, and assessment of dataset readiness.
  * Encourages the use of national platforms (NSF data platforms, NAIRR).
  * Full proposals are due **4 November 2026**. Award tiers and amounts differ between third-party summaries, so check the official solicitation.
  * Sources: nsf.gov solicitation page (seen via search; not opened here), ASU and grant-aggregator summaries.
* **NASA ROSES-2022 B.16 Heliophysics AI/ML-Ready Data (HARD), NNH22ZDA001N-HARD.**
  * Aimed at new tools and methods that produce AI/ML-ready datasets from existing heliophysics mission data, including cleaning, missing-value handling and outlier removal.
  * The ROSES-23 Step-2 deadline moved to 2 May 2024. For ROSES-24 NASA announced no new awards and returned proposals without review.
  * For ROSES-25 the scope moved to **B.5 RADIANT**.
  * NJIT's research office circulated the 2022 call (research.njit.edu).
* **Community guidance:**
  * ESIP Data Readiness Cluster, "Checklist to Examine AI-readiness for Open Environmental Datasets".
  * Poduval et al. 2023 (AI-ready data in space science and solar physics: problems, mitigation and action plan). It stresses documentation of processing steps, uncertainties, biases and instrument degradation, plus standardised metadata.
  * FAIR principles.
* **Caveat:** these are funding opportunities and community recommendations. Nothing here implies that Prof. Wang's group holds or has applied for any of them.

## 7. Research questions this suggests (Data Science angle)

1. **Split manifests as a first-class AI-ready artefact.** Can every SolarDB/CCDC dataset ship a time- and AR-grouped split, together with a leakage test suite that runs automatically? MIST, SolarFlareNet and FlareML show that the measured gain is often larger than the model differences being claimed.
2. **Provenance graphs across derived datasets.** Automatically detect value-level reuse between datasets, as was done manually here (MIST = LSTM Table 2). Record it as machine-readable lineage (PROV-O, RO-Crate).
3. **Missing-value semantics for SHARP-derived tables.** Replace "0 = missing" with explicit masks and quality flags, and measure the downstream effect on flare-prediction skill.
4. **External, cross-instrument validation** (the question raised in the applicant's first e-mail). Stratify MagNet/SolarCNN/DeepHMI errors by instrument (MDI, HMI, IVM, GST), observing conditions, field strength and coverage. Use FlareDB's aligned HMI/AIA patches as a testbed.
5. **Baselines and effective sample size.** Standardise persistence/climatology references and event-level bootstrap confidence intervals, so that results such as "TSS 0.98 on 2 ARs" are reported with honest uncertainty.
6. **Calibration of Bayesian UQ.** Do the Dst/Kp/SYMHnet uncertainty bands achieve their nominal coverage? The current code scales them with ad-hoc constants.

## 8. References

* Nita G. et al. 2022, *Revisiting the Solar Research Cyberinfrastructure Needs: A White Paper of Findings and Recommendations*, arXiv:2203.09544.
* Liu N., Abduallah Y., Kapure T., Li Q., Wang H., Wang J. T. L. 2026, *FlareDB: A Database of Significant Flares in Solar Cycles 24 and 25 with SDO/HMI and SDO/AIA Observations*, Scientific Data, doi:10.1038/s41597-026-06607-7.
* Kasapis S. et al. 2026, *SolARED: Solar Active Region Emergence Dataset for Machine Learning Aided Predictions*, arXiv:2601.13145.
* Jiang H. et al. 2023, Sol. Phys. 298:87, doi:10.1007/s11207-023-02180-z (MagNet).
* Yesmanchyk et al. 2026, *Out-of-Sample Validation of MagNet*, arXiv:2601.15926.
* Poduval B. et al. 2023, *AI-ready data in space science and solar physics: problems, mitigation and action plan*, Front. Astron. Space Sci. 10, doi:10.3389/fspas.2023.1203598.
* ESIP Data Readiness Cluster, *Checklist to Examine AI-readiness for Open Environmental Datasets*, esipfed.org.
* NSF 26-512, *Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets)*, nsf.gov.
* NASA ROSES-2022 B.16, *Heliophysics Artificial Intelligence/Machine Learning-Ready Data* (NNH22ZDA001N-HARD), nspires.nasaprs.com; ROSES-2025 B.5 RADIANT.
* Bloomfield D. S. et al. 2012, ApJL 747:L41 (why TSS for flare forecasting).
