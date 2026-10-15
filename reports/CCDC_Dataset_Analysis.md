# CCDC and AI/ML-Ready Solar Datasets: A Data-Science Analysis

Xiangsheng (Harry) Bao, University of Sydney, October 2026

## 1. Summary

I opened the CCDC page (https://nature.njit.edu/solardb/ccdc) on 2026-10-13 and saved a snapshot in papers/citations/. The page is titled "CCDC: Community Coordinated Data Center" and states: "These AI-ready datasets are now available on GitHub and Zenodo for interested researchers to download and use." At present it lists one dataset, FlareDB. The dataset card links to github.com/Reasopprime/njit-flaredb, and its Citation button points to the Zenodo quick-look movies (doi:10.5281/zenodo.16790538, version 2025-05-31). The dataset paper is Liu et al. 2026 in *Scientific Data*, which is open access and which I read in full. Section 2.1 profiles FlareDB.

I also analysed the ML-ready datasets distributed with the 19 CCSC tools (more than 12 tables and image sets, which I inspected directly), the MagNet validation package on SolarDB, and the published descriptions of other SolarDB datasets and of the partner dataset SolARED. My main findings, detailed in Sections 4 and 5, are:

1. Derived datasets do not say where they come from. MIST's `processed_solar_dataset.csv` is, value for value, the 2010–2013 ≥M training file of the 2019 LSTM study, and LLMFlare's test set is byte-identical to MIST's. Neither relationship is documented.
2. The way the data are split is the largest source of over-optimistic results. The chronological, AR-disjoint split of LSTM Table 2 is a good model to follow. Datasets split at random over overlapping windows give inflated skill, and MIST's TSS of 0.906 falls to 0.197 on independent years. Bundled "test" files sometimes overlap the training files. In SolarFlareNet, 84 % of the 24 h test rows also appear in training.
3. Missing values are encoded as 0 in the SHARP-based tables, so they cannot be told apart from genuine zeros such as flare-history counts. Sentinel fill values (9999.99 and similar) remain in some solar-wind tables.
4. Formats are inconsistent. Timestamps appear in mixed formats, such as "2010-1-1-0", "1/1/2011 1:34" and "2014-12-28T01:22:09.20Z", and some labels are stored as text ("Positive", "P", "p", "padding", flare classes). Several loaders select features by column position, which has already caused one silent column shift, in SolarFlareNet.
5. Some models and inputs may not stay available. Several are hosted only on web.njit.edu (FibrilNet, SDNN), Google Drive (SolarCNN) or nature.njit.edu (MagNet, whose package is missing a code module), with no checksum. DeepGeoCME (a GitHub release with a SHA-256 manifest) and the Zenodo DOIs are the best practice in the collection.
6. RNN-CME ships the same samples twice, with the 18 SHARP columns in different orders. Only one ordering is physically plausible, and the released models were trained on the other. A data dictionary and a column-sanity test would have caught this.
7. FlareDB, the CCDC dataset, is a database of events. It holds 151 ≥M5.0 flares within 50° of disk centre from May 2010 to May 2025, in 82 ARs, with HMI/AIA patches in HPC and CEA projections and 5,285 quick-look movies on Zenodo. It has no non-flare examples, so it suits case studies, flare-evolution modelling and self-supervised learning, but on its own it cannot be used for flare/no-flare classification. The event list on GitHub is an earlier subset (103 flares, 57 ARs, up to 2024-05) with no header, mixed time formats and 45 rows missing two columns. The Zenodo rights metadata lists both CC-BY-4.0 and "All rights reserved".

On funding, NSF's programme *Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets)*, NSF 26-512, is open. Full proposals are due on 4 November 2026 and then on the first Wednesday of November each year (I checked this on nsf.gov). NASA Heliophysics ran ROSES B.16 "Heliophysics AI/ML-Ready Data" (HARD) in 2022–2023. No ROSES-24 awards were made, and the topic now falls under B.5 RADIANT in ROSES-25. These are public programmes, not grants held by Prof. Wang's group.

## 2. What is known about CCDC and SolarDB

| Item | How I checked | Source |
|---|---|---|
| CCDC page, titled "CCDC: Community Coordinated Data Center", with one dataset card (FlareDB) linking to GitHub and to a Zenodo citation | Opened 2026-10-13, snapshot saved | the page |
| SolarDB ("A Cyberinfrastructure for Advancing Space Weather Research"), maintained by NJIT's Institute for Space Weather Sciences with partial NSF support | Search snippet of nature.njit.edu/solardb/ only | web search |
| SolarDB's three ML-ready datasets: (i) a flare database (M5.0+ flares of cycle 24, SDO images from 12 h before to 6 h after each flare), (ii) a global Hα network (8 observatories), (iii) a BBSO data archive | Published description only | Nita et al. 2022, EarthCube white paper, arXiv:2203.09544 |
| FlareDB (the CCDC dataset): 151 significant (≥M5.0) flares from 82 ARs, May 2010 to May 2025; AIA AR patches in HPC and CEA projections, aligned with HMI magnetogram patches; 5,285 quick-look movies on Zenodo | Paper read; GitHub list profiled (Section 2.1) | Liu N., Abduallah Y., Kapure T., Li Q., Wang H., Wang J. T. L., *Scientific Data* (2026), doi:10.1038/s41597-026-06607-7; Zenodo doi:10.5281/zenodo.16790538 |
| MagNet data (/solardb/magnetdata/MagNet/): 12 MDI and 12 HMI validation cases (Hα, LOS, Bx, By, cor), the authors' predicted Bx/By and a 1.18 GB model, labelled "Un-published code" | Listed and checked; `layers/attention.py` is missing (results/MagNet/) | Jiang et al. 2023, Sol. Phys. 298:87 |
| SolARED (partner dataset): acoustic power, unsigned flux and continuum-intensity time series for 50 emerging ARs (2010–2023), 81 tiles, FITS download via sun.njit.edu/sarportal | Published description only | Kasapis et al. 2026, arXiv:2601.13145 |
| Interactive Multi-Instrument Database of Solar Flares (solarflare.njit.edu) | Exists (search result only) | web search |

Before opening the page I had guessed, by analogy with CCSC, that CCDC stands for "Community Coordinated Data Center". The page title confirms this. Its catalogue currently holds only FlareDB.

### 2.1 FlareDB: data-science profile

I built this profile from the full text of the *Scientific Data* paper, the Zenodo/DataCite record and the GitHub repository. The numbers for the GitHub event list are in results/datasets/flaredb_profile.json.

| Aspect | Detail |
|---|---|
| Source | SDO/HMI (`hmi.sharp_720s`, vector field, LOS magnetogram, continuum) and SDO/AIA (UV 1600/1700 Å and EUV channels) from JSOC; the GOES flare list for event selection |
| Selection | All flares ≥ M5.0 within 50° of disk centre, May 2010 to May 2025, giving 151 events from 82 ARs (the paper calls this complete for its criteria) |
| Time window per event | HMI and 12-min AIA from T₀ − 24 h to T₀ + 8 h; full-cadence AIA (12/24 s) over a shorter window around the flare |
| Processing | SunPy; AR patches in HPC and CEA projections, aligned between AIA and HMI; vector fields standardised to Bx, By, Bz; standard JSOC FITS; 13 data types |
| Distribution | Zenodo holds 5,285 quick-look movies (35 per event) in 35 zip files, plus an event list (PDF) and a data-structure figure. GitHub holds download, processing and movie scripts and a CSV event list. The full FITS cubes are described as downloadable from SolarDB |
| GitHub event list | Flare_event_list.csv has 103 rows and no header (I inferred the columns as class, NOAA AR, start, peak, end, HARPNUM and two coordinate-like columns). It contains 62 M and 41 X flares (minimum M5.0) in 57 ARs, with peaks from 2011-02-13 to 2024-05-11 and none in 2016 or 2018–2020. 45 rows lack the last two columns, and times use two formats (HH:MM and HH:MM:SS.s). The last commit is 2024-12-10, so the list is an earlier subset of the 151 events |
| Multiplicity | 19 ARs have more than one event, with up to 10 flares in one AR |
| Labels | Event-level metadata (GOES class, times). There are no negatives (non-flaring ARs or quiet intervals) |
| Licence and metadata | The paper gives CC-BY 4.0. The DataCite record lists both CC-BY-4.0 and "Copyright … All rights reserved", and gives no sizes or formats |

FlareDB is a good fit for case studies, for multimodal representation learning on the aligned HMI and AIA cubes, and for modelling pre- and post-flare evolution, for example predicting the AIA response from the HMI evolution. It also suits flare-onset timing when a flare is known to occur, and benchmarking image-translation tools such as MagNet, SolarCNN and DeepHMI on aligned patches.

On its own it cannot support flare/no-flare forecasting, because every sample is a large flare, which is a selection bias. Negatives would have to come from elsewhere, for example SHARP series of non-flaring ARs, chosen with matching selection criteria (disk position, data quality).

There is also a leakage risk. Events cluster by AR, and for closely spaced flares the −24 h/+8 h windows overlap in time. Splits must therefore be grouped by AR and blocked in time rather than drawn at random by event, and the 50° disk-centre criterion must be applied in the same way to any negative set.

Before use, the CSV needs a header, a single ISO-8601 time format and an explicit missing-value marker for the two incomplete columns. The paper documents the processing steps (projections, value ranges), which is good AI-ready practice.

## 3. Inventory of the ML-ready datasets I inspected (bundled with CCSC tools)

All numbers in this table come from results/datasets/bundled_dataset_profile.json and the per-tool audits.

| Dataset | Raw data | Format | Size | Coverage | Sample and label | Split as shipped |
|---|---|---|---|---|---|---|
| LSTM Table 2 (≥C/≥M/≥M5.0) | SDO/HMI SHARP (JSOC), GOES flares (NOAA NCEI) | CSV, 40 features + metadata | train 84,577 / val 26,473 / test 44,689 rows | 2010-05 to 2018-05 | AR × hour; flare class in next 24 h | chronological, AR-disjoint (0 shared HARPs) |
| LSTM Table 5 (≥C, incl. incomplete samples) | same | CSV (normalised) | 127,709 rows (train) | 2010-05 to 2014-01 | as Table 2 | as Table 2 |
| SolarFlareNet | SHARP, GOES | CSV, 25 SHARP + metadata | train about 180–222k, test about 63–78k rows per class/window | 2010-05 to 2022-09 | HARP × 12 min/1 h; N / P / p ("p" skipped) | 22–84 % of test (2022) rows also in training |
| SEP | SHARP, GOES, DONKI (flare–CME–SEP links) | CSV | train 7,002 / test 801 rows (24 h, FC_S) | 2011 to 2017 | HARP × time; SEP yes/no | chronological; test = 3 ARs (2 positive) |
| RNN-CME | SHARP, GOES, DONKI | CSV (normalised) | 75,063 rows (24 h train) | 2010 to 2019 | HARP × 12 min; N / P / padding | test = 2 ARs |
| FlareML | SHARP, GOES | CSV, 13 features | 845 flares | 2010-05 to 2016-12 | flare day; B/C/M/X | demo test (40) overlaps training sample (31/40) |
| MIST | identical to LSTM Table 2 ≥M train (verified) | CSV + NumPy windows | 84,577 rows, 84,553 windows | 2010-05 to 2014-01 | 24 h window; ≥M in next 24 h | random 80/20 over overlapping windows |
| CMETNet | ICME list with CME (LASCO-derived) and solar-wind features (catalogue sources as given in the paper, not verified here); LASCO C2 FITS | CSV + 351 FITS | 363 ICMEs, 672 MB images | 1996 to 2021 | CME event; transit time (h) | leave-one-year-out; best of 10 rounds chosen on the test year |
| Dst / Kp | NASA OMNIWeb hourly solar wind + indices | CSV | about 96–104k rows | 2010 to 2022 | hour; Dst (nT) / Kp | Kp clean temporal split; Dst test window also inside training frame |
| SYMHnet | OMNI 1-min/5-min + SYM-H | CSV | 315k (1-min) / 72k (5-min) train rows per storm folder | 1998 to 2018, 42 storms | minute; SYM-H | by storm (1–25 train, 26–42 test); 1 h and 2 h folders byte-identical |
| TSInet | SORCE, TCTE (LASP), NRLTSI2, SATIRE-S/M | CSV | 6,017 (SORCE) to 50,038 rows | 2003 to 2020 (+ models) | day; TSI (W m⁻²) | train SORCE, test other series |
| SINet | NOAA F10.7, Toyokawa/Nobeyama F30 | CSV + preprocessing pickle | n/a | test 2009 to 2021 | day; index value | training data not shipped |
| SolarUnet / FibrilNet / SDNN | BBSO/GST (NIRIS, Hα), SWAMIS labels, ME inversion | PNG/FITS | 3 / 5 test images; 242 train pairs | 2017–2018 | image; masks / maps | demo only |
| SolarCNN / DeepHMI / DeepGeoCME | SOHO/MDI + SDO/HMI; HMI + GST/NIRIS; SOHO + SDO | FITS / PyTorch tensors | 3 pairs / 1 pair / 33 events | n/a | image / event; HMI / NIRIS / geoeffective | demo only (SolarCNN's full set is in a restricted-access Zenodo record; the DeepHMI and DeepGeoCME archives hold only the shipped files) |
| RNN-CME CME_data_samples | as RNN-CME | CSV with header | 12–60 h: 20,347–60,769 train / 1,312–3,297 test rows (= paper Table 1) | 2010 to 2019 | HARP × 12 min; N / P | the paper's full split; column order differs from the training copy |
| MagNet validation (SolarDB) | SOHO/MDI, SDO/HMI, BBSO Hα | FITS | 12 MDI + 12 HMI cases, about 35 MB | 2010–2011 | AR image; Bx, By | validation only |
| FlareDB event list (CCDC) | GOES + JSOC | CSV, no header | 103 events (paper: 151) | 2011-02 to 2024-05 | flare event; GOES class | none |

## 4. Answers to the questions in the brief

### 4.1 Sources, formats and size

Most of the data come from NASA and NOAA archives:

- SDO/HMI SHARP parameters from JSOC (Stanford), and SDO/AIA images;
- GOES X-ray flare catalogues (NOAA NCEI), and flare–CME–SEP associations from NASA DONKI;
- solar wind and geomagnetic indices from NASA OMNIWeb/SPDF, and LASCO catalogues;
- TSI from LASP (SORCE, TCTE) and model reconstructions (NRLTSI2, SATIRE);
- ground-based BBSO/GST data from NJIT's own observatory, and the global Hα network.

Tabular time series are stored as CSV, with one row per AR and time, or per time step. Images and magnetograms are in FITS and demo masks in PNG. NumPy, Pickle and PyTorch tensors hold derived windows and models, and model weights come as HDF5, TF checkpoints or SavedModel, and .pth files.

Sizes range from kilobytes (the 845-flare table) to several GB (the SYMHnet storm folders, about 3.7 GB, and SolarFlareNet, about 1.9 GB). The full image datasets (SolarCNN, DeepHMI, FlareDB) are on Zenodo or SolarDB.

### 4.2 Collection, cleaning and integration

From the code, the typical pipeline is:

1. Download the SHARP time series for each HARP.
2. Attach NOAA AR numbers (`harp_to_noaa.csv`).
3. Join GOES flare events by AR and time to create look-ahead labels (e.g. "≥M flare within 24 h of t").
4. Derive flare-history features (counts and decayed sums of past flares).
5. Z-normalise.
6. Drop samples whose physical features contain 0, the convention for "missing".
7. Build fixed-length sequences per AR, zero-padding the history.

For the geomagnetic indices, the OMNI solar-wind parameters are merged with the index at a time offset, and sentinel fill values are then removed.

### 4.3 Missing values and quality issues

These figures come from the same dataset profile as Section 3.

In the SHARP features, 0 is the missing-value code. In LSTM Table 2, 27 % of physical-feature cells are exactly 0, and many of them are genuine zeros in flare-history counts. The `has_zero_record` filters therefore mix up "missing" and "zero", and one of them checks the wrong row (see the LSTM section of the main report).

Sentinel values remain in some solar-wind tables. In the Dst training file, 0.13 % of cells equal 9999.99-type fill values.

Timestamps are not standard. The YYYY-M-D-H format without zero padding does not sort correctly as text, and in TSInet the NRLTSI2 dates failed to parse and fell back to the current date. Timestamps are not guaranteed to be unique either, since TSInet's NRLTSI2 output repeats the same date.

Label encodings vary between Positive/Negative, P/N/p, padding and raw GOES class strings. Alphabetical label encoding silently flips the positive class for `['N','X']`, which inverted the LSTM ≥M5.0 demo.

Some loaders select columns by position. SolarFlareNet's loader assumes another table's column order, so 9 unintended features are used.

### 4.4 Metadata and support for ML

The CSV headers are informative (SHARP keywords), and the READMEs for the LSTM and RNN-CME datasets describe the columns. However, there are no machine-readable data dictionaries (units, missing codes, valid ranges), no declared split manifests and no provenance links between derived datasets (MIST from LSTM Table 2, LLMFlare from MIST). Several READMEs give software versions but not data versions or SHARP release keywords.

The datasets are ready-made supervised-learning tables, with labels attached, normalised and windowed, and they come with pre-trained models and notebooks. This is excellent for onboarding. What is missing for reliable ML is the evaluation side: recommended splits, baselines, and documented class balance and effective sample size.

### 4.5 Leakage and train/test issues

I verified each of the issues below with a script.

| Dataset | Issue | Effect measured |
|---|---|---|
| MIST | random split over 24 h windows that overlap by 23 h; scaler fitted on all rows | TSS from 0.906 to 0.197 on 2015–2018 (AUC from 0.997 to 0.833) |
| SolarFlareNet | 2022 test rows also in the training files (22–84 %) | 8/9 shipped models have no skill on the test files; C/24 h TSS 0.79 overall vs 0.697 on non-duplicated rows |
| FlareML demo | 31/40 test rows in the training sample | demo TSS is not out-of-sample; AR-grouped CV lowers macro-TSS from 0.338 to 0.246 |
| LSTM Table 5 | threshold chosen on test folds | Table 2 retraining: ≥M5.0 TSS 0.734 (fixed) / 0.825 (validation-chosen) / 0.877 (test-chosen) |
| Dst | test window inside training frame; labels used in `predict()` | raw model R² 0.06–0.38 vs reported about 0.94 |
| CMETNet | best of 10 rounds chosen on the hold-out year | test-selected MAE 0.4–2.1 h lower than the average round |
| SEP / RNN-CME | test sets of 3 and 2 ARs | very wide uncertainty on TSS about 0.86–1.00 |

## 5. AI-readiness scorecard of the bundled datasets

I adapted the criteria from the ESIP "AI-Ready Data" checklist and from Poduval et al. (2023, *Front. Astron. Space Sci.* 10, doi:10.3389/fspas.2023.1203598). In the table, ● means good, ◐ partial and ○ missing.

| Criterion | LSTM T2 | SolarFlareNet | MIST | SEP | Dst/Kp | SYMHnet | CMETNet | Image sets | FlareDB (CCDC) |
|---|---|---|---|---|---|---|---|---|---|
| Persistent identifier (DOI) | ● Zenodo | ● | ● (Zenodo, CCSC page) | ● | ● | ● | ● | ◐ | ● (Zenodo + paper) |
| Data dictionary (units, missing codes) | ◐ README | ○ | ○ | ○ | ○ | ◐ ReadMe | ○ | ○ | ◐ (paper tables; CSV has no header) |
| Provenance (raw source, versions, code) | ◐ | ◐ | ○ (undocumented reuse) | ◐ | ◐ | ◐ | ◐ | ◐ | ● (paper + scripts) |
| Explicit split manifest, grouped by time/AR | ● | ○ | ○ | ◐ | ◐ | ● (by storm) | ◐ | ○ | ○ (event list only) |
| Leakage-free as shipped | ● | ○ | ○ | ● | Kp ● / Dst ○ | ◐ (horizon) | ◐ | n/a | n/a (no task defined) |
| Baselines provided (persistence/climatology) | ○ | ○ | ○ | ○ | ○ | ◐ (GBM) | ◐ | ○ | ○ |
| Integrity (checksums) | ○ | ○ | ○ | ○ | ○ | ○ | ○ | DeepGeoCME ● | ◐ (Zenodo keeps per-file MD5; not verifiable here) |
| Licence stated consistently | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ | ◐ (CC-BY vs "All rights reserved") |
| Hosting redundancy | ● GitHub | ● | ● | ● | ● | ● | ● | FibrilNet/SDNN/MagNet ○ | ● (Zenodo + GitHub + SolarDB) |

## 6. NSF/NASA AI/ML-ready data programmes (public information)

NSF 26-512, *Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets)*, supports making existing scientific datasets usable for AI. Its scope covers AI-based feature extraction, metadata generation, dataset integration, robust automated-analysis pipelines, augmentation and harmonisation, and assessment of dataset readiness. It encourages the use of national platforms (NSF data platforms, NAIRR). Full proposals are due on 4 November 2026 and then on the first Wednesday of November each year. The award types are 10–20 Impact awards of up to $2 M, 5–10 Flagship awards of up to $5 M, and 10–20 Planning grants of up to $200 k. Proposals must also address dataset security, integrity and community governance. I took these details from the nsf.gov programme page (published 21 July 2026, updated 5 October 2026), which I opened on 2026-10-13.

NASA's ROSES-2022 B.16, *Heliophysics AI/ML-Ready Data* (HARD), NNH22ZDA001N-HARD, was aimed at new tools and methods that produce AI/ML-ready datasets from existing heliophysics mission data, including cleaning, missing-value handling and outlier removal. The ROSES-23 Step-2 deadline was moved to 2 May 2024. For ROSES-24, NASA announced no new awards and returned proposals without review, and for ROSES-25 the scope moved to B.5 RADIANT. NJIT's research office circulated the 2022 call (research.njit.edu).

Community guidance on AI-ready data includes the ESIP Data Readiness Cluster's "Checklist to Examine AI-readiness for Open Environmental Datasets", the FAIR principles, and Poduval et al. 2023 (AI-ready data in space science and solar physics: problems, mitigation and action plan). Poduval et al. stress documentation of processing steps, uncertainties, biases and instrument degradation, together with standardised metadata.

These are funding opportunities and community recommendations. Nothing here implies that Prof. Wang's group holds or has applied for any of them.

## 7. Research questions this suggests

1. Every SolarDB/CCDC dataset could ship a time- and AR-grouped split manifest together with a leakage test suite that runs automatically. MIST, SolarFlareNet and FlareML show that the measured gain is often larger than the model differences being claimed.
2. Can value-level reuse between datasets be detected automatically, as I did by hand here (MIST = LSTM Table 2)? The result could be recorded as machine-readable lineage (PROV-O, RO-Crate), giving a provenance graph across derived datasets.
3. SHARP-derived tables could replace "0 = missing" with explicit masks and quality flags. How much does this change flare-prediction skill downstream?
4. How well do MagNet, SolarCNN and DeepHMI hold up under external, cross-instrument validation? This is the question I raised in my first e-mail. Their errors could be stratified by instrument (MDI, HMI, IVM, GST), observing conditions, field strength and coverage, using FlareDB's aligned HMI/AIA patches as a testbed.
5. Persistence and climatology references and event-level bootstrap confidence intervals could be standardised, so that results such as "TSS 0.98 on 2 ARs" are reported with honest uncertainty and an effective sample size.
6. Do the Bayesian uncertainty bands of the Dst, Kp and SYMHnet models achieve their nominal coverage? The current code scales them with ad-hoc constants.
7. Turning FlareDB into a forecasting benchmark would need matched negatives (non-flaring or sub-M5 ARs under the same 50° disk-centre and quality criteria) and AR-grouped, time-blocked split manifests. Once leakage is controlled, can AIA-derived precursors in the T₀ − 24 h window improve on SHARP-only models such as the 2019 LSTM?
8. Column-order and schema tests belong in AI-readiness checks. The RNN-CME permutation was detectable in seconds from physical correlations (extensive SHARP parameters vs USFLUX), so checks like this could ship as a data-validation suite with every CCDC dataset.

## 8. References

* Nita G. et al. 2022, *Revisiting the Solar Research Cyberinfrastructure Needs: A White Paper of Findings and Recommendations*, arXiv:2203.09544.
* Liu N., Abduallah Y., Kapure T., Li Q., Wang H., Wang J. T. L. 2026, *FlareDB: A Database of Significant Flares in Solar Cycles 24 and 25 with SDO/HMI and SDO/AIA Observations*, Scientific Data, doi:10.1038/s41597-026-06607-7.
* Liu N., Abduallah Y., Kapure T., Li Q., Wang H., Wang J. 2025, *FlareDB … – Quick-look Movies* (version 2025-05-31) [Dataset], Zenodo, doi:10.5281/zenodo.16790538 (the CCDC Citation button).
* Kasapis S. et al. 2026, *SolARED: Solar Active Region Emergence Dataset for Machine Learning Aided Predictions*, arXiv:2601.13145.
* Jiang H. et al. 2023, Sol. Phys. 298:87, doi:10.1007/s11207-023-02180-z (MagNet).
* Yesmanchyk et al. 2026, *Out-of-Sample Validation of MagNet*, arXiv:2601.15926.
* Poduval B. et al. 2023, *AI-ready data in space science and solar physics: problems, mitigation and action plan*, Front. Astron. Space Sci. 10, doi:10.3389/fspas.2023.1203598.
* ESIP Data Readiness Cluster, *Checklist to Examine AI-readiness for Open Environmental Datasets*, esipfed.org.
* NSF 26-512, *Unlocking Dataset Value for AI-Enabled Scientific Discovery (AI Datasets)*, nsf.gov.
* NASA ROSES-2022 B.16, *Heliophysics Artificial Intelligence/Machine Learning-Ready Data* (NNH22ZDA001N-HARD), nspires.nasaprs.com; ROSES-2025 B.5 RADIANT.
* Bloomfield D. S. et al. 2012, ApJL 747:L41 (why TSS for flare forecasting).
