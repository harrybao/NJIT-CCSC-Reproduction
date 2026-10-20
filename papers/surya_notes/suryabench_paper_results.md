# SuryaBench paper: datasets and results inventory

Paper: *SuryaBench: Benchmark Dataset for Advancing Machine Learning in Heliophysics and Space Weather Prediction* (arXiv 2508.14107).
Notes compiled 2026-10-15 (UTC) using curl through the session proxy. No file larger than 50 MB was downloaded. Nothing was installed, and no account or token was used.
The evidence files (API JSON, cards, extracted tables, CSV analysis) are in `/home/user/work/surya_scout/research/suryabench_evidence/`.

---

## 0. TL;DR

* **arXiv has only v1** (submitted Mon, 18 Aug 2025 00:05:01 UTC, 16,159 KB, licence CC BY 4.0). `https://arxiv.org/abs/2508.14107v2` returns HTTP 404, and the arXiv abstract page has no journal-ref or DOI field.
* **A peer-reviewed version exists, and its numbers differ a lot from arXiv v1:** *Scientific Data* **13**, Article 712 (2026), DOI **10.1038/s41597-026-06552-5**. Received 29 July 2025, accepted 29 December 2025, published 2 March 2026. Crossref records the licence as CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0). It has 25 authors, the same list as arXiv in a slightly different order. An 8-page Supplementary Information PDF goes with it (`41597_2026_6552_MOESM1_ESM.pdf`).
  * The **published version matches the current Hugging Face data**: its dataset references cite exact HF revisions, and those equal the current `sha` of every dataset (see §2).
  * The **arXiv v1 numbers (N per dataset, baseline tables) are out of date**, and several of its tables have internal inconsistencies (§5).
* **Neither version reports any Surya numbers.** The only foundation-model numbers are SSIM/RMSE figures in the text for "the modified long-short Spectral Transformer [4]", where ref. [4] is Roy et al. 2024, arXiv:2410.10841 (HelioFM). All tables hold conventional CNN/U-Net baselines. Surya-on-SuryaBench results belong to the Surya paper (arXiv 2508.14112) and are not covered here.
* **Hugging Face:** the 7 SuryaBench datasets are all under `nasa-ibm-ai4science`. All are public, **not gated**, licence **cc-by-4.0**, last modified 2025-12-16.
  * The collection link in arXiv v1's *Code Availability* (`huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b`) returns **HTTP 401**: it is private or deleted, and so is `nasa-impact/suryabench`, which the GitHub README links to.
  * The link in arXiv v1's *Data Records* (`nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c`) works and redirects to `/collections/nasa-ibm-ai4science/suryabench`.
* **The full core SDO data is not on HF.** It is in the AWS Open Data bucket `s3://nasa-surya-bench` (us-west-2, no account needed). I listed it in full: **615,651 objects, 349,466,462,822,109 bytes (349.47 TB = 317.84 TiB)**, from `2010/05/20100513_0012.nc` to `2024/12/20241231_2348.nc`. On HF, `core-sdo` holds 363.38 GB: the validation and test tar parts, 15 sample files, and index CSVs.

---

## 1. Versions and sources used

| Source | URL / location | Status |
|---|---|---|
| arXiv abstract | https://arxiv.org/abs/2508.14107 | 200; only `[v1]` listed |
| arXiv HTML v1 | https://arxiv.org/html/2508.14107 | 200. Note: the HTML (LaTeXML) render shows every `\dsref{}` as "DS2", but the TeX has DS1–DS6 (`\dslabel{1..6}`) |
| arXiv TeX source v1 | https://arxiv.org/src/2508.14107v1 (`arXiv-2508.14107v1.tar.gz`, 16,546,502 bytes) | 200; tables were copied from `main.tex` (saved as `suryabench_evidence/paper/arxiv_2508.14107v1_main.tex`) |
| arXiv BibTeX | https://arxiv.org/bibtex/2508.14107 | 200 (`roy2025suryabenchbenchmarkdatasetadvancing`, 25 authors, eprint 2508.14107, astro-ph.SR) |
| Scientific Data article | https://www.nature.com/articles/s41597-026-06552-5 (+ `/tables/1`, `/tables/2`, `/tables/3`) | 200 |
| Scientific Data SI | https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41597-026-06552-5/MediaObjects/41597_2026_6552_MOESM1_ESM.pdf (3,316,695 bytes, 8 pages, created 2026-02-17) | 200; **holds no result tables** (its "Table ??" references are broken) |
| Crossref | https://api.crossref.org/works/10.1038/s41597-026-06552-5 | 200 |
| GitHub NASA-IMPACT/SuryaBench | `git ls-remote` works: HEAD = `refs/heads/main` = `refs/tags/V1.0` = `ce8b841bb999a7a917bcc004cd17a57ca2109b81`. The README came from raw.githubusercontent.com | github.com HTML returned 403 through the proxy, the GitHub REST API is blocked for this session, and my `add_repo` request was denied. **I did not inspect the repository contents beyond the README** |
| AWS Open Data registry | https://registry.opendata.aws/surya-bench/ | 200 (but see §6: its "Documentation" link is HTTP 401, and its description says the data ends **July 31, 2024**) |

Citation records found: arXiv BibTeX (above). The GitHub README has `@misc{suryabench, ... year={2025}}` with **21** authors and no eprint (it lacks Kshitiz Mandal, Iksha Gurung, Campbell Watson and Juan Bernabe-Moreno). Each HF dataset has a DOI (see §2). The AWS registry's "How to Cite" points to Roy et al. 2024, arXiv:2410.10841, not to SuryaBench.

---

## 2. Benchmark dataset inventory

### 2.1 Paper-level description (as printed)

**arXiv v1, Table 2** ("Statistic summary of auxiliary datasets", copied from `main.tex`):

| | Application | Broader Topic | N | C | T | H | W | Dynamic Range |
|---|---|---|---|---|---|---|---|---|
| DS1 | AR segmentation | Magnetic energy in solar atmosphere | 109,175 | 1 | 1 | 4,096 | 4,096 | 0,1 |
| DS2 | AR emergence forecasting | Magnetic energy in solar atmosphere | 50 | 6 | 240 | 1 | 1 | −1.7×10^4 to 4.0×10^3 |
| DS3 | Coronal field extrapolation | Coronal structure & magnetism | 51,156 | 2 | 1 | 4,186 | 1 | −4.3×10^3 to 4.3×10^3 |
| DS4 | Flare forecasting | Space weather forecasting | 128,352 | 1 | 1 | 1 | 1 | 0,1 |
| DS5 | Solar wind forecasting | Solar forcing of magnetosphere | 120,748 | 1 | 1 | 1 | 1 | 2.4×10^2 to 8.8×10^2 |
| DS6 | EUV forecasting | Solar forcing of ionosphere | 189,397 | 1,343 | 1 | 1 | 1 | 1.0×10^−9 to 1.1×10^−2 |

**Scientific Data 2026, Table 1** ("Statistical summary of auxiliary datasets", copied from `/tables/1`):

| | Application | Broader Topic | N | C | T | H | W | Dynamic Range |
|---|---|---|---|---|---|---|---|---|
| DS1 | AR Segmentation | Magnetic energy in solar atmosphere | 121,963 | 1 | 1 | 4,096 | 4,096 | {0, 1} |
| DS2 | AR Emergence Forecasting | Magnetic energy in solar atmosphere | 56 | 6 | 240 | 9 | 9 | − 1.7 × 10^4 to 4.0 × 10^3 |
| DS3 | Coronal Field Extrapolation | Coronal structure & magnetism | 5,347 | 1 | 12 | 1 | 4,095 (2) | − 4.3 × 10^3 to 4.3 × 10^3 |
| DS4 | Flare Forecasting | Space weather forecasting | 128,328 | 2 | 1 | 1 | 1 | {0, 1} |
| DS5 | Solar Wind Forecasting | Solar forcing of magnetosphere | 119,225 | 5 | 1 | 1 | 1 | 2.4 × 10^2 to 1.096 × 10^3 |
| DS6 | EUV Forecasting | Solar forcing of thermosphere | 189,344 | 1,343 | 1 | 1 | 1 | 1.0 × 10^−9 to 1.1 × 10^−2 |

Footnote: "The shape of all datasets has been standardized with the following dimensions: N → Number of datapoints, C → Number of channels, T → Number of timestamps, H → Height, and W → Width."

### 2.2 Per-dataset details, cross-checked against Hugging Face

Method:
* Metadata came from `https://huggingface.co/api/datasets/<id>`.
* File sizes came from `.../tree/main?recursive=true`, using the LFS size where present.
* Split row counts came from `https://datasets-server.huggingface.co/size?dataset=<id>`.
* For every CSV under 50 MB, I downloaded the file and counted rows and time ranges myself (`suryabench_evidence/hf/csv_analysis.txt`, `csv_stats.txt`).

Every dataset below has `gated: False`, `private: False`, `disabled: False`, licence `cc-by-4.0`, and lastModified on 2025-12-16.

`usedStorage` is the API's storage figure, which includes git history. "Current files" is the sum of the files at `main`.

#### DS1: Active Region (with PIL) segmentation, "ARPIL"
* **HF id:** `nasa-ibm-ai4science/surya-bench-ar-segmentation`. Revision `615a7df4ac3498c97d0d039e7b0654774ad86602`, which matches the published ref 61 "revision 615a7df". DOI 10.57967/hf/7278.
* **Task:** binary segmentation (4096×4096) of ARs that contain polarity inversion lines.
* **Source instrument:** SDO/HMI line-of-sight magnetograms. Masks use thresholds of ±50 G, a size filter of 100 px, and 10-px dilation.
* **Period:**
  * arXiv v1 text: "January 2011 to December 2024", "119,454 full-disk AR binary masks covering 14 years". Note that arXiv Table 2 says N = 109,175.
  * Published text: "covers between May 13, 2024 to December 31, 2024 … 121,963 full-disk AR binary masks covering 14 years" (the start year is printed as 2024).
  * HF card: May 13, 2010 – Dec 31, 2024.
* **Splits (HF CSV, my counts):**

  | split | rows | rows with `present`=1 | time range | rule (HF card) |
  |---|---|---|---|---|
  | train | 74,760 | 70,823 | 2010-05-13 00:00 → 2019-12-31 23:00 | Feb 15–Dec 31 of each year 2010–2019 |
  | validation | 3,672 | 3,477 | 2011-01-15 → 2019-01-31 23:00 | Jan 15–31 |
  | test | 43,848 | 41,910 | 2020-01-01 → 2024-12-31 23:00 | all of 2020–2024 |
  | leaky_validation | 6,048 | 5,753 | 2011-01-01 → 2019-02-14 23:00 | Jan 1–14 and Feb 1–14 |

  The rows total **128,328**, and the `present`=1 rows total **121,963**, which equals the published N exactly. The HF card says "Total 128,352 instances".
* **Size on disk (HF):**
  * Current files total 1,313,355,815 bytes. `data.tar.gz` is 1,306,605,651 bytes; the CSVs are 3,922,895 + 192,661 + 2,311,618 + 318,065 bytes.
  * `usedStorage` is 5,971,018,868.
  * The card says "~ 53kB" per file, "~ 1.2GB" in total, and "[2, 4096, 4096] per file" (keys `intersection` and `union_with_intersect`).
  * The published *Data Records* says `data.tar.gz` contains "yearly directories spanning 2020 to 2024".
  * I did not open the archive (it is above 50 MB), so whether training-year masks are in it is **unverified**. My own arithmetic: 121,963 × 53 kB ≈ 6.5 GB, which does not fit inside 1.2 GB.

#### DS2: Active Region emergence forecasting
* **HF id:** `nasa-ibm-ai4science/ar_emergence`. Revision `48970a12c290b515912baadba87fa74a49e3c680`, matching published ref 62 "revision 48970a1". DOI 10.57967/hf/7279.
* **Task:** spatio-temporal regression that predicts continuum intensity per tile from acoustic power (4 bands: 2–3, 3–4, 4–5, 5–6 mHz) and magnetic flux. The setup in the SI is T = 120, C = 5, S = 63 (7×9 tiles).
* **Source instrument:** SDO/HMI (Doppler velocity → acoustic power, magnetic flux, continuum intensity), tracked over a 512×512 px patch (the card says 30°×30°) on a 9×9 grid.
* **Period:**
  * arXiv: ARs "between March 1st, 2010 and June 1st, 2023".
  * HF card: 2010 – 2023.
  * Published SI: "November 2010 to February 2023".
  * HF CSVs: `t_start` runs from 2010.11.28_23:59:59Z to 2023.02.07_00:00:00Z.
* **Number of ARs:** arXiv says **50**, published text, Table 1 and Data Records say **56**, and the HF card and CSVs say **59**. The SI says "Approximately two-thirds of the instances (37/59) belong to the training set". The SI also says, separately, "The dataset comprises 3,479 unique regions".
* **Splits (HF CSV):** train 37 ARs, validation 9, test 13 (total 59). The published SI rule is: train Feb 15–Dec 31 of 2010–2019, validation Jan 1–Feb 14 of 2010–2019, test all of 2020–2023. The card calls the validation file `valid.csv`, but the repository file is `validation.csv`.
* **Size on disk (HF):** current files total 23,102,338 bytes (`data.zip` 23,085,581). `usedStorage` is 3,809,756,040. The card says "~23.1 MB" and "(6, 240, 9, 9) per instance". The `.npz` names differ: the card has `mean_int`, `mean_mag`, `mean_pmdop`, while the published Data Records has `mean_mag`, `mean_vel`, `mean_int`.

#### DS3: Coronal magnetic field extrapolation (ADAPT-WSA PFSS emulation)
* **HF id:** `nasa-ibm-ai4science/surya-bench-coronal-extrapolation`. Revision `622892941688f21e69bbad37598b33787083cdbd`, matching published ref 58 "revision 6228929". DOI 10.57967/hf/7275.
* **Task:** regress the spherical-harmonic coefficients of the PFSS magnetic potential: G and H, Schmidt-normalised, truncated at order 90, source surface at 2.51 R☉. There are 12 ADAPT realisations, R000–R011.
* **Source:** the ADAPT + WSA models driven by SDO/HMI magnetograms, at daily cadence.
* **Period:** arXiv says an "11-year (full solar cycle) span", the published version says "approximately 15-year span (Solar Cycle 24 and partially Solar Cycle 25)", and the HF card says May 2010 – Dec 2024. The CSVs run from 2010-05-13 08:00 to 2024-12-31 08:00.
* **N:** arXiv says "51,156 sets of harmonic coefficients". The published version says "5,347 instances; each instance is composed of 12 realizations in 2 channels … 4,186 harmonic coefficients", and its Table 1 W column says "4,095 (2)". The HF card says "5,347 instances * 12 realizations".
* **Splits (HF CSV):**

  | split | timestamps | non-empty realisation paths | time range | rule (HF card) |
  |---|---|---|---|---|
  | train | 3,115 | 37,380 | 2010-05-13 → 2019-12-31 | Feb 15–Dec 31, 2010–2019 |
  | validation | 153 | 1,836 | 2011-01-15 → 2019-01-31 | **Jan 15–28** |
  | test | 1,827 | 21,924 | 2020-01-01 → 2024-12-31 | all of 2020–2024 |
  | leaky_validation | 252 | 3,024 | 2011-01-01 → 2019-02-14 | **Jan 1–14 and Jan 29–Feb 11** |

  There are 5,347 timestamps in total, which gives 64,164 FITS paths. The card's validation and leaky windows for DS3 differ from the generic SI rule (Jan 15–31; Jan 1–14 and Feb 1–14).
* **Size on disk (HF):** current files total 36,181,107,009 bytes. `data.tar.gz` is **36,176,131,632 bytes**; the CSVs are 1,408,050 + 69,226 + 825,874 + 113,974 bytes. `usedStorage` is 212,586,020,461. The card says "~36.2 GB (compressed)" and "~763 KB" per file. A 2,293,760-byte example tar (`CoronalFieldExtrapolation_examples.tar`) is also provided.

#### DS4: Full-disk solar flare forecasting
* **HF id:** `nasa-ibm-ai4science/surya-bench-flare-forecasting`. Revision `bf474bc1e7752bd529e9707650f9310db8693763`, matching published ref 59 "revision bf474bc". DOI 10.57967/hf/7274.
* **Task:** binary classification over a 24-hour window at hourly cadence. There are two labels: `label_max` (maximum GOES class at M1.0 or above) and `label_cum` (cumulative index of 10 or above). The CSV also carries `max_goes_class` and `cumulative_index`.
* **Source:** GOES X-ray flare events (labels); the inputs are core SDO images.
* **Period:** 2010-05-13 00:00 → 2024-12-31 23:00 (from `data.csv`).
* **N:** arXiv says "total 128,352 labels". The published version and the HF card both say 128,328, and `data.csv` has **128,328** rows.
* **Splits (HF CSV, my counts):**

  | split | rows | `label_max`=1 | `label_cum`=1 |
  |---|---|---|---|
  | train | 74,760 | 9,051 (0.1211) | 18,017 (0.2410) |
  | validation | 3,672 | 400 (0.1089) | 743 (0.2023) |
  | test | 43,848 | 12,903 (0.2943) | 21,587 (0.4923) |
  | leaky_validation | 6,048 | 901 (0.1490) | 1,620 (0.2679) |

  The rules are the same as DS1. In `data.csv`, 23,255 rows have `label_max`=1 and 41,967 have `label_cum`=1, and 28,563 rows are `FQ` (flare-quiet).
* **Threshold check:** all 1,511 windows whose maximum class is `M1.0` have `label_max`=1, and all 150 windows with `cumulative_index`=10.0 have `label_cum`=1. The thresholds are therefore **inclusive (≥)**, matching the HF card.
  * Both paper versions write "L_max > M1.0 and L_cum > 10".
  * The SI (published and arXiv) says the max-label threshold is "10−4 W/m2", which corresponds to X1.0, not M1.0.
* **Size on disk (HF):** current files total 8,760,535 bytes (`data.csv` is 4,377,302). `usedStorage` is 2,591,745. The card says "~4.4MB".

#### DS5: Solar wind speed forecasting
* **HF id:** `nasa-ibm-ai4science/Surya-bench-solarwind`. Revision `20547a25cbd3aaa359e9cda18be913b94c225e7b`, matching published ref 60 "revision 20547a2". DOI 10.57967/hf/7276.
* **Task:** regress the solar-wind speed V at t + 4 days from SDO images (the 4-day horizon is stated in the SI).
* **Source:** OMNIWeb hourly data at L1, shifted to the bow-shock nose. arXiv v1 names ACE. The published version names ACE SWEPAM and MAG and says ICMEs were removed (background wind only). The CSV columns are `timestamp, V, BX_GSE, BY_GSM, BZ_GSM, N`.
* **Period:**
  * arXiv: "14 years from 2010 to 2023". The SI, in both versions, still says "2010-01-01 through 2023-12-31".
  * Published main text: "about 15 years from 2010 to 2024".
  * HF CSVs: 2010-05-13 07:00 → 2024-12-31 00:00.
* **N:** arXiv says "120,748 measurements". The published version and the HF card both say **119,225**, and my CSV count sums to 119,225.
* **Splits (HF CSV):**

  | split | rows | V min | V max | time range |
  |---|---|---|---|---|
  | train | 69,957 | 240.0 | 800.0 | 2010-05-13 07:00 → 2019-12-31 23:00 |
  | validation | 3,429 | 259.0 | 753.0 | 2011-01-15 → 2019-01-31 23:00 |
  | test | 39,975 | 256.0 | 1006.0 | 2020-01-01 → 2024-12-31 00:00 |
  | leaky_validation | 5,864 | 266.0 | 878.0 | 2011-01-01 → 2019-02-14 23:00 |

  The overall V range in the HF data is 240.0 to **1006.0** km/s. The published Table 1 and text say "2.4 × 10^2 to 1.096 × 10^3", and arXiv says "2.4×10^2 to 8.8×10^2". No V values are missing.
* **Size on disk (HF):** current files total 5,332,522 bytes. `usedStorage` is 2,039,220.

#### DS6: Solar EUV spectra (SDO/EVE MEGS-A)
* **HF id:** `nasa-ibm-ai4science/euv-spectra`. Revision `aafccf9c1364bd39eaf49e058e39f24f70719657`, matching published ref 63 "revision aafccf9". DOI 10.57967/hf/7280.
* **Task:** regress a 1,343-bin EUV spectrum from AIA images ("image-to-spectra").
* **Source:** SDO/EVE MEGS-A Level 2B (version 8), averaged over 1 minute and matched to AIA's 12-minute cadence. GOES/XRS flare catalogue for event stratification.
* **Wavelength grid:** I verified `euv_wavelengths.csv`: 1,343 bins from 6.489999771118164 to 33.33000183105469 nm, spacing 0.0200 nm. This matches the published "6.49 nm to 33.33 nm with a 0.02 nm spectral spacing".
* **Period:**
  * HF card: May 2010 – May 2014. SI: "May 13, 2010 to May 26, 2014". Published text: "spanning 2010-2014 during Solar Cycle 24".
  * arXiv v1 instead says "covering Solar Cycle 24 and parts of Solar Cycle 25", which contradicts the MEGS-A period.
* **N:** arXiv says 189,397, while the published version and the HF card say 189,344. The datasets-server totals **189,344** rows with 1,344 columns.
* **Splits:**

  | split | rows (datasets-server) | file size (bytes) | rule (SI/HF card) |
  |---|---|---|---|
  | train | 163,097 | 2,984,963,110 | Feb 15–Dec 31 of 2010–2014 |
  | validation | 3,650 | 66,789,263 | Jan 15–31 of 2012 and 2013 |
  | test | 6,051 | 110,815,206 | Jan 15–31 of 2011 and 2014 |
  | leaky_validation | 16,546 | 302,924,931 | Jan 1–14 and Feb 1–14 of 2010–2014 |

* **Size on disk (HF):** current files total 12,062,209,078 bytes. The top-level files take 3,465,525,883 bytes and an `archive/` folder of older and stratified files takes 8,596,683,195 bytes. `usedStorage` is 14,497,894,075. The card says "~3.2GB". The Parquet conversion (datasets-server) is 2,435,718,370 bytes.
* **Dynamic range:** the card says "1e-9 to 1e-1 W/m^2/nm", and both paper versions say "1.0 × 10−9 to 1.1 × 10−2". I did **not** verify this, because every spectra CSV is larger than 50 MB.

#### Core SDO dataset (pre-training input; not a DS# task)
* **HF id:** `nasa-ibm-ai4science/core-sdo`. Revision `9c10abe52c1f206d1b66715745229e29ef543cc6`, matching published ref 64 "revision 9c10abe". DOI 10.57967/hf/7273. Licence cc-by-4.0, not gated.
* **Instruments and channels:** 13 channels in total.
  * SDO/AIA (8): 94, 131, 171, 193, 211, 304, 335 and 1600 Å.
  * SDO/HMI (5): `hmi_m` (LOS magnetogram), `hmi_bx`, `hmi_by`, `hmi_bz` and `hmi_v`.
  * JSOC series: `aia.lev1_euv_12s`, `aia.lev1_uv_24s`, `hmi.M_720s`, `hmi.B_720s`, and `hmi.V_720s` (the last is named only in the published version).
  * Format: level 1.5, 0.6″/px, disk radius fixed at 976″, 12-minute cadence, netCDF float32 of shape [13, 4096, 4096].
* **Period:** May 13, 2010 – Dec 31, 2024 (paper and card). arXiv v1's abstract says "May 2010 to July 2024", and the AWS registry also says "July 31, 2024".
* **Splits as printed:**
  * arXiv v1: "training (2010-2018), validation (2019), and test (2020) sets, which include 379,920, 43,680, and 43,800 files, respectively". My arithmetic: 43,800 = 365 × 120 and 43,680 = 364 × 120, which are full 12-minute slot counts for a whole year.
  * Published: these counts are removed. It refers to the HF `train.csv`/`validation.csv`/`test.csv` and to the generic SI rule (train Feb 15–Dec 31, val Jan 15–31, leaky Jan 1–14 and Feb 1–14 for 2010–2019, test 2020–2024).
* **What HF actually holds** (363,376,705,743 bytes; `usedStorage` 363,375,594,449):
  * `test_splitted_tars/test.tar.part_aa…aq`: 17 parts, 175,829,800,960 bytes.
  * `validate_splitted_tars/val.tar.part_aa…aq`: 17 parts, 177,452,584,960 bytes.
  * `infer_data/`: 15 `.nc` files of 662–673 MB each plus `infer_index_surya_1_0.csv`. The files are dated 2011-01-20, 2012-01-15, 2013-01-28 and 2019-01-23.
  * `train_index_surya_1_0.csv`: 345,721 rows (309,412 present), 2011-02-15 → 2019-12-31.
  * `valid_index_surya_1_0.csv`: 18,360 rows (17,710 present), Jan 15–31 of 2011–2019.
  * `train.csv` (2,880 rows), `val.csv` (360) and `test.csv` (361): all fall in January 2011, so they are sample lists only. There is **no `validation.csv`**; the file is named `val.csv`.
  * `scalers.yaml`.
  * There are **no training-year image files** on HF.
* **AWS S3 `nasa-surya-bench`** (full anonymous listing, per year):

  | year | objects | bytes |
  |---|---|---|
  | 2010 | 26,789 | 15,723,231,988,505 |
  | 2011 | 41,886 | 24,608,346,206,438 |
  | 2012 | 42,381 | 24,676,121,092,039 |
  | 2013 | 41,439 | 24,307,791,248,883 |
  | 2014 | 42,230 | 24,821,084,655,420 |
  | 2015 | 42,162 | 24,733,141,577,205 |
  | 2016 | 41,387 | 24,138,505,988,229 |
  | 2017 | 42,109 | 24,486,223,785,505 |
  | 2018 | 42,067 | 24,313,981,031,688 |
  | 2019 | 42,257 | 24,275,668,547,225 |
  | 2020 | 42,338 | 24,177,309,454,229 |
  | 2021 | 42,079 | 24,064,998,961,923 |
  | 2022 | 42,162 | 23,552,656,178,057 |
  | 2023 | 42,046 | 23,446,820,526,353 |
  | 2024 | 42,319 | 18,140,581,580,410 |
  | **total** | **615,651** | **349,466,462,822,109** |

  * Objects from 2010 to 2018 total 362,450, against the 379,920 files arXiv v1 gives for training. 2019 has 42,257 objects (arXiv: 43,680) and 2020 has 42,338 (arXiv: 43,800).
  * Typical file sizes are about 580 MB. Many 2024 files are smaller, about 313–327 MB (sampled from 2024/01, /06 and /12). I did not investigate why.
  * Paper and card: "about 600 MB" per file (arXiv) or "about 570 MB" (published) or "~570 MB" (card), and "approximately 360 TB" in total.

#### Related (not part of SuryaBench): `nasa-ibm-ai4science/Surya-1.0_validation_data`
This is the validation set for Surya 1.0. Licence **MIT**, not gated, revision `3b6710423d8ca10889e7d9f154f587b91266f6be`. It holds 24 `.nc` files from 20140107_1500 to 20140107_1924, totalling 15,427,169,811 bytes (about 670.3–671.0 MB each). It is not in the SuryaBench collection. Note that it is larger than the roughly 9 GB of free disk.

---

## 3. Quantitative results: arXiv v1 (2508.14107v1)

All values are copied from `main.tex` (Section "Technical Validation" and the Supplement). Hardware as stated: "All experiments were performed on 4 Nvidia A100 GPUs with 80 GB of memory."

### 3.1 Results in the text (core SDO forecasting, HelioFM)
"…the modified long-short Spectral Transformer [4] model demonstrated strong performance after training for just 20 epochs on four years of data."

| Channel | SSIM | RMSE |
|---|---|---|
| AIA 171 Å | 0.83 | 0.11 |
| AIA 193 Å | 0.90 | 0.095 |
| AIA 211 Å | 0.86 | 0.10 |
| remaining AIA bands | "ranging from 0.4 to 0.65" | not given |
| HMI channel | 0.73 | 0.65 |

### 3.2 Table 3: "Baseline performance for (a) solar wind prediction and (b) solar flare classification using common deep learning models on test data"

**(a) Solar Wind Forecasting**

| Model | RMSE | MAE | Val Loss |
|---|---|---|---|
| UNet | 0.1499 | 0.1116 | 0.0225 |
| AttentionUNet | 0.1449 | 0.1157 | 0.0225 |
| ResNet18 | 0.2108 | 0.2388 | 0.0233 |
| ResNet34 | 0.1462 | 0.1149 | 0.0226 |
| ResNet50 | 0.1445 | 0.1145 | 0.0221 |

**(b) Solar Flare Prediction**

| Model | TSS | HSS | CSS | F1 |
|---|---|---|---|---|
| AlexNet | 0.359 | 0.354 | 0.356 | 0.679 |
| MobileNet | 0.326 | 0.312 | 0.319 | 0.662 |
| ResNet18 | 0.320 | 0.317 | 0.318 | 0.660 |
| ResNet34 | 0.290 | 0.289 | 0.289 | 0.645 |
| ResNet50 | 0.261 | 0.281 | 0.271 | 0.627 |

The text calls F1 "F1-macro".

### 3.3 Table 4: "Baseline performance for (a) EVE Prediction and (b) AR Emergence Forecasting on test data"

**(a) EVE Prediction**

| Model | RMSE | MAE | Val Loss |
|---|---|---|---|
| UNet | 0.0754 | 0.0558 | 0.00569 |
| AttentionUNet | 0.0754 | 0.0558 | 0.00569 |
| ResNet18 | 0.2108 | 0.2388 | 0.02330 |
| ResNet34 | 0.1462 | 0.1149 | 0.02255 |
| ResNet50 | 0.1445 | 0.1145 | 0.02208 |

**(b) AR Emergence Forecasting (ST: Spatiotemporal)**

| Model | MSE | RMSE |
|---|---|---|
| ST Attention | 0.1538 | 0.3921 |
| ST ResNet | 0.1527 | 0.3908 |
| (LSTM) | 0.0140 | 0.1180 |

The row label is printed literally as "(LSTM)". The TeX has a `%% TODO LSTMS` comment next to it.

### 3.4 Supplementary Table 5: "Solar Flare Forecasting Performance"
This is identical to Table 3(b) (AlexNet 0.359/0.354/0.356/0.679 … ResNet50 0.261/0.281/0.271/0.627). The supplement adds: "The results indicate that AlexNet performs best among the evaluated architectures across all metrics".

### 3.5 Other result-like statements (no numbers)
* Solar wind: "we achieved optimal performance metrics and lowest validation loss with the ResNet-50 architecture … (33 million parameters)". Figure 8 shows training-loss curves for ResNet 18/34/50/101/152, U-Net and Attention U-Net.
* EVE: Figure 10 shows training-loss curves.
* Figure 6 shows "Results for AR emergence forecasting using the HelioFM when validating on AR11698 (left) and AR11726 (right)". It is a plot only, with no numbers.
* **No baseline results** are given for AR segmentation (DS1) or coronal field extrapolation (DS3).

---

## 4. Quantitative results: Scientific Data 13:712 (2026), published version

Hardware: "All experiments were performed on a single Nvidia A100 GPU with 80 GB of memory." This changed from the 4 A100s in arXiv v1.
Main text: "Baseline results are currently provided for four tasks-solar flare prediction, active region segmentation, solar irradiance prediction, and solar wind forecasting". DS2 and DS3 have no baselines.

### 4.1 Results in the text (core SDO forecasting, HelioFM)
The values are identical to arXiv v1: SSIM 0.83 / 0.90 / 0.86 for 171 / 193 / 211 Å, remaining bands 0.4–0.65, RMSE 0.11 / 0.095 / 0.10, and HMI SSIM 0.73 with RMSE 0.65. The training period is now specified as "four years' (2011–2014)".

### 4.2 Table 3: "Results from baseline evaluation experiments across four application datasets … (a) solar wind forecasting, (b) solar irradiance prediction, (c) solar flare forecasting, and (d) active region segmentation using standard deep learning architectures on held-out test datasets."
Footer: "Sampling splits are outlined in Supplementary Information. The evaluation scores are obtained from the testing set."

**(a) Solar Wind Forecasting**

| Model | RMSE (km/s) | MAE (km/s) |
|---|---|---|
| AlexNet | 123.77 | 108.99 |
| MobileNet | 125.75 | 111.04 |
| ResNet18 | 123.11 | 106.60 |
| ResNet34 | 154.58 | 127.25 |
| ResNet50 | 118.34 | 102.65 |

**(b) EVE Prediction**

| Model | MSE | MAE |
|---|---|---|
| AlexNet | 0.0001 | 0.007 |
| MobileNet | 0.0792 | 0.253 |
| ResNet18 | 0.0078 | 0.069 |
| ResNet34 | 0.0902 | 0.236 |
| ResNet50 | 0.0339 | 0.139 |

**(c) Solar Flare Prediction**

| Model | TSS | HSS | F1 | Precision | Recall |
|---|---|---|---|---|---|
| AlexNet | 0.16 | 0.21 | 0.29 | 0.85 | 0.17 |
| MobileNet | 0.53 | 0.44 | 0.65 | 0.51 | 0.89 |
| ResNet50 | 0.05 | 0.08 | 0.11 | 0.85 | 0.06 |

The SI says: "Note that for our experiments, we used the Maximum Flare Intensity label", and "MobileNet achieved the best overall performance across TSS, HSS, and F1 metrics".

**(d) Active Region Segmentation**

| Model | IoU | Dice |
|---|---|---|
| UNet | 0.62 | 0.76 |
| Attention UNet | 0.52 | 0.68 |

The SI PDF has **no numeric tables**. Its references to "Table ??" are unresolved.

### 4.3 Comparison between versions
The published version replaces every arXiv v1 baseline table:
* Solar wind is now reported in km/s with AlexNet/MobileNet/ResNet, where arXiv v1 used normalised RMSE/MAE/Val Loss with UNet/ResNet.
* The EVE metrics changed.
* Flare results now include precision and recall and drop CSS, ResNet18 and ResNet34. AlexNet, previously the best, is now beaten by MobileNet.
* AR segmentation (IoU/Dice) is new.
* AR emergence results (ST Attention, ST ResNet, (LSTM)) are dropped.

The two versions share no values except the HelioFM SSIM/RMSE text.

**The code locations differ between versions:**
* arXiv v1: "Our code for dataset preparation and creation, and baseline model training is publicly available at https://github.com/NASA-IMPACT/SuryaBench".
* Published: dataset pipelines are at github.com/NASA-IMPACT/SuryaBench, and "the baseline models and training scripts used for evaluation are available at https://github.com/NASA-IMPACT/Surya/tree/main/downstream_examples". The local clone at `/home/user/upstream/surya/Surya` (v1.0.0) has `downstream_examples/{ar_segmentation, euv_spectra_prediction, solar_flare_forcasting, solar_wind_forcasting}`. For example, `euv_spectra_prediction/config.yaml` lists `model_type: spectformer # Options: spectformer, resnet18, resnet34, resnet50, alexnet, mobilenet`. I did not run anything in it.

---

## 5. Internal consistency checks

All checks below are my own arithmetic, done without running any model.

1. **arXiv v1 Table 3(a) and Table 4(a): MAE is larger than RMSE for ResNet18** (RMSE 0.2108 < MAE 0.2388). For any single set of errors, MAE ≤ RMSE always holds, so this row cannot be correct as labelled.
2. **arXiv v1 Table 4(a) (EVE) copies Table 3(a) (solar wind)** for the ResNet rows: RMSE/MAE 0.2108/0.2388, 0.1462/0.1149 and 0.1445/0.1145 are identical. The Val Loss values are the same numbers with more digits (0.0233 → 0.02330, 0.0226 → 0.02255, 0.0221 → 0.02208). UNet and AttentionUNet are also identical within Table 4(a). These rows look like copy-paste errors.
3. **N changes between arXiv v1 and the published version for every dataset** (§2.1). The published N matches the HF data for DS1 (121,963 present masks), DS3 (5,347 timestamps), DS4 (128,328), DS5 (119,225) and DS6 (189,344). It does **not** match for DS2: the published version says 56 ARs, while HF and the SI say 59.
4. **Within arXiv v1, DS1 has two values:** Table 2 says N = 109,175 and the text says 119,454.
5. **Published Table 1, DS3:** W = "4,095 (2)", but the text says "4,186 harmonic coefficients". For a (91×91) upper-triangular array, 91·92/2 = 4,186.
6. **DS5 dynamic range:** the published version says "1.096 × 10^3" km/s, but the maximum V in the HF CSVs is 1006.0 km/s (test split).
7. **Flare thresholds:** the paper writes ">" and the data uses "≥" (§2.2 DS4). The SI's "10−4 W/m2" corresponds to X1.0, not M1.0.
8. **Published DS1 period** is printed as "May 13, 2024 to December 31, 2024" next to "covering 14 years".
9. **Published Table 3 sanity checks pass:**
   * RMSE ≥ MAE in (a).
   * MSE ≥ MAE² in (b).
   * F1 is consistent with 2PR/(P+R) within rounding in (c): AlexNet gives 0.283 against 0.29 printed, MobileNet 0.648 against 0.65, ResNet50 0.112 against 0.11.

---

## 6. Links: reachable and unreachable

| Link (where cited) | Result |
|---|---|
| huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b (arXiv v1 Code Availability; also on the IBM blog) | **401** (API: "Invalid username or password."), so private or removed |
| huggingface.co/collections/nasa-impact/suryabench (GitHub README) | **401** |
| huggingface.co/datasets/nasa-impact/Surya-bench (AWS registry "Documentation") | **401** |
| huggingface.co/collections/nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c (arXiv v1 Data Records) | 200, redirects to `/collections/nasa-ibm-ai4science/suryabench`. 7 items: Surya-bench-solarwind, surya-bench-flare-forecasting, surya-bench-ar-segmentation, euv-spectra, surya-bench-coronal-extrapolation, ar_emergence, core-sdo; all `gated: False` |
| registry.opendata.aws/surya-bench and s3://nasa-surya-bench | 200; anonymous listing works |
| github.com/NASA-IMPACT/SuryaBench | Repository exists (`git ls-remote` works, tag V1.0) and the raw README is readable. The HTML page and REST API were **blocked** for this session, and an `add_repo` request was denied, so the code itself was **not reviewed** |
| HF datasets-server `/size` for core-sdo | Fails (`config-size` failed). The HF viewer is disabled for core-sdo |

---

## 7. What this means for "reproduce all results" on this machine (4 vCPU, 15 GB RAM, no GPU, about 9 GB disk)

* **Label and target files that fit:** all HF CSVs for DS1, DS2, DS4 and DS5, the DS3 index CSVs, and the DS2 `data.zip` (23 MB) are small.
* **Files that are too large:**
  * DS1 masks: 1.31 GB `data.tar.gz`; possibly test years only.
  * DS3 coefficients: a 36.18 GB archive, which does not fit in 9 GB.
  * DS6 spectra: train.csv is 2.98 GB, and every split CSV is above the 50 MB download limit used in this scouting step.
* **Every published baseline takes core SDO images as input** (4096×4096×13, about 570 MB per file). The test period alone is 2020–2024: 210,944 objects and 113,382,366,700,972 bytes on S3 (sum of the 2020–2024 rows in §2.2). Re-running any Table 3 baseline as described is **not feasible here**. In addition, the papers do not say which input subsampling or resolution the baselines used ("details omitted here" in the SI).
* **No trained baseline weights or prediction files** are linked from either paper version.

---

## 8. Evidence files (local)
* `suryabench_evidence/paper/`:
  * `arxiv_2508.14107v1_main.tex`: the source of the arXiv tables.
  * `arxiv_2508.14107v1_html.txt`
  * `scidata_s41597-026-06552-5.txt`
  * `scidata_table{1,2,3}.txt`: the published tables, extracted from the nature.com table pages.
  * `scidata_SI_MOESM1.txt`
  * `crossref_s41597-026-06552-5.json`
  * `SuryaBench_github_README.md`
* `suryabench_evidence/hf/`:
  * `api_*.json`, `rtree_*.json` (recursive trees with sizes), `card_*.md`, `dss_*.json` (datasets-server sizes), `coll_*.json` (collections).
  * `csv_analysis.txt` and `csv_stats.txt`: my counts.
* `suryabench_evidence/aws/count_<year>.json`: the S3 per-year object and byte counts.
* Downloaded CSVs (about 41 MB, not copied): `/tmp/claude-0/-home-user-NJIT-CCSC-Reproduction/cafab427-273b-538d-9e6f-5e9528085d79/scratchpad/sb/hfdata/`.
