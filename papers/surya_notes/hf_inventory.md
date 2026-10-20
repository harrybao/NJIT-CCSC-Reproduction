# Hugging Face artefact inventory: Surya and SuryaBench

Compiled 2026-10-15 (UTC). All web access used `curl` through the session proxy. I did not log in, use a token, install anything, or download any file over 50 MB. I downloaded only three small files: `config.yaml` (680 B), `infer_data/infer_index_surya_1_0.csv` (643 B) and `SuryaFM.pdf` (18,347,597 B).

Raw evidence is in `/home/user/work/surya_scout/research/sources/hf_inventory/`:
* `repos/<type>__<name>/` holds one folder per repo, each with:
  * `info.json` (`/api/<type>s/<id>?blobs=true`)
  * `tree_recursive.json` (built folder by folder from `/tree/main` and `/tree/main/<folder>`, following `Link: rel="next"` pagination)
  * `README.md` (the raw card)
  * `commits.json` (`/api/<type>s/<id>/commits/main`)
  * `fetch_log.txt` (each URL with its HTTP code)
* `models_*.json`, `datasets_*.json`, `spaces_*.json`, `colls_*.json`, `coll_*.json` and `search_*.json` are the listing and search results.
* `datasets_server/size_*.json` has the split row counts (from `https://datasets-server.huggingface.co/size?dataset=<id>`).
* `datacite/hf_*.json` has the DOI records.
* `downloads_small/SuryaFM.pdf` and `.txt` are the PDF shipped in the model repo, plus its extracted text.

Size units: **B** = bytes as reported by the HF tree API (the LFS size where the file is in LFS). **GB** = 10^9 B. **GiB** = 2^30 B.

---

## 0. TL;DR

* **13 Surya-related repos exist, all under `nasa-ibm-ai4science`**: 5 models and 8 datasets.
  * All are public, **not gated** (`gated: false`), not private and not disabled.
  * Licences: models **apache-2.0**; the 7 SuryaBench datasets **cc-by-4.0**; `Surya-1.0_validation_data` **mit**.
* **`nasa-impact` currently owns no Surya repo.** I checked its 18 models, 9 datasets, 1 public space and 5 collections (lists in §5).
  * Several old `nasa-impact/...` dataset URLs now **307-redirect** to `nasa-ibm-ai4science/...`, so the datasets were moved.
* **The SuryaBench collection linked from the IBM blog is unreachable.** `https://huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b` returns **HTTP 401** (HTML page: "404 – Hugging Face"), and so does `/api/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b`.
  * The live collection is `nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c`. It lists only **4 of the 7** SuryaBench datasets.
* **Only two cards report evaluation numbers:**
  * `Surya-1.0`: "15%", "IoU 0.768", "Dice 0.853".
  * `ar_segmentation_surya`: a 2-row IoU/Dice table, identical to Surya paper (arXiv 2508.14112v2) Table 3.
  * The `solar_flares_surya`, `solar_wind_surya` and `euv_spectra_surya` cards are empty (31-byte README with only `license: apache-2.0`).
  * No dataset card reports model scores.
* **What fits in about 9 GB of disk** (details in §4):
  * **Whole repo fits:** every model repo (1.78–1.82 GB each), `surya-bench-flare-forecasting` (8.8 MB), `Surya-bench-solarwind` (5.3 MB), `ar_emergence` (23.1 MB) and `surya-bench-ar-segmentation` (1.31 GB).
  * **Only part of the repo fits:**
    * `Surya-1.0_validation_data` is 15.43 GB in total.
    * `euv-spectra` is 12.06 GB.
    * `core-sdo` is 363.38 GB.
    * `surya-bench-coronal-extrapolation` is 36.18 GB, and its data is one 36.18 GB tarball.
  * **The upstream `download_data.sh` scripts need 13.67–14.97 GB each**, which exceeds 9 GB. The script for EUV is the largest, and its Python fallback needs 25.71 GB.
  * Measured free space on `/` at 18:20 UTC was **5.4 GB** (`df -h /`), probably because the main experiment is running.

---

## 1. How the list was built

| Query | Result |
|---|---|
| `GET /api/models?author=nasa-ibm-ai4science&limit=1000&full=true` | 9 models: 5 Surya + 4 Lunar FM |
| `GET /api/datasets?author=nasa-ibm-ai4science&limit=1000&full=true` | 13 datasets: 8 Surya/SuryaBench + 5 SomBench (lunar) |
| `GET /api/spaces?author=nasa-ibm-ai4science` | 1 public space (`nasa-ibm-lunar-foundation-model-generation-demo`, lunar). The org overview says `numSpaces: 2`, so 1 is not public |
| `GET /api/organizations/nasa-ibm-ai4science/overview` | `numModels 9, numDatasets 13, numSpaces 2, numUsers 13`. The listings above are therefore complete for models and datasets |
| `GET /api/models?author=nasa-impact…` | 18 models. None is Surya-related (all NLP / SDE / INDUS; list in §5) |
| `GET /api/datasets?author=nasa-impact…` | 9 datasets. None is Surya-related (list in §5) |
| `GET /api/organizations/nasa-impact/overview` | `numModels 18, numDatasets 9, numSpaces 2, numUsers 52`. 1 public space (`nasa-impact/geoguard-demo`) |
| `GET /api/collections?owner=nasa-ibm-ai4science` | 4 collections. **SuryaBench** (`suryabench-689276db205c56e81c10070c`, lastUpdated 2025-12-15T19:15:24.734Z) has 4 items, in position order: `Surya-bench-solarwind`, `surya-bench-flare-forecasting`, `surya-bench-ar-segmentation`, `euv-spectra`. **Surya** (`surya-6a957f7eeebf28997ebf3b70`, "Foundation model for heliophysics", lastUpdated 2026-08-31T13:20:09.346Z) has **0 items**. The other two are lunar |
| `GET /api/collections?owner=nasa-impact` | 5 collections (AKD Agents, INDUS-SDE-ST, SDE Downstreams, indus-sde, WxC Bench). **No SuryaBench** |
| Blog link `https://huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b` | **HTTP 401**; the HTML title is "404 – Hugging Face". The API form is also 401. Neither `/api/collections/68265ce306fc2470c121af7b` nor `/api/collections/nasa-ibm-ai4science/suryabench-68265ce306fc2470c121af7b` exists (404). The same dead link is printed in SuryaBench arXiv v1 §6 "Code Availability" |
| `https://huggingface.co/collections/nasa-ibm-ai4science/suryabench-689276db205c56e81c10070c` | 302 → `/collections/nasa-ibm-ai4science/suryabench` (200). That short URL is the one the `core-sdo` card and *Scientific Data* 13:712 print |
| Full-text search `api/{models,datasets,spaces}?search=surya / suryabench / surya-bench / heliophysics` | Found no extra repo in either org. Related repos outside the two orgs are listed in §5.3 |

Old `nasa-impact` repo URLs:

| Old URL | Result |
|---|---|
| `api/datasets/nasa-impact/surya-bench-flare-forecasting` | 307 → `nasa-ibm-ai4science/surya-bench-flare-forecasting` |
| `api/datasets/nasa-impact/Surya-bench-solarwind` | 307 → `nasa-ibm-ai4science/Surya-bench-solarwind` |
| `api/datasets/nasa-impact/ar_emergence` | 307 → `nasa-ibm-ai4science/ar_emergence` |
| `api/datasets/nasa-impact/surya-bench-ar-segmentation` | 307 → `nasa-ibm-ai4science/surya-bench-ar-segmentation` |
| `api/datasets/nasa-impact/surya-bench-coronal-extrapolation` | 307 → `nasa-ibm-ai4science/surya-bench-coronal-extrapolation` |
| `datasets/nasa-impact/surya-bench-euv-spectra` (linked in upstream `downstream_examples/euv_spectra_prediction/README.md:167`) | 307 → `datasets/nasa-ibm-ai4science/euv-spectra` |
| `nasa-impact/core-sdo`, `nasa-impact/euv-spectra`, `nasa-impact/Surya-1.0_validation_data`, `nasa-impact/Surya-bench`, `nasa-impact/SuryaBench`, `nasa-impact/surya-bench-core-sdo`, `nasa-impact/surya-bench-sdo`, models `nasa-impact/Surya-1.0`, `nasa-impact/Surya`, `nasa-impact/surya`, `nasa-impact/solar_flares_surya` | 401 (do not exist, or private) |

The Surya paper (arXiv 2508.14112v2, "Code and Data Availability") says: "The model and datasets are publicly available on Huggingface: https://huggingface.co/nasa-ibm-ai4science". That link works.

---

## 2. Master table (state at fetch time, 2026-10-15)

`downloads` is the API's rolling 30-day download count. "Current files" is the sum of the files at `main`. `usedStorage` includes git history.

| # | id | type | created | last modified | sha (main) | licence | gated | files | current files | usedStorage | fits in ~9 GB? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 | `nasa-ibm-ai4science/Surya-1.0` | model | 2025-08-13T15:22:00Z | 2025-08-20T00:57:57Z | `5cc4b5386d5f78fda3896b1389589d4e173bf212` | apache-2.0 | no | 7 | 1,820,449,157 B (1.820 GB) | 1,821,260,355 | **Yes** |
| M2 | `nasa-ibm-ai4science/solar_flares_surya` | model | 2025-08-18T21:51:04Z | 2025-08-18T22:22:30Z | `ec7c42ab22fbf126c4bfffadb420f6e45a669832` | apache-2.0 | no | 3 | 1,794,024,134 B (1.794 GB) | 1,794,022,584 | **Yes** |
| M3 | `nasa-ibm-ai4science/solar_wind_surya` | model | 2025-08-18T22:32:45Z | 2025-08-18T22:37:24Z | `6ac4aa79a96f8785a1ab300842d13727e52b58a5` | apache-2.0 | no | 3 | 1,805,657,108 B (1.806 GB) | 1,805,655,558 | **Yes** |
| M4 | `nasa-ibm-ai4science/euv_spectra_surya` | model | 2025-08-18T22:41:10Z | 2025-08-26T15:42:52Z | `6ed3bc13eeece973653422cc83e583aabd31dae0` | apache-2.0 | no | 3 | 1,779,724,270 B (1.780 GB) | 3,556,704,296 | **Yes** |
| M5 | `nasa-ibm-ai4science/ar_segmentation_surya` | model | 2025-08-18T22:54:32Z | 2025-08-20T20:20:32Z | `73a6de53c4bea812f7bc2ac29b9351ae2f2fe5fe` | apache-2.0 | no | 4 | 1,784,514,037 B (1.785 GB) | 1,784,510,844 | **Yes** |
| D1 | `nasa-ibm-ai4science/Surya-1.0_validation_data` | dataset | 2025-08-13T17:20:42Z | 2025-08-17T17:22:41Z | `3b6710423d8ca10889e7d9f154f587b91266f6be` | mit | no | 25 | 15,427,169,811 B (15.427 GB) | 15,427,165,708 | **Partly**: one test initialisation needs 4 files, about 2.68 GB |
| D2 | `nasa-ibm-ai4science/core-sdo` | dataset | 2025-05-16T01:08:37Z | 2025-12-16T21:21:41Z | `9c10abe52c1f206d1b66715745229e29ef543cc6` | cc-by-4.0 | no | 60 | 363,376,705,743 B (363.377 GB) | 363,375,594,449 | **Partly**: CSVs and YAML (19.6 MB); single `infer_data/*.nc` files (0.66–0.67 GB each) |
| D3 | `nasa-ibm-ai4science/surya-bench-flare-forecasting` | dataset | 2025-05-15T21:47:20Z | 2025-12-16T21:25:30Z | `bf474bc1e7752bd529e9707650f9310db8693763` | cc-by-4.0 | no | 7 | 8,760,535 B (8.8 MB) | 2,591,745 | **Yes** |
| D4 | `nasa-ibm-ai4science/surya-bench-ar-segmentation` | dataset | 2025-05-15T22:46:38Z | 2025-12-16T21:48:14Z | `615a7df4ac3498c97d0d039e7b0654774ad86602` | cc-by-4.0 | no | 7 | 1,313,355,815 B (1.313 GB) | 5,971,018,868 | **Yes** |
| D5 | `nasa-ibm-ai4science/Surya-bench-solarwind` | dataset | 2025-05-15T21:28:28Z | 2025-12-16T21:30:54Z | `20547a25cbd3aaa359e9cda18be913b94c225e7b` | cc-by-4.0 | no | 7 | 5,332,522 B (5.3 MB) | 2,039,220 | **Yes** |
| D6 | `nasa-ibm-ai4science/euv-spectra` | dataset | 2025-05-16T07:50:41Z | 2025-12-16T21:59:32Z | `aafccf9c1364bd39eaf49e058e39f24f70719657` | cc-by-4.0 | no | 15 | 12,062,209,078 B (12.062 GB) | 14,497,894,075 | **Partly**: the split CSVs are 3.47 GB |
| D7 | `nasa-ibm-ai4science/ar_emergence` | dataset | 2025-05-16T00:20:25Z | 2025-12-16T21:51:53Z | `48970a12c290b515912baadba87fa74a49e3c680` | cc-by-4.0 | no | 6 | 23,102,338 B (23.1 MB) | 3,809,756,040 | **Yes** |
| D8 | `nasa-ibm-ai4science/surya-bench-coronal-extrapolation` | dataset | 2025-05-15T20:25:45Z | 2025-12-16T21:28:30Z | `622892941688f21e69bbad37598b33787083cdbd` | cc-by-4.0 | no | 9 | 36,181,107,009 B (36.181 GB) | 212,586,020,461 | **No** for the data (one 36.18 GB `data.tar.gz`). Only the CSVs and the 2.3 MB examples tar fit |

Other metadata:
* Downloads (30 days): M1 238, M2–M5 0. D1 795, D2 1788, D3 260, D4 202, D5 407, D6 159, D7 127, D8 120.
* Likes: M1 108, M5 10, D2 9, D5 3.
* `private: false` and `disabled: false` for all 13.

**DOIs.** Each of the 7 SuryaBench datasets has a Hugging Face DOI. The DataCite `version` field equals the current HF sha prefix, so **the DOIs pin exactly the current `main`**:

| Dataset | DOI | DataCite version |
|---|---|---|
| D2 core-sdo | 10.57967/hf/7273 | `9c10abe` |
| D3 flare | 10.57967/hf/7274 | `bf474bc` |
| D8 coronal | 10.57967/hf/7275 | `6228929` |
| D5 solarwind | 10.57967/hf/7276 | `20547a2` |
| D4 ar-seg | 10.57967/hf/7278 | `615a7df` |
| D7 ar_emergence | 10.57967/hf/7279 | `48970a1` |
| D6 euv | 10.57967/hf/7280 | `aafccf9` |

All 7 DOIs were created on 2025-12-16, per `datacite/hf_72xx.json`. The same revisions are cited as refs 58–64 of *Scientific Data* 13:712 (2026). I checked the saved article text, `suryabench_evidence/paper/scidata_s41597-026-06552-5.txt` lines 379–385; I did not re-fetch the article. **The models and D1 have no DOI.**

---

## 3. Per-artefact details

### 3.1 Models

#### M1 `nasa-ibm-ai4science/Surya-1.0` (sha `5cc4b538…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 1,678 | git |
| `README.md` | 4,206 | git |
| `SuryaFM.pdf` | 18,347,597 | LFS (sha256 `e03932b772c3c902bb0a117105c4daf955ea1742a043091d3f19e270d71535d8`, which I verified after download) |
| `config.yaml` | 680 | git |
| `scalers.yaml` | 3,835 | git |
| `surya.366m.v1.pt` | 1,800,384,309 | LFS |
| `surya_arch.png` | 1,706,852 | LFS |
| **Total** | **1,820,449,157 (1.820 GB / 1.695 GiB)** | |

* **What the card says it is for** (README lines 16–17, 45): "Surya is a 366M-parameter transformer model pretrained on **9 years (≈218 TB)** of multi-instrument data from NASA's [Solar Dynamics Observatory (SDO)]… including 8 Atmospheric Imaging Assembly (AIA) channels and 5 Helioseismic and Magnetic Imager (HMI) products." The variant is "**`surya.366m.v1`** — pretrained on 9 years of SDO AIA/HMI data with forecasting objective + rollout tuning."
* **Contents**, as listed on the card: `surya.366m.v1.pt` ("Model weights"), `config.yaml` ("Configuration file") and `scalers.yaml` ("Preprocessing & normalization parameters").
* **`config.yaml`** (downloaded, 680 B): `img_size: 4096`, `patch_size: 16`, `embed_dim: 1280`, `depth: 10`, `n_spectral_blocks: 2`, `num_heads: 16`, `time_delta_input_minutes: [-60, 0]`, `time_delta_target_minutes: +60`, 13 `sdo_channels`.
* **Evaluation numbers on the card**, copied verbatim (README lines 36–39):
  * "🌞 **Solar flare forecasting** — surpasses existing benchmarks by **15%** in preliminary tests, with 24h binary classification (M/X-class flares)."
  * "🌬 **Solar wind speed prediction** — downstream fine-tuning achieves strong performance compared to physics-based models."
  * "☀️ **Active region segmentation** — outperforms baseline UNet with **IoU 0.768** and **Dice 0.853**."
  * "🔭 **EUV spectral forecasting** — accurate prediction of solar spectra."
* **Which paper results it supports.** This is the backbone for every Surya result:
  * Zero-shot forecasting, Surya paper §2.5.1 / Fig. 5 / App. B.1. The Table 9 caption says "The 5 hour ahead tuned version (rightmost column) corresponds to Surya as released".
  * The zero-shot flare and blurring figures (§2.5.2–2.5.3).
  * The starting point for all fine-tunes in Tables 3–7.
  * The repo test `tests/test_surya.py` (`REFERENCE_LOSS`, 8 values, with D1).
  * It does **not** support Table 8 (ablations: separate 16-GPU / 10,000-step runs) or the intermediate Table 9 columns (those checkpoints were not released).
* **`SuryaFM.pdf`** is a 36-page copy of the Surya paper (pdfinfo CreationDate "Wed Aug 20 00:51:40 2025 UTC").
  * Its Tables 3, 4, 5, 6 and 7 print the same numbers as arXiv v2 HTML (checked line by line, e.g. Table 3 "Surya 4.1 M 0.768 0.853" and Table 4 "Surya 0.436 0.522 0.561").
  * Its author list includes Anne Spalding, as arXiv v2 does.
  * The string "15%" appears in neither the PDF nor the v2 HTML.

#### M2 `nasa-ibm-ai4science/solar_flares_surya` (sha `ec7c42ab…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 1,519 | git |
| `README.md` | 31 | git |
| `solar_flare_weights.pth` | 1,794,022,584 | LFS |
| **Total** | **1,794,024,134 (1.794 GB / 1.671 GiB)** | |

* **Card:** the whole README is front-matter `license: apache-2.0`. It has **no description and no numbers.**
* **Purpose**, from the upstream repo rather than the card: `downstream_examples/solar_flare_forcasting/{download_data.sh, infer.py:252}` downloads `*.pth` from this repo. `infer.py` loads it with `model.load_state_dict(model_state, strict=True)` (line 53), so the file holds the full fine-tuned network, not a small adapter.
* **Supports:** Surya paper §2.6.3, **Table 4** (Surya row: TSS 0.436, HSS 0.522, F1 0.561). The card says none of this.
* Two commits: initial commit, then "Upload solar_flare_weights.pth with huggingface_hub" (2025-08-18T22:22:30Z).

#### M3 `nasa-ibm-ai4science/solar_wind_surya` (sha `6ac4aa79…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 1,519 | git |
| `README.md` | 31 | git |
| `solar_wind_weights.pth` | 1,805,655,558 | LFS |
| **Total** | **1,805,657,108 (1.806 GB / 1.682 GiB)** | |

* **Card:** `license: apache-2.0` only. **No description and no numbers.**
* **Purpose** (upstream): `downstream_examples/solar_wind_forcasting/{download_data.sh:42, infer.py:294}`.
* **Supports:** Surya paper §2.6.5, **Tables 6 and 7** (Surya RMSE 75.92 km s⁻¹, MAE 58.06, validation MSE 5698.62).
* Two commits, the last on 2025-08-18T22:37:24Z.

#### M4 `nasa-ibm-ai4science/euv_spectra_surya` (sha `6ed3bc13…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 1,519 | git |
| `README.md` | 31 | git |
| `euv_spectra_weights.pth` | 1,779,722,720 | LFS |
| **Total** | **1,779,724,270 (1.780 GB / 1.657 GiB)** | |

* **Card:** `license: apache-2.0` only. **No description and no numbers.**
* **Purpose** (upstream): `downstream_examples/euv_spectra_prediction/{download_data.sh:43, infer.py:348}`.
* **Supports:** Surya paper §2.6.4, **Table 5** (Surya MSE 0.0001260741, MAE 0.0045114677, MAPE 1.4792510271).
* **The weights were uploaded twice**: 2025-08-18T22:41:44Z (commit `253bf1091d`) and 2025-08-26T15:42:52Z (`6ed3bc13ee`). This explains why `usedStorage` (3,556,704,296) is about twice the current size. The second upload is **after** arXiv v2 (2025-08-21), so it is unknown which upload produced Table 5.

#### M5 `nasa-ibm-ai4science/ar_segmentation_surya` (sha `73a6de53…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 1,566 | git |
| `README.md` | 1,627 | git |
| `ar_seg.png` | 497,492 | LFS |
| `ar_segmentation_weights.pth` | 1,784,013,352 | LFS |
| **Total** | **1,784,514,037 (1.785 GB / 1.662 GiB)** | |

* **What the card says it is for** (lines 11–12): "This repository hosts **fine-tuned weights of Surya** – a heliophysics foundation model – for the task of **solar Active Region (AR) segmentation**." The card's base model is `nasa-ibm-ai4science/Surya-1.0`. For usage it says "Follow the instructions at Surya/downstream_examples/ar_segmentation".
* **Evaluation numbers on the card** (lines 17–22, verbatim): "We benchmarked Surya against a standard UNet baseline on the ARPIL dataset."

  | Model  | Params | IoU   | Dice Coeff |
  |--------|--------|-------|------------|
  | UNet   | 9.2 M  | 0.688 | 0.801      |
  | **Surya (LoRA)** | **4.1 M**  | **0.768** | **0.853**      |

  The card's example image caption reads "Input SDO/HMI data (Date: 2014-02-01, Time: 08:12)".
* **Supports:** Surya paper §2.6.2, **Table 3**. The values are identical. The paper labels the row "Surya", and the card labels it "Surya (LoRA)".
* **Note:** "4.1 M" is the trainable parameter count, but the checkpoint file is 1.78 GB. `ar_segmentation/infer.py:116,132` loads it with `strict=True` into the full model.

### 3.2 Datasets

#### D1 `nasa-ibm-ai4science/Surya-1.0_validation_data` (sha `3b671042…`)

| File | Bytes |
|---|---|
| `.gitattributes` | 3,680 |
| `README.md` | 423 |
| `20140107_1500.nc` | 670,597,471 |
| `20140107_1512.nc` | 670,754,018 |
| `20140107_1524.nc` | 670,677,114 |
| `20140107_1536.nc` | 670,695,444 |
| `20140107_1548.nc` | 670,698,365 |
| `20140107_1600.nc` | 670,337,550 |
| `20140107_1612.nc` | 670,770,534 |
| `20140107_1624.nc` | 670,596,895 |
| `20140107_1636.nc` | 670,692,233 |
| `20140107_1648.nc` | 670,640,155 |
| `20140107_1700.nc` | 670,740,129 |
| `20140107_1712.nc` | 670,855,932 |
| `20140107_1724.nc` | 670,646,002 |
| `20140107_1736.nc` | 670,480,758 |
| `20140107_1748.nc` | 670,678,121 |
| `20140107_1800.nc` | 670,941,435 |
| `20140107_1812.nc` | 670,647,691 |
| `20140107_1824.nc` | 671,046,501 |
| `20140107_1836.nc` | 670,897,034 |
| `20140107_1848.nc` | 670,865,203 |
| `20140107_1900.nc` | 671,028,891 |
| `20140107_1912.nc` | 670,946,002 |
| `20140107_1924.nc` | 670,932,230 |
| **Total** | **15,427,169,811 (15.427 GB / 14.368 GiB)**. The 23 `.nc` files are all LFS and sum to 15,427,165,708 |

* **Card** (verbatim): "This dataset comprises imagery from NASA's Solar Dynamics Observatory (SDO). The data can and should be used to validate a local installation of the Surya Foundation Model for Heliophysics. The data is compressed; you should use the hdf5plugin to read it directly."
* **No numbers on the card.** The datasets-server has no viewer for it: "No (supported) data files found".
* **Supports:** the repo self-test `tests/test_surya.py`.
  * It downloads `allow_patterns="20140107_1[5-9]??.nc"`, which matches all 23 files (15,427,165,708 B).
  * It compares against the hard-coded `REFERENCE_LOSS` (8 values, lines 19–30) with tolerance `1.0e-4` (line 290).
  * This is not a paper table. The earlier notes (`surya_paper_results.md` §4) link it to the paper's Fig. 14 setting.
* One initialisation with 2 rollout steps needs 4 files 1 h apart:
  * 15:00, 16:00, 17:00 and 18:00 come to 2,682,616,585 B.
  * 16:24, 17:24, 18:24 and 19:24 come to 2,683,221,628 B.

#### D2 `nasa-ibm-ai4science/core-sdo` (sha `9c10abe5…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 3,483 | git |
| `README.md` | 3,966 | git |
| `aia_hmi.png` | 4,936,462 | LFS |
| `dataview.png` | 18,592,147 | LFS |
| `scalers.yaml` | 3,835 | git |
| `test.csv` | 13,508 | git |
| `train.csv` | 110,390 | git |
| `train_index_surya_1_0.csv` | 18,557,847 | LFS |
| `val.csv` | 13,476 | git |
| `valid_index_surya_1_0.csv` | 961,993 | git |
| `infer_data/20110120_0100.nc` | 672,958,156 | LFS |
| `infer_data/20110120_0200.nc` | 672,890,851 | LFS |
| `infer_data/20110120_0300.nc` | 672,994,824 | LFS |
| `infer_data/20120115_0048.nc` | 671,908,873 | LFS |
| `infer_data/20120115_0100.nc` | 672,074,493 | LFS |
| `infer_data/20120115_0148.nc` | 671,793,509 | LFS |
| `infer_data/20120115_0200.nc` | 671,786,603 | LFS |
| `infer_data/20120115_0248.nc` | 671,769,397 | LFS |
| `infer_data/20120115_0300.nc` | 671,771,088 | LFS |
| `infer_data/20130128_1812.nc` | 670,589,974 | LFS |
| `infer_data/20130128_1912.nc` | 670,515,243 | LFS |
| `infer_data/20130128_2012.nc` | 670,510,815 | LFS |
| `infer_data/20190123_0200.nc` | 663,160,144 | LFS |
| `infer_data/20190123_0300.nc` | 663,408,883 | LFS |
| `infer_data/20190123_0400.nc` | 662,989,220 | LFS |
| `infer_data/infer_index_surya_1_0.csv` | 643 | git (downloaded; 15 rows, listed in the paragraph after this table) |
| `test_splitted_tars/test.tar.part_aa` … `part_ap` (16 files) | 10,737,418,240 each | LFS |
| `test_splitted_tars/test.tar.part_aq` | 4,031,109,120 | LFS |
| `validate_splitted_tars/val.tar.part_aa` … `part_ap` (16 files) | 10,737,418,240 each | LFS |
| `validate_splitted_tars/val.tar.part_aq` | 5,653,893,120 | LFS |
| **Total (60 files)** | **363,376,705,743 (363.377 GB / 338.421 GiB)** | |

Subtotals:
* `test_splitted_tars/` 175,829,800,960 B (16 × 10,737,418,240 + 4,031,109,120).
* `validate_splitted_tars/` 177,452,584,960 B.
* `infer_data/*.nc` 10,051,122,073 B.
* Index CSVs + `scalers.yaml` 19,524,318 B.

The `infer_index_surya_1_0.csv` rows form five sets of 3 timestamps one hour apart: 2011-01-20 01:00/02:00/03:00; 2012-01-15 01:00/02:00/03:00; 2012-01-15 00:48/01:48/02:48; 2013-01-28 18:12/19:12/20:12; 2019-01-23 02:00/03:00/04:00.

* **What the card says it is for:** "This dataset provides machine learning (ML)-ready solar data curated from NASA's Solar Dynamics Observatory (SDO), covering observations from **May 13, 2010, to Dec 31, 2024**." It goes on: "The dataset is designed to facilitate large-scale learning applications in heliophysics, such as space weather forecasting, unsupervised representation learning, and scientific foundation model development."
  * Under "Dataset Download Instructions" it says: "To download the data please visit https://registry.opendata.aws/surya-bench/". It gives the ARN `arn:aws:s3:::nasa-surya-bench` and the command `aws s3 ls --no-sign-request s3://nasa-surya-bench/`.
  * Card table: "Data Format | netCDF (`.nc`), float32"; "Temporal Granularity | 12 minutes"; "Data Shape | `[13, 4096, 4096]` per file"; "Size per File | ~570 MB"; "Total Size | ~360TB".
* **No evaluation numbers.**
* **Supports:** pretraining input for Surya (paper §2.1, Table 1) and the SuryaBench "Core SDO Dataset" (SuryaBench §2.1).
  * The HF copy holds only the **validation and test** tar parts, index CSVs and 15 sample files. The full training data is on AWS (see `suryabench_paper_results.md` for the S3 listing: 349.47 TB).
  * Every upstream `download_data.sh` pulls `*_index_surya_1_0.csv`, `infer_data/*` and `scalers.yaml` from here, 10,070,646,391 B in total.
* The datasets-server size endpoint fails for it: `"failed": [{"kind": "config-size", …}]`.

#### D3 `nasa-ibm-ai4science/surya-bench-flare-forecasting` (sha `bf474bc1…`)

| File | Bytes |
|---|---|
| `.gitattributes` | 2,461 |
| `README.md` | 3,284 |
| `data.csv` | 4,377,302 |
| `leaky_validation.csv` | 206,107 |
| `test.csv` | 1,531,173 |
| `train.csv` | 2,517,593 |
| `validation.csv` | 122,615 |
| **Total** | **8,760,535 (8.8 MB)**. All files are plain git (no LFS) |

* **Card:** "This dataset provides labels for solar flare forecasting derived from NOAA GOES flare events from May 2010 to December 2024. Labels are constructed using a 24h rolling prediction window sampled at an hourly cadence." The labels are "**`label_max`**: 1 if the maximum flare intensity in the window is ≥ M1.0" and "**`label_cum`**: 1 if the cumulative flare intensity in the window is ≥ 10".
  * Splits: "`train.csv`: Instances from Feb 15 to Dec 31 in each year between 2010–2019"; "`validation.csv`: Instances from Jan 15–31 of each year between 2010–2019"; "`test.csv`: All instances from each year between 2020–2024"; "`leaky_validation.csv`: Instances from Jan 1–14 and Feb 1-14 of each year between 2010–2019".
  * Card table: "Data Size | Total 128,328 instances"; "Total File Size | ~4.4MB".
  * Authors: Jinsu Hong, Kang Yang, Berkay Aydin (GSU).
* **No model scores on the card.**
* The datasets-server reports 128,328 rows: train 74,760, validation 3,672, test 43,848, leaky_validation 6,048. This matches the card.
* **Supports:** SuryaBench DS4 (§2.2.4; arXiv v1 Table 3(b) and Table 5; *Sci Data* Table 3(c)) and Surya paper §2.6.3 Table 4.

#### D4 `nasa-ibm-ai4science/surya-bench-ar-segmentation` (sha `615a7df4…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 2,506 | git |
| `README.md` | 2,419 | git |
| `data.tar.gz` | 1,306,605,651 | LFS |
| `leaky_validation.csv` | 318,065 | git |
| `test.csv` | 2,311,618 | git |
| `train.csv` | 3,922,895 | git |
| `validation.csv` | 192,661 | git |
| **Total** | **1,313,355,815 (1.313 GB / 1.223 GiB)** | |

* **Card:** "This dataset provides hourly binary segmentation maps (4096×4096 resolution) derived from Solar Dynamics Observatory (SDO) / Helioseismic and Magnetic Imager (HMI) line-of-sight magnetograms. The maps highlight regions containing Active Regions (ARs) and Polarity Inversion Lines (PILs)."
  * "Each .h5 file contains two keys: `intersection` … `union_with_intersect`".
  * Card table: "Total Instances | Total 128,352 instances"; "Size per File | ~ 53kB"; "Total Dataset Size | ~ 1.2GB".
  * `pretty_name: ARPIL`.
* **No model scores on the card.**
* The datasets-server reports **128,328** rows (74,760 / 3,672 / 43,848 / 6,048), not the card's 128,352.
* **Supports:** SuryaBench DS1 (§2.2.1; *Sci Data* Table 3(d)) and Surya paper §2.6.2 Table 3 (via M5).

#### D5 `nasa-ibm-ai4science/Surya-bench-solarwind` (sha `20547a25…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 2,461 | git |
| `README.md` | 2,755 | git |
| `leaky_validation.csv` | 256,013 | git |
| `solar_wind_data.png` | 120,652 | LFS |
| `test.csv` | 1,746,629 | git |
| `train.csv` | 3,054,147 | git |
| `validation.csv` | 149,865 | git |
| **Total** | **5,332,522 (5.3 MB)** | |

* **Card:** "This dataset provides hourly solar wind plasma and interplanetary magnetic field (IMF) parameters at L1, derived from NASA's OMNI dataset. The primary forecasting target is the solar wind speed (`V`)…"
  * Card table: "Number of Time Steps | 119,225"; "Cadence | 1 hour".
  * Author: Vishal Upendran.
* **No model scores on the card.**
* The datasets-server reports 119,225 rows (69,957 / 3,429 / 39,975 / 5,864). This matches the card.
* **Supports:** SuryaBench DS5 (§2.2.5; arXiv v1 Table 3(a); *Sci Data* Table 3(a)) and Surya paper §2.6.5 Tables 6–7 (via M3).

#### D6 `nasa-ibm-ai4science/euv-spectra` (sha `aafccf9c…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 3,168 | git |
| `README.md` | 4,606 | git |
| `euv_wavelengths.csv` | 25,599 | git |
| `leaky_validation.csv` | 302,924,931 | LFS |
| `test.csv` | 110,815,206 | LFS |
| `train.csv` | 2,984,963,110 | LFS |
| `validation.csv` | 66,789,263 | LFS |
| `archive/AIA_EVE_dataset_combined.nc` | 1,018,976,075 | LFS |
| `archive/AIA_EVE_temporally_stratified.nc` | 933,907,814 | LFS |
| `archive/test_eve_spectra.csv` | 516,804,580 | LFS |
| `archive/test_eve_spectra_stratified.csv` | 89,377,631 | LFS |
| `archive/train_eve_spectra.csv` | 2,419,364,430 | LFS |
| `archive/train_eve_spectra_stratified.csv` | 3,027,846,821 | LFS |
| `archive/val_eve_spectra.csv` | 530,475,194 | LFS |
| `archive/val_eve_spectra_stratified.csv` | 59,930,650 | LFS |
| **Total** | **12,062,209,078 (12.062 GB / 11.234 GiB)** | |

Subtotals: the top-level split CSVs are 3,465,492,510 B, and `archive/` is 8,596,683,195 B.

* **Card:** "This dataset provides time-aligned Extreme Ultraviolet (EUV) irradiance spectra from NASA's SDO/EVE (Extreme Ultraviolet Variability Experiment) instrument." It says the dataset "includes 1,343-dimensional EUV spectrum spanning ~6.5–33.3 nm (with a 0.02 nm spectral resolution and 12-minute cadence). The dataset covers May 2010 to May 2014".
  * Splits: "`validation.csv`: Instances from Jan 15–31 from years 2012 and 2013"; "`test.csv`: Instances from Jan 15–31 from years 2011 and 2014".
  * Card table: "Data Size | Total 189,344 instances"; "Total File Size | ~3.2GB".
  * Author: Shah Mohammad Bahauddin (LASP). The card's "Citation" section cites Woods et al. 2012 (EVE) and Woodraska & Eparvier 2024.
* **No model scores on the card.**
* The datasets-server reports 189,344 rows (163,097 / 3,650 / 6,051 / 16,546). This matches the card.
* **Supports:** SuryaBench DS6 (§2.2.6; arXiv v1 Table 4(a); *Sci Data* Table 3(b)) and Surya paper §2.6.4 Table 5 (via M4).
* The card does not describe `archive/` at all.
* The upstream script `euv_spectra_prediction/download_data.sh:41` uses `--include "*_temporally_stratified.nc"`. That pattern now matches only `archive/AIA_EVE_temporally_stratified.nc`, moved there by commit `d3787e07f1` "Move .nc files into archive folder" on 2025-10-01. The script's Python fallback (line 64) downloads the **whole** repo.

#### D7 `nasa-ibm-ai4science/ar_emergence` (sha `48970a12…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 2,834 | git |
| `README.md` | 4,179 | git |
| `data.zip` | 23,085,581 | LFS |
| `test.csv` | 2,168 | LFS |
| `train.csv` | 6,056 | LFS |
| `validation.csv` | 1,520 | git |
| **Total** | **23,102,338 (23.1 MB)** | |

* **Card:** "The **Active Region Emergence Dataset** is designed to support research on the early detection of solar Active Regions (ARs) and the development of predictive models for space weather." It covers "**59 large ARs** that emerged on the visible solar disk between **2010 and 2023**".
  * Card table: "Data Size | Total 59 instances"; "Total File Size | ~23.1 MB"; "Data Shape | (6, 240, 9, 9) per instance".
  * Reference [1] is Kasapis et al., ApJS, 10.3847/1538-4365/adfbe2 (2025).
* **No model scores on the card.**
* The datasets-server reports 59 rows (37 / 9 / 13). This matches the card's "training split (37 ARs)", "validation split (9 ARs)" and "test split (13 ARs)".
* **Supports:** SuryaBench DS2 (§2.2.2; arXiv v1 Table 4(b)).
  * The Surya paper has no AR-emergence result.
  * *Sci Data* has no DS2 baseline, per `suryabench_paper_results.md` §4: "DS2 and DS3 have no baselines".

#### D8 `nasa-ibm-ai4science/surya-bench-coronal-extrapolation` (sha `62289294…`)

| File | Bytes | Storage |
|---|---|---|
| `.gitattributes` | 2,461 | git |
| `CoronalFieldExtrapolation_examples.tar` | 2,293,760 | LFS |
| `README.md` | 3,801 | git |
| `data.tar.gz` | 36,176,131,632 | LFS |
| `leaky_validation.csv` | 113,974 | git |
| `test.csv` | 825,874 | git |
| `train.csv` | 1,408,050 | git |
| `validation.csv` | 69,226 | git |
| `wsa_fields.png` | 258,231 | LFS |
| **Total** | **36,181,107,009 (36.181 GB / 33.696 GiB)** | |

* **Card:** "This dataset contains spherical harmonic coefficients of the coronal magnetic potential generated by emulating the physics-based **ADAPT-WSA PFSS** (Potential Field Source Surface) code, driven by **SDO/HMI** solar magnetogram observations."
  * Card table: "Data Size | 5,347 instances * 12 realizations"; "Total File Size | ~36.2 GB (compressed)"; "Size per file | ~763 KB".
  * Contact: Daniel da Silva.
* **No model scores on the card.**
* The datasets-server reports 5,347 rows (3,115 / 153 / 1,827 / 252). This matches the card.
* **Supports:** SuryaBench DS3 (§2.2.3) only. **No baseline or Surya result exists in any of the papers checked.**
* The data comes as one 36.18 GB tarball, so it cannot be fetched in part over plain HTTP without range tricks. Only `CoronalFieldExtrapolation_examples.tar` (2.3 MB) and the CSVs are small.

---

## 4. Disk-fit analysis (budget about 9 GB; 5.4 GB free at 18:20 UTC)

| Use case | Files | Bytes | Fits? |
|---|---|---|---|
| Surya-1.0 backbone (`config.yaml`, `scalers.yaml`, `surya.366m.v1.pt`; the `test_surya.py` patterns) | M1 | 1,800,388,824 (1.80 GB) | Yes |
| `test_surya.py` as written (all 23 validation files + weights) | M1 + D1 | 17,227,554,532 (17.23 GB) | **No** |
| One `test_surya.py` initialisation (4 files) + weights | M1 + 4 of D1 | ≈4.48 GB | Yes |
| Upstream `solar_flare_forcasting/download_data.sh` | D3 + M2 `*.pth` + M1 `.pt` + D2 subset | 13,673,813,819 (13.67 GB) | **No**; without `infer_data` it is 3.62 GB |
| Upstream `solar_wind_forcasting/download_data.sh` | D5 + M3 + M1 + D2 subset | 13,682,018,780 (13.68 GB) | **No**; 3.63 GB without `infer_data` |
| Upstream `euv_spectra_prediction/download_data.sh` (hf CLI path) | D6 `*_temporally_stratified.nc` + M4 + M1 + D2 subset | 14,584,661,234 (14.58 GB) | **No** |
| Same script, Python fallback (full D6 snapshot) | D6 + M4 + M1 + D2 subset | 25,712,962,498 (25.71 GB) | **No** |
| Upstream `ar_segmentation/download_data.sh` | D4 + M5 + M1 + D2 subset | 14,968,399,867 (14.97 GB) | **No**; 4.92 GB without `infer_data` |
| Backbone + one head + one `infer_data` set of 3 timestamps | M1 + M2…M5 + 3 `.nc` | ≈1.80 + ≈1.79 + ≈2.02 = ≈5.6 GB | Yes, one task at a time |
| All benchmark label CSVs (D3, D5, D7, the CSVs of D4/D8, the top-level CSVs of D6) | | 3,511,850,268 B (3.51 GB): D6 top-level CSVs 3,465,492,510 B plus 46,357,758 B for all the rest | Yes |
| Paper-table reproduction (Surya Tables 3–7, full test periods) | Needs the SDO input cubes for the whole test period: the `core-sdo` test tars (175.83 GB), or S3 | ≥ 175 GB | **No** |

The D2 subset is `*_index_surya_1_0.csv`, `infer_data/*` and `scalers.yaml`, which is 10,070,646,391 B. Of that, `infer_data/*.nc` is 10,051,122,073 B.

Conclusion: on this machine the HF artefacts allow
1. downloading every set of weights (one or two at a time),
2. a grouped or partial run of the repo validation test, and
3. sample inference for each downstream head on single `infer_data` sets of 3 timestamps.

They do not allow re-computing any paper table end to end. The upstream download scripts must be replaced with `--include` patterns that fetch fewer files.

---

## 5. Repos excluded from the inventory

### 5.1 `nasa-ibm-ai4science`, not Surya
* Models (lunar): `NASA-IBM-Lunar-Foundation-Model`, `Crater-Detection-NASA-IBM-Lunar-Foundation-Model`, `IMP-Segmentation-NASA-IBM-Lunar-Foundation-Model`, `Ice-Prospectivity-NASA-IBM-Lunar-Foundation-Model`. All four were last modified in September 2026.
* Datasets (lunar SomBench): `Sombench-NAC-Crater-Detection`, `Sombench-WAC-Crater-Detection`, `Sombench-IMP-Segmentation`, `Sombench-Ice-Prospectivity-Regression`, `Sombench-pretraining-data`.
* Space: `nasa-ibm-lunar-foundation-model-generation-demo`.

### 5.2 `nasa-impact`, none Surya-related
* Models (18): `bert-e-base-mlm`, `nasa-smd-ibm-v0.1`, `nasa-smd-ibm-st`, `nasa-smd-ibm-st-v2`, `document-classifier`, `nasa-smd-ibm-ranker`, `nasa-ibm-st.38m`, `nasa-smd-ibm-distil-v0.1`, `science-keyword-classification`, `division-classifier`, `indus-sde-v0.1`, `tdamm-classification`, `indus-sde-v0.2`, `modernbert-sde-v0.2`, `sde-content-relevancy`, `indus-sde-st-v0.2`, `tdamm-classification-v2`, `indus-sde-st-equat-v0.1`.
* Datasets (9): `WxC-Bench` (weather, Prithvi WxC), `nasa-smd-IR-benchmark` (the only gated repo seen: `gated: manual`), `nasa-smd-qa-benchmark`, `EO-via-NLP`, `nasa-sde-IR-benchmark-20251024-v5`, `nasa-science-repos-sme-benchmark`, `nasa-science-github-repos`, `nasa-science-code-benchmark-v0.1.1`, `nasa-science-code-benchmark-v0.1`.
* Space: `geoguard-demo`.

### 5.3 Surya-related repos outside the two orgs (from full-text search; not inventoried in depth)
* `johannesschmude/surya_visual_forecasting_demo`: a Gradio Space by a Surya co-author, sha `704b74f7226fe4c6f2ead453c9c6e6406975e335`, lastModified 2025-08-27. Its runtime stage at fetch time was **`RUNTIME_ERROR`**.
* `Smith42/surya_filament_dataset`: a cc-by-4.0 parquet dataset (created 2026-01-13). It has Surya-style 13-channel `*_current` / `*_previous` fields, 52 examples and a download_size of 33,289,879,714 B. It is third-party.
* `MartialTerran/HART-SURYA_model`: a third-party model (lastModified 2025-08-29) with no tags. Not examined.
* Search hits for `vikp/surya_*` and `datalab-to/surya*` are the unrelated "Surya OCR" project.

---

## 6. Inconsistencies and red flags (copy-exact evidence)

1. **The blog's SuryaBench link is dead.** It returns 401, as described in §1. The same URL is printed in SuryaBench arXiv v1 §6, and the SuryaBench GitHub README links `collections/nasa-impact/suryabench`, which is also 401. The working replacement is `https://huggingface.co/collections/nasa-ibm-ai4science/suryabench`.
2. **The live SuryaBench collection is incomplete.** It has 4 items and lacks `core-sdo`, `ar_emergence` and `surya-bench-coronal-extrapolation`. Yet the `core-sdo` card says "All the downstream tasks that uses core-sdo dataset can be found in the SuryaBench Hugging Face Collections". The org's "Surya" collection is empty.
3. **"15%" on the Surya-1.0 card is unsupported.**
   * The card says "surpasses existing benchmarks by **15%** in preliminary tests". The string "15%" appears in neither arXiv 2508.14112v2 nor the `SuryaFM.pdf` in the same repo.
   * Paper Table 4 gives Surya TSS 0.436 / HSS 0.522 / F1 0.561 against AlexNet 0.358 / 0.398 / 0.454.
   * The card also says "**9 years (≈218 TB)**", while the paper (§2.1, v2 HTML) says "The total size of the data for dataset is around 257 TB".
4. **Surya-1.0 card is stale.**
   * It says "Fine-tuned versions (coming soon)", but the four fine-tuned repos exist (uploaded 2025-08-18 to 2025-08-26). The card was last edited 2025-08-19.
   * Its BibTeX has `eprint={XXXX.XXXXX}` and `url={https://arxiv.org/abs/XXXX.XXXXX}`.
   * Its author list lacks Anne Spalding, who is in arXiv v2 and in `SuryaFM.pdf`.
   * The code fence is never closed.
5. **Three of four fine-tuned model cards are empty.** `solar_flares_surya`, `solar_wind_surya` and `euv_spectra_surya` each have a 31-byte README: no task, data, metric, base model, or link to code or paper.
6. **The `ar_segmentation_surya` row label differs from the paper.** The card says "Surya (LoRA)" and the paper's Table 3 says "Surya". The numbers are identical.
7. **The datasets changed after both arXiv papers.**
   * The arXiv papers are dated 2025-08-18 (SuryaBench v1) and 2025-08-21 (Surya v2).
   * Every SuryaBench dataset was re-split or re-uploaded between 2025-09-17 and 2025-10-08. Examples:
     * flare "Remove dataset_infos.yaml…" (2025-09-30) and "Add data.csv" (2025-10-06)
     * ar-seg "Re-upload CSV splits from splits_v2" (2025-09-30) and `data.tar.gz` re-uploaded 2025-10-04
     * solarwind "Update solar_wind splits … - v2" (2025-09-30)
     * euv "Replace split CSVs with new versions" (2025-10-01)
     * coronal "Upload data.tar.gz" (2025-10-01)
     * core-sdo "restructuring and name change" (2025-10-08)
   * Each then got a README-only commit on 2025-12-16.
   * **The current HF data is therefore probably not what produced the arXiv tables.** It is what *Scientific Data* 13:712 cites (refs 58–64), and its numbers differ from arXiv v1 (see `suryabench_paper_results.md` §4.3).
   * `usedStorage` far exceeds the current size for ar_emergence (3,809,756,040 vs 23,102,338), coronal (212,586,020,461 vs 36,181,107,009), ar-seg (5,971,018,868 vs 1,313,355,815) and euv (14,497,894,075 vs 12,062,209,078). Older revisions held different files.
8. **The `euv_spectra_surya` weights were re-uploaded on 2025-08-26**, after arXiv v2.
9. **Count mismatch in D4.** The card says "Total 128,352 instances", while the split CSVs have 128,328 rows.
10. **Small card errors.**
    * D7's card lists "`valid.csv`", but the file is `validation.csv` (renamed by commit `4161307e34`, 2025-09-30). D7's `pretty_name` is "Active Region Emergenc".
    * D2's card says "Size per File | ~570 MB", but the HF `.nc` files are 662,989,220–672,994,824 B; the paper says "about 630 MB". D2's card says "Total Size | ~360TB" for a repo holding 363.38 GB; the figure refers to the AWS bucket. D2 has the tag "faoundation-model" and `pretty_name: SuryaBench`.
    * D6's "~3.2GB" matches the split CSVs only in GiB (3,465,492,510 B = 3.23 GiB).
11. **Licence difference.** D1 is `mit`, while the other datasets are `cc-by-4.0` and the models `apache-2.0`.
12. **Stale upstream links.** `downstream_examples/solar_wind_forcasting/README.md:129` links `nasa-impact/Surya-bench-solarwind`, and `euv_spectra_prediction/README.md:167` links `nasa-impact/surya-bench-euv-spectra`. Both still resolve through 307 redirects.

---

## 7. What I could not reach or verify

* `https://huggingface.co/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b` (the blog's link): **HTTP 401**. I cannot see what it contained.
* The second (non-public) space of each org: the overviews report `numSpaces: 2`, but only 1 per org is listed publicly.
* Model weights and any dataset file over 50 MB were **not downloaded**, per the rules. Their content (keys, parameter counts, dtype) is not verified, and sizes and hashes come only from the HF tree API.
* I did not re-fetch the *Scientific Data* article in this session. Its refs 58–64 were read from the previously saved text `suryabench_evidence/paper/scidata_s41597-026-06552-5.txt`.
* `datasets-server` size information for `core-sdo` (failed) and `Surya-1.0_validation_data` (no supported data files).
