# Surya repo: runnable paths, data, checkpoints, CPU blockers

Written 2026-10-15 by reading the code. **I ran no model code.** Every number below is copied from the file, API response or page named next to it. Lines marked **[derived]** are my own arithmetic or my reading of the code. Lines marked **[estimate]** come from tensor shapes, not from a measurement.

Sources:
- Clone: `/home/user/upstream/surya/Surya` (read only, not modified).
- HF API: `https://huggingface.co/api/{models,datasets}/<id>/tree/main?recursive=true`.
- S3: anonymous `https://nasa-surya-bench.s3.amazonaws.com/?list-type=2&prefix=...` listings and `curl -sI`.
- Paper: arXiv HTML `https://arxiv.org/html/2508.14112`.

Downloads: only small files (<1 MB each, or a 2 MiB / 1 MiB / 256 KiB byte range). Scratch copies are in `/tmp/claude-0/-home-user-NJIT-CCSC-Reproduction/cafab427-273b-538d-9e6f-5e9528085d79/scratchpad/` (subfolders `hfapi/`, `hfsmall/`, `s3/`, `arxiv/`, `ckpthead/`, `nchead/`, `argz/`, `sim/`).

## 0. Which code version this is

- `git describe --tags` gives `v1.0.0-65-g78b827e`. So HEAD `78b827e` (2026-03-07, "Merge pull request #46 … rohit/zs_inference") is **65 commits after** tag `v1.0.0` (`bea20c1`, 2025-08-17). It is not the tag itself.
- `git ls-remote https://github.com/NASA-IMPACT/Surya.git` shows remote `main` = `78b827e…`, so the clone is current. `v1.0.0` is the only tag. The GitHub REST API was blocked by the session proxy ("GitHub access to this repository is not enabled for this session"), so I could not list releases.
- Changes from `v1.0.0` to HEAD (`git diff --stat v1.0.0 HEAD`): 54 files changed.
  - **All four `downstream_examples/*/infer.py`, all `config_infer.yaml`, all tutorial notebooks and the whole `easy_inference/` folder are new since v1.0.0.**
  - `tests/test_surya.py` changed `token=True` to `token=None` in both `snapshot_download` calls. So v1.0.0 needed a HF login; HEAD downloads anonymously.
  - `downstream_examples/download_data.py` was renamed to `download_sample_train_data.py`.
- This machine: `python3` is 3.13.16 and has none of torch, torchvision, peft, timm, einops, xarray, h5netcdf, h5py, hdf5plugin, numba, skimage, sunpy, huggingface_hub, wandb, matplotlib or pytest (only pandas and yaml). `aws`, `hf`, `huggingface-cli` and `torchrun` are not on PATH. `df` showed **6.8 GB available** on `/` at the time of writing.

## 1. Summary table

The HF datasets and models are all under `nasa-ibm-ai4science/`. "FM" means the foundation-model checkpoint `surya.366m.v1.pt` (1,800,384,309 B).

| Path | What it does | Data it needs (exact bytes from HF / S3) | Fine-tuned ckpt | GPU-only? | What it can be compared against |
|---|---|---|---|---|---|
| `tests/test_surya.py` | 2-step rollout of the FM on 2014-01-07; checks the mean MSE against 8 hard-coded reference losses | FM + 23 files `Surya-1.0_validation_data/20140107_1500…1924.nc` = 15,427,169,811 B in the whole dataset repo (each file ≈670.3–671.0 MB) | none | No. CUDA is optional; the autocast is hard-coded to `"cuda"`, so on CPU it runs fp32 | `REFERENCE_LOSS` in the test; README "Local loss 0.31665" |
| `easy_inference/run_easy_inference.py` | Zero-shot rollout of the FM over a date window taken from S3 (default 2014-10-23 10:00–17:00, 60 min cadence, 5 rollout steps); writes `prediction.nc` | FM + 8 S3 files `2014/10/20141023_{10..17}00.nc` = 4,712,058,364 B | none | No: device order cuda→mps→cpu. Needs the `aws` CLI for its download step | Nothing in the README or paper. It is a demo, and its MSE is in physical units for one sample |
| AR segmentation `infer.py` | Prints/saves figures of LoRA AR masks for 3 samples. **No ground truth, no IoU** | download_data.sh (hf branch) total 14,968,399,867 B [derived]. Minimum for `--num_samples 1`: 3 infer_data files, 2,018,843,831 B [derived] | `ar_segmentation_surya/ar_segmentation_weights.pth` 1,784,013,352 B | No (`--device cpu`) | README/paper IoU 0.768: **not computed anywhere in the repo** |
| Flare `infer.py` | Prints 0/1 prediction vs `label_max` for up to 3 samples | hf branch total 13,673,813,819 B. Uses 9 infer_data files = 6,024,034,262 B; index 0 alone needs 3 files, 2,018,843,831 B [derived] | `solar_flares_surya/solar_flare_weights.pth` 1,794,022,584 B | No | TSS 0.436: **not computable from infer_data** (all 3 samples have `label_max`=0, see §6.2) |
| Wind `infer.py` | Prints predicted vs observed V (km/s) and MAE/RMSE/R² over the chosen samples | hf branch total 13,682,018,780 B. Uses 9 infer_data files (same 9 as flare); index 0 alone needs 2 files, 1,345,849,007 B [derived] | `solar_wind_surya/solar_wind_weights.pth` 1,805,655,558 B | No | Paper RMSE 75.92 is over a whole split; infer covers at most 6 samples |
| EUV `infer.py` | Prints MAE/RMSE/R²/correlation on normalised spectra for the matched samples | hf branch total 14,584,661,234 B, including `euv-spectra/archive/AIA_EVE_temporally_stratified.nc` (933,907,814 B, needed whole). Notebook shows 2 samples, 4 files, 2,684,807,599 B [derived] | `euv_spectra_surya/euv_spectra_weights.pth` 1,779,722,720 B | No | Paper MAPE 1.4792510271: **MAPE is not implemented anywhere in the repo** |
| All four `finetune.py` | LoRA / head fine-tuning under `torchrun` | Full training SDO data (TB scale, see §5.3) | produces its own | **Yes.** It asserts CUDA, uses NCCL, and DDP with `device_ids=[torch.cuda.current_device()]` | Paper Tables 3–6 (only after full training) |

**[derived] The 9 GB disk budget versus these paths:**
- The full test (15.43 GB of data + 1.80 GB FM) does not fit.
- Every `download_data.sh` run as written (13.7–15.0 GB) does not fit.
- Each script downloads its own copy of the FM and the infer_data into its own `assets/`, so running all four tasks duplicates about 11.9 GB four times.
- What does fit:
  - test subsets, grouped by reference time (§3.4);
  - easy_inference with `rollout_steps: 0` and a 3-hour window (§4.4);
  - each downstream `infer.py` with `--num_samples 1` and only the 2–3 infer_data files that index 0 needs, if the FM file is shared (symlink) and not re-downloaded.

## 2. Assets on Hugging Face and S3

### 2.1 Model repos

| Repo (`nasa-ibm-ai4science/…`) | File | Size (B) | Card |
|---|---|---|---|
| `Surya-1.0` | `surya.366m.v1.pt` | 1,800,384,309 | Full card with BibTeX. Its arXiv id is still a placeholder `eprint={XXXX.XXXXX}` |
| | `config.yaml` / `scalers.yaml` | 680 / 3,835 | |
| | `SuryaFM.pdf`, `surya_arch.png` | 18,347,597 / 1,706,852 | |
| `ar_segmentation_surya` | `ar_segmentation_weights.pth` | 1,784,013,352 | Has a results table: UNet 9.2 M IoU 0.688 Dice 0.801; Surya (LoRA) 4.1 M IoU **0.768** Dice **0.853** |
| `solar_flares_surya` | `solar_flare_weights.pth` | 1,794,022,584 | README.md is 31 B (license line only) |
| `solar_wind_surya` | `solar_wind_weights.pth` | 1,805,655,558 | README.md is 31 B |
| `euv_spectra_surya` | `euv_spectra_weights.pth` | 1,779,722,720 | README.md is 31 B |

The Surya-1.0 card still says "Fine-tuned versions (coming soon)", but all four fine-tuned repos exist.

**Checkpoint internals.** I read the first 2 MiB of each file with an HTTP Range request and parsed the opcodes of `data.pkl` with `pickletools.genops`. Nothing was unpickled.

| Checkpoint | Zip archive root name | State-dict key style |
|---|---|---|
| AR | `epoch_40_loss_0.002` | `base_model.model.…` with `lora_A`/`lora_B` (PEFT-wrapped) |
| flare | `epoch_5_loss_0.189` | same, plus `base_model.model.cls_token` |
| wind | `epoch_13_loss_0.001` | same as flare |
| EUV | `checkpoint` | plain keys `embedding.…`, `backbone.…`, no LoRA |
| FM | `epoch_2_loss_0.230` | plain keys |

This matches `infer.py`, which builds the model, then applies PEFT LoRA (AR, flare, wind), then calls `load_state_dict(strict=True)`. **[derived]** The AR archive is named `epoch_40`, but `ar_segmentation/config.yaml` has `max_epochs: 15`. So the released AR weights were probably not produced by the config that ships in the repo.

### 2.2 Dataset repos (`nasa-ibm-ai4science/…`)

| Dataset | Relevant files and sizes (B) |
|---|---|
| `Surya-1.0_validation_data` | 23 × `20140107_HHMM.nc` (15:00–19:24, 12 min cadence), 670,337,550–671,046,501 each. Repo total 15,427,169,811. Card: "The data is compressed; you should use the hdf5plugin" |
| `core-sdo` (total 363,376,705,743) | `infer_data/*.nc`: 15 files, 10,051,122,073 total (662,989,220–672,994,824 each) |
| | `infer_data/infer_index_surya_1_0.csv` 643 |
| | `train_index_surya_1_0.csv` 18,557,847; `valid_index_surya_1_0.csv` 961,993; `scalers.yaml` 3,835 |
| | `validate_splitted_tars/val.tar.part_aa…aq`: 17 parts, 177,452,584,960 total |
| | `test_splitted_tars/test.tar.part_aa…aq`: 17 parts, 175,829,800,960 total |
| `surya-bench-ar-segmentation` | `data.tar.gz` 1,306,605,651 (HDF5 masks). CSVs: train 3,922,895, validation 192,661, test 2,311,618, leaky_validation 318,065 |
| `surya-bench-flare-forecasting` | CSVs: data 4,377,302, train 2,517,593, validation 122,615, test 1,531,173, leaky_validation 206,107 |
| `Surya-bench-solarwind` | CSVs: train 3,054,147, validation 149,865, test 1,746,629, leaky_validation 256,013 |
| `euv-spectra` (total 12,062,209,078) | `archive/AIA_EVE_temporally_stratified.nc` 933,907,814 (the only file the code reads); `archive/AIA_EVE_dataset_combined.nc` 1,018,976,075 |
| | CSVs: train 2,984,963,110, validation 66,789,263, test 110,815,206, leaky_validation 302,924,931 |

- The `core-sdo` include set used by every `download_data.sh` is `*_index_surya_1_0.csv`, `infer_data/*` and `scalers.yaml`. It is 19 files, 10,070,646,391 B **[derived]**. The notebooks log "Fetching 19 files", which matches.
- infer_data contents: `infer_index_surya_1_0.csv` lists 15 hourly-ish timestamps on 2011-01-20 (01:00–03:00), 2012-01-15 (00:48–03:00, 6 files), 2013-01-28 (18:12–20:12) and 2019-01-23 (02:00–04:00). Paths are relative, e.g. `20110120_0100.nc`.
- `valid_index_surya_1_0.csv` (downloaded):
  - 18,360 rows, 17,710 with `present=1`;
  - only Jan 15–31 of 2011–2019; 3,477 present rows fall on the hour;
  - paths are S3-style relative paths such as `2011/01/20110115_0000.nc`.
- Split naming:
  - The task CSV cards define `validation.csv` as Jan 15–31 of 2010–2019 and `test.csv` as 2020–2024. For EUV, validation is Jan 15–31 of 2012 and 2013, and test is Jan 15–31 of 2011 and 2014.
  - The paper's pretraining split (§2.1, "Train/test split") calls days 15–31 the **test** set.
  - I did not find a sentence in the downstream sections of the paper saying which split Tables 3–6 use.
- CSV headers (downloaded):
  - flare `timestamp,max_goes_class,cumulative_index,label_max,label_cum`. The repo flare README says `timestep,…`, but the code reads `timestamp`, which is correct.
  - AR `timestamp,file_path,present`, with paths like `data/2011/01/20110115_0000.h5`.
  - wind `timestamp,V,BX_GSE,BY_GSM,BZ_GSM,N`.
- AR `data.tar.gz` layout: I range-read the first 256 KiB and parsed the tar headers. Members start `2010/`, `2010/08/20100818_2100.h5` (42,637 B) and so on. So the README step `mkdir -p data; tar -xvzf data.tar.gz -C data` gives the `data/YYYY/MM/…h5` paths the CSVs expect.
- The `nasa-impact/Surya-bench-solarwind` and `nasa-impact/surya-bench-euv-spectra` links in the READMEs redirect (307) to the `nasa-ibm-ai4science/` ids.
- `nasa-impact/Surya-bench`, the Documentation link in the AWS registry entry, returned 401.
- The SuryaBench collection API `https://huggingface.co/api/collections/nasa-impact/suryabench-68265ce306fc2470c121af7b` returned `{"error":"Invalid username or password."}`, so **I could not reach the collection contents through the API.**

### 2.3 S3 bucket `nasa-surya-bench`

- **Public and not requester-pays.**
  - An anonymous ListObjectsV2 (`?list-type=2&delimiter=/`) returned HTTP 200, with `x-amz-bucket-region: us-west-2`.
  - An anonymous `curl -sI https://nasa-surya-bench.s3.amazonaws.com/2014/10/20141023_1000.nc`, sent with no `x-amz-request-payer` header, returned 200 OK: `Content-Length: 588351188`, `Content-Type: application/x-netcdf`, `x-amz-storage-class: INTELLIGENT_TIERING`.
  - A requester-pays bucket refuses anonymous requests. Also, the AWS Open Data registry entry (`https://raw.githubusercontent.com/awslabs/open-data-registry/main/datasets/surya-bench.yaml`) has no `RequesterPays` field, carries the tag `aws-pds`, and lists ARN `arn:aws:s3:::nasa-surya-bench`, Region `us-west-2`.
- Layout: top-level prefixes `2010/` … `2024/` plus `index.html` (44,168 B). Keys look like `YYYY/MM/YYYYMMDD_HHMM.nc`.
- Example days:
  - `2014/10/20141023*`: 113 objects, 66,611,518,532 B. Some 12-min slots are missing (e.g. 0600, 0612, 1336, 1412, 1424, 1800, 1812), but every hourly 10:00–17:00 file exists.
  - `2014/01/20140107*`: 118 objects, 69,827,005,911 B. The 15:00–19:24 test window is 23 files, 13,624,296,917 B.
- **The S3 copies are not byte-identical to the HF copies.**
  - Sizes: `2014/01/20140107_1600.nc` is 593,250,192 B on S3 and 670,337,550 B on HF validation_data. `2011/01/20110120_0100.nc` is 591,219,147 B on S3 and 672,958,156 B in HF infer_data.
  - Compression (from the first 1 MiB of each): the S3 header contains the string `deflate`. The HF header contains `HDF5 lz4 filter; see http://www.hdfgroup.org/services/contributions.html`.
  - Metadata: both carry the same FITS metadata (`t_rec 2014-01-07T15:59:25.000`). The latest embedded dates differ: `2025-04-03T12:41:18` on S3 and `2025-05-22T11:23:58` on HF.
  - Both encodings are lossless, but **I did not verify that pixel values match.** So the test's reference losses are only guaranteed for the HF files.

## 3. `tests/test_surya.py`

### 3.1 What it does
1. A module fixture runs `snapshot_download` twice, both with `token=None`:
   - `nasa-ibm-ai4science/Surya-1.0` with `allow_patterns=["config.yaml","scalers.yaml","surya.366m.v1.pt"]`, into `data/Surya-1.0`;
   - dataset `nasa-ibm-ai4science/Surya-1.0_validation_data` with `allow_patterns="20140107_1[5-9]??.nc"`, into `data/Surya-1.0_validation_data`. That pattern matches all 23 files, 15:00–19:24.
2. It builds `HelioSpectFormer` from the HF `config.yaml`:
   - img 4096, patch 16, embed 1280, depth 10, 2 spectral blocks, 16 heads;
   - time_delta_input [-60, 0], target +60;
   - `checkpoint_layers=[0..9]`, `dtype=torch.bfloat16`. The docstring in `surya/models/helio_spectformer.py` says dtype is "Not used".
   - It then loads `surya.366m.v1.pt` with `strict=True`.
3. Dataset: `HelioNetCDFDataset(index_path="tests/test_surya_index.csv", rollout_steps=1, phase="valid")`. The index has 23 rows with relative paths `data/Surya-1.0_validation_data/…`. DataLoader: `batch_size=1, num_workers=2, prefetch_factor=None, pin_memory=True`, no shuffle.
4. Device: `torch.cuda.current_device()` if CUDA is available, else `"cpu"`, with the warning "No GPU detected. Running the test on CPU."
5. Loop: `with torch.autocast(device_type="cuda", dtype=torch.bfloat16): with torch.no_grad(): batch_step(..., rollout=1)`.
   - `batch_step` makes 2 forward passes: a 1-hour forecast, then a 2-hour forecast that feeds the first forecast back in.
   - It computes `F.mse_loss` in normalised (signum-log, standardised) units at each step and divides the sum by 2.
6. Assertion: `abs(mean(loss) - mean(REFERENCE_LOSS)) < 1.0e-4`.
   - `REFERENCE_LOSS` = [0.24890851974487305, 0.31166040152311325, 0.32497961074113846, 0.32198894023895264, 0.3090871721506119, 0.3190929591655731, 0.36053016781806946, 0.33693042397499084].
   - **[derived]** Their mean is 0.31664727441966534. The README's expected output prints "Local loss 0.31665".
7. Only if the assertion passes, it writes the 8×4 AIA-94 panel `surya_model_validation.png`. It also creates `logs/data/`.

### 3.2 Reference times and files [derived]
Valid references are the timestamps t where t−60, t, t+60 and t+120 all exist. That gives 8 references, 16:00, 16:12, …, 17:24, each reading 4 files. Because the loader is not shuffled, `REFERENCE_LOSS[i]` should belong to reference 16:00 + 12·i min. That mapping is my reading of the code; the test itself only checks the mean.

| i | ref | files (all 2014-01-07) | bytes |
|---|---|---|---|
| 0 | 16:00 | 1500, 1600, 1700, 1800 | 2,682,616,585 |
| 1 | 16:12 | 1512, 1612, 1712, 1812 | 2,683,028,175 |
| 2 | 16:24 | 1524, 1624, 1724, 1824 | 2,682,966,512 |
| 3 | 16:36 | 1536, 1636, 1736, 1836 | 2,682,765,469 |
| 4 | 16:48 | 1548, 1648, 1748, 1848 | 2,682,881,844 |
| 5 | 17:00 | 1600, 1700, 1800, 1900 | 2,683,048,005 |
| 6 | 17:12 | 1612, 1712, 1812, 1912 | 2,683,220,159 |
| 7 | 17:24 | 1624, 1724, 1824, 1924 | 2,683,221,628 |

Pairs (0,5), (1,6) and (2,7) share 3 files, so each pair needs only 5 files (about 3.35 GB) **[derived]**.

### 3.3 Things that matter on CPU
- `torch.autocast(device_type="cuda")` on a CPU-only machine is turned off with a warning, so the test runs **fp32 on CPU**. The AR notebook output shows the same warning: "User provided device_type of 'cuda', but CUDA is not available. Disabling".
- Getting bf16 on CPU would need a modified copy of the test, kept outside the repo. The reference values were presumably made on GPU with bf16 autocast. That is not stated, only implied by the hard-coded `"cuda"`.
- **[estimate] RAM risk.** One sample holds `ts` and `forecast`, each 13×2×4096×4096 fp32 = 1,744,830,464 B, so about 3.49 GB per batch. With `num_workers=2` and PyTorch's default `prefetch_factor=2`, up to 4 batches (about 14 GB) can be buffered while the slow CPU forward runs. On a 15 GB machine a modified copy should use `num_workers=0`.
- **Paths are relative to the working directory** (`data/…`, `tests/test_surya_index.csv`, `logs/data`, and the output PNG). With a read-only clone, run it from a scratch directory that contains a copy of `tests/`, with `PYTHONPATH` pointing at the clone.
- The line numbers in the README's expected log (`test_surya.py:188/195/199/201/215`) no longer match the file (now 254/261/267/269/285). This is cosmetic.
- `pyproject.toml` declares the script `surya-test = "surya.utils.misc:run_tests"`, but `surya/utils/misc.py` has no `run_tests`, so that entry point is broken.

## 4. `easy_inference/`

### 4.1 What it does (`run_easy_inference.py`, 1,920 lines, new since v1.0.0)
1. It reads `config_easy.yaml`:
   - `user`: start `2014-10-23 10:00:00`, end `2014-10-23 17:00:00`, `prompt_for_dates: true`, `rollout_steps: 5`, `output_dir: easy_inference/outputs_24h`.
   - `advanced`: `cadence_minutes: 60`, `time_delta_input_minutes: [-60, 0]`, `time_delta_target_minutes: 60`, `s3_bucket: nasa-surya-bench`, `device: auto`, `dtype: auto`, `num_workers: 0`, `gt_prefetch_workers: 4`, `prediction_dtype: float32`.
   - It asks for the dates on stdin unless `--no-prompt` is given.
2. `_validate_rollout_against_window` checks that the requested rollout fits the window. In the default, the first reference is 11:00 and the maximum rollout is 5.
3. `ensure_model_assets` downloads the FM (`config.yaml`, `scalers.yaml`, `surya.366m.v1.pt`) from `nasa-ibm-ai4science/Surya-1.0` into `data/Surya-1.0` if any is missing.
4. Download: it runs `aws s3 ls s3://nasa-surya-bench/YYYY/MM/ --no-sign-request` and `aws s3 cp … --no-sign-request` as subprocesses. **This needs the AWS CLI**, which `pyproject.toml` lists as `awscli>=1.44.36` and which is not on PATH here. `--skip-download` reuses files already in `validation_data_dir` (`data/Surya-1.0_validation_data_20141023_60min`), so `curl` from `https://nasa-surya-bench.s3.amazonaws.com/<key>` can stand in.
5. It builds the index CSV, then the custom `InputOnlyRolloutDataset`. Only the **first valid sample** is used (batch 1).
6. It runs an autoregressive rollout of `rollout_steps+1` steps (default 6: 12:00 … 17:00), inverse-transforms to physical units, and writes `prediction.nc` with `<channel>` and `gt_<channel>` variables, dims (sample, prediction_time, y, x).
7. It prints a report that includes "Avg rollout MSE". That is the mean over steps of a per-step MSE over all 13 channels in **physical units** (`np.mean(diff*diff)` after the inverse transform). It is not the paper's model-unit MSE.
8. `--skip-gt` skips the ground-truth loading. `--dry-run` prints the settings only.

### 4.2 Device, precision and CUDA handling
- `resolve_device("auto")` picks cuda, then mps, then cpu.
- `resolve_dtype("auto", "cpu")` picks bf16 if `torch.cpu.is_bf16_supported()`, else fp32. On this AVX512-BF16/AMX CPU it will probably choose **bf16 autocast on CPU**. `dtype: float32` or `disable_autocast: true` in a copied config turns that off.
- `torch.compile(mode="reduce-overhead")`, TF32, cuDNN benchmark, `torch.cuda.synchronize` and the memory statistics only run when `device_type == "cuda"`. **No hard CUDA dependency.**

### 4.3 What it can be compared against
- No README or paper number corresponds to it.
- `compare_prediction_groundtruth.ipynb` has saved outputs for channel aia94 in physical units:
  - step 01 MSE=0.782106, step 02 0.421440, step 03 0.442715, step 04 0.551782, step 05 3.435462, step 06 0.932981;
  - source file `/nas/rgroup/aifm/rohit/Surya/easy_inference/outputs_24h/prediction.nc`. The window used for that run is not recorded.
- The README says the run writes `easy_inference/outputs_.../metrics/*`, but the script writes no metrics directory (grep finds no metrics/CSV/JSON writer). Only `prediction.nc` and the optional debug log are written.

### 4.4 Disk and RAM [estimate]
- Per frame: 13 × 4096 × 4096 × 4 B = 872,415,232 B.
- `prediction.nc` stores predictions and ground truth uncompressed (`h5netcdf create_variable` with no compression arguments): 2 × 13 × 4096² × 4 B = 1,744,830,464 B per prediction step.
  - Default 6 steps: about **10.47 GB** in fp32, half that with `prediction_dtype: float16`.
  - Default run total: 4.71 GB inputs + 1.80 GB FM + about 10.5 GB output, roughly 17 GB. **That does not fit in 9 GB.**
- A run that does fit: `rollout_steps: 0` with window 10:00–12:00.
  - Inputs 10:00 and 11:00, ground truth 12:00: 3 S3 files, 1,765,937,327 B.
  - `prediction.nc` about 1.74 GB; FM 1.80 GB. Total about 5.3 GB.
- RAM: the ground-truth prefetch submits all `prediction_steps` frame loads at once (`ThreadPoolExecutor(max_workers=max(4, prediction_steps))`), so the default run holds 6 × 0.87 GB, plus 2 input frames, the model (366 M params × 4 B ≈ 1.46 GB) and activations.

## 5. `downstream_examples/`: what all four tasks share

### 5.1 `download_data.sh` (identical structure in all four)
- If the `hf` CLI exists, it downloads:
  1. the full task dataset repo into `assets/<repo name>`. EUV passes `--include "*_temporally_stratified.nc"`; the others take everything.
  2. the task's fine-tuned `*.pth` into `assets/`;
  3. `Surya-1.0/surya.366m.v1.pt` into `assets/`, a second FM copy alongside the test's `data/Surya-1.0` copy;
  4. `core-sdo` with `--include "*_index_surya_1_0.csv" "infer_data/*" "scalers.yaml"` into `assets/`.
- If `hf` is missing, it runs a Python heredoc fallback instead:
  - **AR and flare fallbacks are syntactically broken.** The last `snapshot_download(…, allow_patterns=["surya.366m.v1.pt"],` is never closed (`ar_segmentation/download_data.sh` and `solar_flare_forcasting/download_data.sh`).
  - The wind and EUV fallbacks are closed correctly, but they download the **whole** task repo. For EUV that is 12,062,209,078 B; the hf branch takes 933,907,814 B.
  - None of the fallbacks downloads the fine-tuned `.pth`. Each `infer.py` does that itself (§5.2).
- `HF_HUB_ENABLE_HF_TRANSFER=1` is set but never exported, so it has no effect.
- Download totals **[derived]**:

  | Task | hf branch total | Python fallback total |
  |---|---|---|
  | AR | 14,968,399,867 B | |
  | flare | 13,673,813,819 B | |
  | wind | 13,682,018,780 B | |
  | EUV | 14,584,661,234 B | 25,712,962,498 B |

- The notebooks call it as `!sh download_data.sh`. The script relies on bash features: shebang `bash`, `[[ ]]`, `set -o pipefail`.

### 5.2 `infer.py` (identical pattern in all four, new since v1.0.0)
- `__main__` first runs `snapshot_download(repo_id="nasa-ibm-ai4science/<task>_surya", local_dir="./assets", allow_patterns='*.pth', token=None)`, then `main()`.
- CLI flags:
  - `--config_path` defaults to `./config_infer.yaml`. The READMEs' table says `./config.yaml`; that is wrong for HEAD.
  - `--device` defaults to `cuda`. It falls back to CPU if CUDA is unavailable, but autocast still uses `device_type=args.device`.
  - `--num_samples` defaults to 3.
  - `--data_type` is parsed but **never used** in any of the four.
- Samples: `set_global_seed(42)`, then `Subset(dataset, [0] + (torch.randperm(len-1)[:num_samples-1]+1))`. Index 0 is always included. `batch_size=1`, `num_workers` from the config, `prefetch_factor=None`, `pin_memory=True`.
- Model loading:
  - Flare, wind and EUV call `finetune.get_model`. That **raises `ValueError` unless `pretrained_path` (`./assets/surya.366m.v1.pt`) exists**, loads the FM weights, and is then overwritten by the strict load of the fine-tuned checkpoint. **So these three need the 1.8 GB FM file on disk even though its weights end up overwritten.**
  - AR builds its model locally and does not need the FM file.
  - The EUV `load_model` falls back to `strict=False` and only prints missing or unexpected keys, so a partial load would go unnoticed.
- The "Advanced Usage" commands in the READMEs pass `--config_path ./config.yaml`. **[derived] That breaks in all four:**
  - flare `config.yaml` has no `data.flare_data_path` (KeyError);
  - EUV `config.yaml` has no `data.infer_data_path` or `data.infer_solar_data_path` (KeyError);
  - AR and wind `config.yaml` point `sdo_data_root_path` at the authors' cluster path `/nobackupnfs1/sroy14/processed_data/Helio/nc`.
- `HelioNetCDFDataset.__getitem__` catches a failed load, moves to `idx+1` and tries up to 100 times. So if an infer_data file is missing, the script **silently uses a different sample** instead of failing.
- **Saved notebook outputs**: the authors ran all four tutorial notebooks on CPU ("CUDA available: False", "PyTorch version: 2.8.0+cu126"). Those notebooks log `Loading pretrained model from ../../data/Surya-1.0/surya.366m.v1.pt`, while the current `config_infer.yaml` files say `./assets/surya.366m.v1.pt`. So they were run with an earlier config.

### 5.3 `finetune.py` (all four): GPU-only
- Command: `torchrun --nnodes=1 --nproc_per_node=1 --standalone finetune.py`.
- `--gpu` is declared `default=True, action="store_true"`, so it cannot be switched off. `if not args.gpu: raise ValueError("Training scripts are not configured for CPU use…")`.
- `surya.utils.distributed.init_ddp(use_gpu=True)` runs `assert torch.cuda.is_available()`, `init_dist("nccl", …)` and `torch.cuda.set_device(local_rank)`. There is a `gloo` branch, but it is unreachable from the CLI.
- `DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])`; `autocast(device_type="cuda", dtype=config["dtype"])`; `GradScaler()`; `dist.all_reduce` inside `evaluate_model`. That means `evaluate_model` cannot run without an initialised process group.
- `import wandb` is required at import time. W&B logging only happens with `--wandb`, and then in offline mode.
- Data: `config.yaml` needs `train_index_surya_1_0.csv` / `valid_index_surya_1_0.csv`, whose paths are relative S3-style paths, with `sdo_data_root_path` changed from the authors' cluster path.
  - **[derived]** The validation index alone has 17,710 present 12-min files. At about 0.59 GB each (S3) that is roughly 10 TB.
  - `download_sample_train_data.py` would download the 17-part `val.tar` (177,452,584,960 B) and `test.tar` (175,829,800,960 B) splits. Its own README warns that the parts, the combined tar and the extracted files must all fit on disk at once.
  - The `downstream_examples/README.md` directory tree shows what those tars contain: `common_data/test/20110103_1200.nc … 20110105_2348.nc` and `common_data/validate/20110106_0000.nc … 20110108_1148.nc`. That is roughly 3 days of 12-min frames per split. The script's CSV builder hard-codes 2011-01 to 2011-02 (`start_year=2011, start_month=1, end_year=2011, end_month=2`).
  - **[derived]** Jan 3–8 fall in the "leaky" buffer days 1–14, not in the Jan 15–31 validation days. I did not run this script.
- Validation inside training runs only `iters_per_epoch_valid: 200` batches per epoch from a `StatefulDistributedSampler`. Those are not full-split metrics.
- **No `.cuda()` calls, no flash-attn, no custom CUDA kernels anywhere** (see §7). The backbone itself (spectral gating with `torch.fft.rfft2` cast to fp32; long-short attention) is plain PyTorch.

## 6. Per task

### 6.1 Active-region segmentation (`ar_segmentation/`)
- **Data**:
  - `nasa-ibm-ai4science/surya-bench-ar-segmentation`: `data.tar.gz` 1,306,605,651 B of HDF5 masks (key `union_with_intersect`), plus the CSVs in §2.2. The card says "Total Instances: Total 128,352 instances", "Size per File ~ 53kB", "Total Dataset Size ~ 1.2GB".
  - core-sdo infer_data, about 10.07 GB.
  - The README's extra step `python create_ar_csv.py` is **obsolete**. It looks for `*.pth` masks under `assets/surya-bench-ar-segmentation/{year}_extracted/…` and writes `assets/ar_csv_files/ar_2011_m01d06-m01d08.csv`, which nothing reads. The config reads the HF CSVs directly (`ar_index_train/valid/test`).
  - The README lists `run_inference_example.sh`, which **does not exist**.
  - The README's system requirements say "Hugging Face account for data access", but `token=None` is used everywhere.
- **Checkpoint**: `nasa-ibm-ai4science/ar_segmentation_surya/ar_segmentation_weights.pth`, 1,784,013,352 B, PEFT LoRA r=32, α=64.
- **Inference path**: `python infer.py --device cpu --num_samples N`.
  - Uses plain `HelioNetCDFDataset` on infer_data with time_delta_input [0], target +60, **rollout_steps=1** (it reads t, t+60 and t+120 even though only t is used).
  - Five valid samples **[derived]**: 2011-01-20 01:00, 2012-01-15 00:48, 2012-01-15 01:00, 2013-01-28 18:12, 2019-01-23 02:00. All 15 infer_data files are touched across them. Index 0 needs `20110120_0100/0200/0300.nc` (2,018,843,831 B).
  - Output: `inference_results/test_{i}.png`, with input AIA channels and the sigmoid prediction. **No masks are loaded and no metric is computed.** The README's claim that the output shows "predictions, and ground truth" is not true of this script.
  - The notebook's saved output is images only.
- **What it can be compared against**: README IoU 0.768 vs 0.688 (UNet); paper Table 3: Unet 9.2 M IoU 0.688 Dice 0.801, Surya 4.1 M IoU 0.768 Dice 0.853. **Nothing in the repo computes IoU or Dice as a metric.** `DiceLoss` and `IoULoss` in `finetune.py` are defined but unused; the criterion is `BCEWithLogitsLoss`, and `evaluate_model` reports MAE/RMSE/R².
- **Smallest IoU check [derived]**: masks exist in `validation.csv` (`present` 1.0) for the 9 on-the-hour infer_data times (2011-01-20 01–03, 2012-01-15 01–03, 2019-01-23 02–04). An IoU on those 9 frames would need:
  - those masks, which can be taken out of `data.tar.gz` while streaming it (1.31 GB transferred, about 53 kB stored each);
  - a short evaluation script kept outside the repo.
  
  9 frames is not the paper's split.
- **Defaults**: `config.yaml` batch_size 1, `num_data_workers` 8, `dtype: bfloat16`, LoRA r 32 / α 64 / dropout 0.1 on q/k/v/out_proj and fc1/fc2, lr 0.001, `max_epochs` 15, `iters_per_epoch_train` 2000, `checkpoint_layers: []`, `pretrained_path: ../../data/Surya-1.0/surya.366m.v1.pt`.
- **Training bug [derived]**: `finetune.get_model` loads FM weights only `if config["model"]["model_type"] == "spectformer"`. The AR config uses `spectformer_lora`, so **running the AR fine-tune as shipped never loads the pretrained backbone**, and PEFT then freezes a randomly initialised base.

### 6.2 Solar-flare forecasting (`solar_flare_forcasting/`)
- **Data**:
  - `nasa-ibm-ai4science/surya-bench-flare-forecasting` CSVs, 8,760,535 B in total.
  - The card says `label_max` = 1 if max ≥ M1.0, 24 h window, "Total 128,328 instances". `validation.csv` has 3,672 rows, of which 400 have `label_max=1` **[derived]**.
  - core-sdo infer_data.
  - `config_infer.yaml` uses `scalers_path: ../../data/Surya-1.0/scalers.yaml`, unlike the other three tasks. So it depends on the test's download location.
- **Checkpoint**: `solar_flares_surya/solar_flare_weights.pth`, 1,794,022,584 B. The notebook prints "trainable params: 1,024,000 || all params: 364,593,153 || trainable%: 0.28%".
- **Inference path**: `python infer.py --device cpu`.
  - `SolarFlareDataset` with input [-60, 0], target +60, rollout 0, filtered to timestamps in `validation.csv`.
  - **[derived]** Exactly 3 valid samples: 2011-01-20 02:00 (`B4.2`), 2012-01-15 02:00 (`C2.5`), 2019-01-23 03:00 (`B1.2`). **All have `label_max`=0.** They use 9 files, 6,024,034,262 B.
  - It prints `int(sigmoid(logit) > 0.5)` against the label in a table. There is no aggregate metric. With zero positives, recall and therefore TSS are undefined for this set.
  - The notebook's saved output lists the earliest *input* time against a target 2 h later: `2011-01-20T01:00 | 2011-01-20T03:00 | 0 | 0`, `2019-01-23T02:00 | 2019-01-23T04:00 | 0 | 0`, `2012-01-15T01:00 | 2012-01-15T03:00 | 1 | 0`. That is one false positive in three.
  - The README's example rows (2014-01-07/08/09) are illustrative. Those times are not in the current infer_data.
- **What it can be compared against**: README TSS 0.436 vs 0.358 (AlexNet). Paper Table 4 (TSS/HSS/F1): AlexNet 0.358/0.398/0.454, ResNet50 0.018/0.028/0.055, Surya 0.436/0.522/0.561.
  - TSS and HSS are implemented only in `metrics.py` (`DistributedClassificationMetrics`), which is used inside `finetune.evaluate_model` (needs DDP).
  - **[derived]** Its threshold of 0.5 is applied to **logits**, which corresponds to probability > 0.622. `infer.py` applies sigmoid > 0.5. The two are inconsistent.
  - **[derived]** Reproducing TSS on `validation.csv` needs 3,243 samples (350 positive) with t−60, t and t+60 present. That is about 3,477 on-the-hour SDO files, roughly 2 TB from S3.
- **Defaults**: batch 1, workers 4, `dtype: float32` (CPU autocast turns itself off: "CPU Autocast only supports dtype of torch.bfloat16, torch.float16"), `checkpoint_layers [0..9]`, LoRA r 8 / α 8, lr 0.0001, 20 epochs, `global_class_token: True`, `freeze_backbone: false`.

### 6.3 Solar-wind speed (`solar_wind_forcasting/`)
- **Data**:
  - `nasa-ibm-ai4science/Surya-bench-solarwind` CSVs (5,332,522 B including a PNG). The card says OMNI hourly, 2010-05-13 to 2024-12-31, "Number of Time Steps 119,225".
  - core-sdo infer_data.
- **Checkpoint**: `solar_wind_surya/solar_wind_weights.pth`, 1,805,655,558 B. The notebook prints "trainable params: 8,192,000 || all params: 367,501,313 || trainable%: 2.23%". LoRA r 64 / α 32.
- **Inference path**: `python infer.py --device cpu`.
  - `WindSpeedDSDataset` on infer_data with input [0], target +60. It matches each SDO time t to the V observed at t + 4 days (`merge_asof`, forward, tolerance 1h), using **`validation.csv`**, which `config_infer.yaml` assigns to `solarwind_train_index`.
  - **[derived]** 6 valid samples: 2011-01-20 01:00 / 02:00 (V 344.0 / 351.0), 2012-01-15 01:00 / 02:00 (361.0 / 360.0), 2019-01-23 02:00 / 03:00 (433.0 / 438.0). They use the same 9 files as flare.
  - Saved notebook output:
    - `393.63 | 344.00` (2011-01-20T01:00), `391.95 | 433.00` (2019-01-23T02:00), `383.07 | 351.00` (2011-01-20T02:00);
    - "Mean Absolute Error | 40.9170", "Root Mean Square Error | 41.5408", "R² Score | -0.0569", 3 samples.
  - My simulation of the index logic reproduces exactly these three ground-truth values (344.0, 433.0, 351.0).
  - The printed "Time Target" is t + 60 min, from the SDO dataset. It is not the real 4-day target time.
- **Normalisation issue [derived]**: `WindSpeedDSDataset` min-max scales V with the min and range of *whichever CSV it was given*:
  - `train.csv`: min 240.0, range 560.0, max 800.0;
  - `validation.csv`: min 259.0, range 494.0.
  
  Training normalises with the train statistics, while `infer.py` and the training-time validation set use the validation statistics. So the km/s predictions printed by `infer.py` are rescaled with the wrong constants. Re-mapping the three notebook predictions with the train constants gives about 392.6, 390.7 and 380.7 km/s, a shift of 1–2.4 km/s.
- **What it can be compared against**: README RMSE 75.92 vs 93.76 (ResNet50). Paper Table 6 (RMSE / MAE / validation MSE): AlexNet 118.6 / 95.7 km / 13839.49, ResNet50 93.76 / 74.65 / 8547.924, Surya 75.92 / 58.06 / 5698.62. Table 7 compares against literature (Surya 75.92, WindNet 193 84.33 ± 2.31, 27-day persistence 89, and others).
- **Defaults**: batch 1, workers 8, `dtype: float32`, `checkpoint_layers [0..9]`, lr 0.001, 20 epochs (`config_infer.yaml` says `max_epochs: 2`), `ds_time_delta_in_out: "4D"`, `ds_time_tolerance: "1h"`.
- Minor: `type(x) == "float"` is always False, which produces the harmless message "Scalers are not a list of torch tensors…". `invlogsin` calls `torch.pow` with one argument, but it is never used.

### 6.4 EUV spectra (`euv_spectra_prediction/`)
- **Data**:
  - `nasa-ibm-ai4science/euv-spectra/archive/AIA_EVE_temporally_stratified.nc`, 933,907,814 B, read whole through `xarray` (`train_time/val_time/test_time` with `*_spectra`, 1343 bins).
  - Preprocessing: zeros are replaced by per-wavelength minima, then log10, then min-max scaling with the fixed range −9.00 to −1.96.
  - core-sdo infer_data.
- **Checkpoint**: `euv_spectra_surya/euv_spectra_weights.pth`, 1,779,722,720 B. There is **no LoRA**: `freeze_backbone: True`, and the notebook reports "5 parameters require gradients: cls_token, linear.weight, linear.bias, unembed.weight, unembed.bias". The output passes through a sigmoid.
- **Inference path**: `python infer.py --device cpu`.
  - `EVEDSDataset` with `ds_time_column="val_time"`, tolerance "6m", forward matching.
  - The notebook got "Dataset size: 2", samples 2012-01-15T00:48 and 2013-01-28T18:12. Those use files 20120115_0048/0148 and 20130128_1812/1912, 2,684,807,599 B **[derived]**.
  - Saved output: "Mean Absolute Error | 0.003811", "Root Mean Square Error | 0.013978", "R² Score | 0.966645", "Average Spectral Correlation | 0.983955", 2 samples.
  - The README example (2011-01-07 samples: MAE 0.005164, RMSE 0.008369, R² 0.985208, correlation 0.994224) came from a different data setup; 2011-01-07 is not in infer_data.
- **What it can be compared against**: README MAPE 1.48% vs 1.68% (AlexNet). Paper Table 5 (MSE / MAE / MAPE): Alexnet 0.0001311409 / 0.0061858603 / 1.6834715604, ResNet50 0.0030414662 / 0.0529725812 / 8.8635520935, Surya 0.0001260741 / 0.0045114677 / 1.4792510271.
  - **MAPE is not implemented anywhere in the repo.**
  - The paper's conclusion gives different MAE numbers from its own Table 5: "MAE of 0.0043 compared to 0.0065 for AlexNet and 0.0256 for ResNet50".
  - It is unclear whether Table 5 is in normalised or physical units. The `infer.py` metrics are on normalised [0, 1] spectra.
- **Defaults**: batch 1, workers 4, `dtype: float32`, `checkpoint_layers: []`, lr 0.0001, 20 epochs. `config_infer.yaml` sets `iters_per_epoch_train/valid: 2`.

## 7. CUDA-only calls (grep over `*.py`, excluding `.uvcache`)
- **Hard CUDA requirements**:
  - `surya/utils/distributed.py:40-45`: `assert torch.cuda.is_available()`, `init_dist("nccl", …)`, `torch.cuda.set_device`.
  - Every `finetune.py`: `device_ids=[torch.cuda.current_device()]` in DDP (AR:508, flare:471, wind:531, EUV:435).
  - `surya/utils/misc.py:77-86` `throughput()`: `.cuda(non_blocking=True)`, `torch.cuda.synchronize()`. This helper is unused.
- **Hard-coded `autocast(device_type="cuda")`**, which turns itself off on CPU and leaves fp32:
  - `tests/test_surya.py:273`;
  - `finetune.py`: AR:147/534, flare:193/498, wind:213/573, EUV:120/461.
- **Device-guarded, so they work on CPU**:
  - every `infer.py` (`torch.cuda.is_available()` check, `torch.amp.autocast(device_type=args.device)`);
  - `easy_inference` (`torch.compile`, `torch.cuda.synchronize`, memory stats, TF32 only when `device_type=="cuda"`);
  - `set_global_seed`'s `torch.cuda.manual_seed`, which is a safe no-op without CUDA.
- `pin_memory=True` in the test, the infer scripts and the fine-tune loaders produces only a warning on CPU, as the notebook outputs show.
- **Not present**: flash-attn, xformers, custom CUDA kernels, `.cuda()` in the model code. `F.scaled_dot_product_attention` appears in `surya/models/embedding.py:244` (perceiver embedding, not used by the released configs) and works on CPU.
- `pyproject.toml` sends torch, torchvision and torchaudio to the CUDA 12.6 wheel index on non-macOS (`[tool.uv.sources] … pytorch-cu126`). `uv sync` therefore pulls CUDA wheels even on this CPU-only machine. They still run on CPU.

## 8. Defects in the repo and its docs [derived from reading]
1. The main README tells you to run `python download_data.py`; the file is `download_sample_train_data.py`.
2. `downstream_examples/README.md` says "six key tasks" and lists four.
3. The AR and flare `download_data.sh` Python fallbacks have an unclosed `snapshot_download(` (SyntaxError).
4. The `surya-test` entry point references a `run_tests` function that does not exist.
5. AR `finetune.py` never loads the pretrained FM for `spectformer_lora` (§6.1).
6. The wind train/infer normalisation constants differ (§6.2–6.3).
7. Flare training metrics threshold logits at 0.5, while `infer.py` thresholds sigmoid at 0.5.
8. `--data_type` is unused in every `infer.py`. The README "Advanced Usage" commands with `--config_path ./config.yaml` raise KeyError or file-not-found.
9. The AR README promises ground truth in its figures and a `run_inference_example.sh`; neither exists.
10. The easy_inference README promises a `metrics/*` output that the script does not write.
11. The flare README gives the CSV column as `timestep`; the data and the code use `timestamp`.

## 9. Things I could not reach or verify
- GitHub REST API (releases list): blocked by the session proxy. `git ls-remote` worked.
- HF SuryaBench collection API: returned "Invalid username or password."
- `nasa-impact/Surya-bench` (the AWS registry documentation link): HTTP 401.
- Whether S3 and HF `.nc` pixel values are identical: not verified, because it would need large downloads.
- The actual RAM and runtime of a 4096² forward pass on this CPU: not measured, since nothing was run.
- The full contents of `train_index_surya_1_0.csv` (18.6 MB): not downloaded.
- Which data split the paper's downstream Tables 3–6 use: I found no explicit statement in the HTML.
