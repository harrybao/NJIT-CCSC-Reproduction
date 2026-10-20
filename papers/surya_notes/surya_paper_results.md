# Surya paper (arXiv 2508.14112): inventory of quantitative results

Compiled 2026-10-15. Every number below is copied from the source named next to it. Where I did arithmetic myself (consistency checks, size estimates), the text says "my arithmetic" or "my estimate".

## 0. Which version I read, and from where

| Item | Value | Source |
|---|---|---|
| Title | "Surya: Foundation Model for Heliophysics", Sujit Roy and 32 other authors | https://arxiv.org/abs/2508.14112 |
| Versions on arXiv | **v1** Mon, 18 Aug 2025 05:44:25 UTC (17,648 KB); **v2** Thu, 21 Aug 2025 16:53:36 UTC (17,517 KB). No v3 was listed on 2026-10-15. | abs page, "Submission history" |
| **Version used for this inventory** | **v2 (latest)**. Read as arXiv HTML (https://arxiv.org/html/2508.14112v2), checked against the v2 TeX source (https://arxiv.org/src/2508.14112v2, file `iclr2025_conference.tex`) and the v2 PDF (https://arxiv.org/pdf/2508.14112v2). Plot-only figures (Fig. 5, Fig. 7) were rendered from the PDFs in the TeX source (`figure/rollout_eval.pdf`, `figure/eve_emissions_20150113.pdf`). | |
| Template | ICLR 2025 style. The HTML still shows the "Anonymous authors / Paper under double-blind review" box. | v2 HTML |
| Copy on HF | `SuryaFM.pdf` in https://huggingface.co/nasa-ibm-ai4science/Surya-1.0 (18,347,597 bytes, PDF CreationDate 2025-08-20). Its numbers match arXiv v2 (I diffed every decimal number in `pdftotext` output; only reference DOIs differ). | HF model repo |

### v1 → v2 differences that affect numbers (diff of v1 and v2 HTML)
- **Table 2 header**: v1 "Loss (MAE)" → v2 "Loss (MSE)". The values are unchanged (0.594044030 and 0.337624282).
- **EUV / FISM sentence (Sec. 2.6.4)**: v1 says "FISM ... produces an MAPE of 3.4 %, which is 2.33 × worse than the Surya forecast". v2 says "FISM ... attains a MAPE of 1.5% and MSE of 0.00031 on the same dataset where Surya is tested, which is 2.46× worse than the Surya prediction".
- The section was renamed from "EUV spectra prediction" to "EUV spectra modeling", and the text now gives the band range "ranging from 5 nm to 35 nm".
- One author (Anne Spalding) and one affiliation were added. The conclusion and data-availability text were lightly edited.
- No table values changed between v1 and v2.

---

## 1. Index of every table and figure that carries numbers

| # | What | Experiment group | Has numbers? |
|---|---|---|---|
| Table 1 | SDO/AIA and HMI instrument properties | Data | yes (instrument specs) |
| Table 2 | Baseline scores, 1 h ahead (persistence, learned flow) | Pretraining baselines | yes |
| Table 3 | AR segmentation (UNet vs Surya) | Downstream | yes |
| Table 4 | Flare forecasting (AlexNet, ResNet50, Surya) | Downstream | yes |
| Table 5 | EUV spectra (AlexNet, ResNet50, Surya) | Downstream | yes |
| Table 6 | Solar wind (AlexNet, ResNet50, Surya) | Downstream | yes |
| Table 7 | Solar wind vs literature models | Downstream (literature comparison) | yes |
| Table 8 (App. A.1) | Architecture ablations and baselines (16 GPUs, 10k steps) | Ablations | yes |
| Table 9 (App. B.1) | MSE per lead time 1–12 h for each pretraining stage | Zero-shot forecasting | yes (data behind Fig. 5) |
| Fig. 3 | Signum-log transform and histograms | Data normalisation | qualitative only (15 G noise level, scale factor 10^-2) |
| Fig. 5 | MSE vs hours ahead (plot of Table 9) | Zero-shot forecasting | yes, via Table 9 |
| Fig. 7 | Integrated (per-channel summed) emission, 2015-01-13 flare | Zero-shot flare | plot only; no tabulated values |
| Figs. 1, 6, 8, 12, 13, 14 | Input / output / ground-truth images | Zero-shot flare (visual) | timestamps only |
| Figs. 9, 10 | 600×600 crops, 1 h forecast (AIA 171, HMI_m) | Zero-shot blurring | qualitative only |
| Fig. 11 | AR segmentation example (2014-02-01 08:12) | Downstream | qualitative only |
| Figs. 2, 4 | SDO channels; architecture diagram | — | no results |

The paper has **no scaling-law study** (no model-size or data-size sweep). Section 2.4.1 is called "Scaling" but covers only memory and engineering. There are **no error bars, seeds or repeated runs** for any Surya result. The only "±" values are literature WindNet numbers in Table 7.

---

## 2. Data (Sec. 2.1)

### Table 1 (copied exactly)
Caption: "Key properties of the SDO/AIA and SDO/HMI instruments. Cadence values refer to both instrument-native and standardized dataset cadence."

| Instrument | Resolution | Cadence (instr./dataset) | Dynamic range | Channels / Measurements |
|---|---|---|---|---|
| AIA | 1.2″ (≈725 km) | 12 s, 24 s / 12 min | 0–16,383 DN | 94, 131, 171, 193, 211, 304, 335, 1600 Å |
| HMI | 1.0″ (≈870 km) | 45 s, 12 min / 12 min | ±4,500 G (B), ±10^4 m/s (V) | B_x, B_y, B_z, B_los, V_los |

### Numbers stated in the text (Sec. 2.1.1–2.1.2)
- Source series: aia.lev1_euv_12s, aia.lev1_uv_24s, hmi.M_720s, hmi.B_720s, hmi.V_720s, from JSOC. Cadence 12 min. AIA co-aligned within ±2 minutes. Grid 0.6″/pixel. HMI native 0.5″/pixel, reprojected with bilinear interpolation.
- Coverage: **May 13, 2010 to December 31, 2024**. "about 2.9% data unavailable (18,261 out of 623,280 total timestamps) for hmi.M_720s series" (my arithmetic: 18,261/623,280 = 2.93%).
- Storage: netCDF, float32, data shape [13, 4096, 4096], lz4 compression ("≈ 30% improvement" in training performance), "about 630 MB" per file, **total "around 257 TB"**.
- **Pretraining train/test split**: observations from **2011 to 2019**, split by day-of-year (DOY). DOY 1–14 and 32–45 are buffers (excluded). **DOY 15–31 = test.** DOY ≥ 46 = train. Solar Cycle 25 data was deliberately excluded.
- Normalisation (Eq. 1): sign(X)·log1p(|10^-2 x|), then a per-channel standard scale with ε = 10^-8. HMI_m noise level 15 Gauss. "about 1/3 of the transformed scale would be occupied by data that is dominated by the inherent detector noise" without the 10^-2 scale.

**What the released index files actually contain** (my analysis of `core-sdo` CSVs from https://huggingface.co/datasets/nasa-ibm-ai4science/core-sdo):
- `train_index_surya_1_0.csv`: 345,721 rows (309,412 with `present=1`), 2011-02-15 00:00 to 2019-12-31 00:00, DOY 46–366 only. This matches the paper's train split.
- `valid_index_surya_1_0.csv`: 18,360 rows (17,710 present), 2011-01-15 to 2019-01-31, DOY 15–31 only. This is the paper's "test" period; the repo calls it "valid".
- The full-resolution files are in the public S3 bucket `s3://nasa-surya-bench` (anonymous HTTPS listing works; years 2010/ to 2024/). In the 2014/01 listing the mean object size is about 592 MB.

---

## 3. Pretraining: baselines, architecture numbers, protocol and compute

### Table 2 (copied exactly). "Baseline scores for 1 hour ahead forecasting. In model units."
| Baseline | Parameters | Loss (MSE) |
|---|---|---|
| Persistence | N/A | 0.594044030 |
| Learned flow | 642 | 0.337624282 |

- Persistence means X̂_{t+1} = X_t. The learned flow is an MLP with two linear layers, internal dimension 128, GELU, 642 parameters. It is applied with `F.grid_sample(..., mode="bilinear")` (Sec. 2.2.2, footnote 2).
- The Table 2 values are identical to the baseline rows of Table 8, so Table 2 probably comes from the same 16-GPU / 10k-step evaluation as Table 8. The paper does not say.

### Architecture and scaling numbers (Sec. 2.3, 2.4.1, App. A.1)
- 366M parameters. Table 8 gives 366.19 M, and the repo test logs "366.19 M total parameters".
- 2 spectral gating blocks, 8 long–short attention blocks and 1 decoder block. Embedding D = 1280, patch 16×16, N = 65,536 tokens. Input 13×2×4096×4096.
- Spectral weight W_c has H_f = 256, W_f = 129 and "84,541,440 real parameters" per block (my arithmetic: 256·129·1280·2 = 84,541,440 ✓).
- Two timestamps take 2×13×4096×4096×32 bit = **1.7 GB** (Eq. 17) (my arithmetic: 1.745 GB ✓).
- Tokenisation compression is 2×13×16×16 = 6,656 pixels per patch into 1,280 dimensions, "a factor of 5.2".
- Engineering: FSDP, mixed precision (bf16 inside the transformer, fp32 input/output and FFT) and gradient checkpointing.
- Sec. 2.4.1: "the use of the spectral gating layers leads to the same loss at 6% less GPU memory". The data is in Table 8.

### Pretraining protocol and **compute** (Sec. 2.4.2). These are the only compute figures in the paper.
| Phase | Steps | Hardware | Batch | LR / optimiser |
|---|---|---|---|---|
| Phase 1: 1 h ahead | **160,000** | **128 NVIDIA A100 GPUs** | 1 per GPU (effective 128) | cosine 10^-4 → 10^-5, no warm-up, grad clip 0.1, AdamW defaults |
| Rollout 2 steps | 20,000 | "Generally, we used 64 GPUs for rollout tuning" | — | constant 10^-5 |
| Rollout 3, 4, 5 steps | 4,000 each | 64 GPUs (as above) | — | 10^-6 |
| Ablations (Table 8) | 10,000 each | 16 GPUs | — | — |

- **No GPU-hours, wall-clock time or energy figures are reported anywhere** (I searched the TeX for "hours", "days", "GPU", "wall" and "node").
- Sec. 2.5.1: "Surya could be tuned up to 24 hours ahead ... when using an 80 GB A100 GPU". Data loading was the bottleneck.
- The README (`/home/user/upstream/surya/Surya/README.md`) repeats "Phase 1: One-step ahead forecasting (160k steps, 128 GPUs)" and "~257TB processed".
- The HF model card says "pretrained on 9 years (≈218 TB)" and "rollout tuning up to 12 hours". Both differ from the paper (257 TB; rollout tuning to 5 steps). See §8.

**Can it be re-run?** No. The repo (HEAD 78b827e = `v1.0.0-65-g78b827e`, so 65 commits after the v1.0.0 tag) contains **no pretraining or rollout-tuning script**. I checked all 107 commits for train/pretrain/rollout/ablation file names and found only `downstream_examples/download_sample_train_data.py`. The model, dataset and flow classes are present (`surya/models/helio_spectformer.py`, `surya/models/flow.py`, `surya/datasets/helio.py`). Only the final checkpoint `surya.366m.v1.pt` (1,800,384,309 bytes) is released.

---

## 4. Zero-shot forecasting (Sec. 2.5.1, Fig. 5, App. B.1, Table 9)

- **Data and split**: the paper says "validation loss (MSE) per lead time" (Table 9 caption). Fig. 5 calls it "MSE across all channels in model units". The test period defined in Sec. 2.1.2 is DOY 15–31 of 2011–2019. **The number of samples or initialisation times used is not stated.**
- **Inputs**: two timestamps 60 min apart. Autoregressive 1 h steps out to 12 h.
- **Metric**: MSE in normalised model units (Eq. 1), averaged over all 13 channels.
- **Checkpoints**: one column per pretraining stage. The "5 hours" column "corresponds to Surya as released".

### Table 9 (copied exactly; **bold** = best in row, as in the paper)
Caption: "Impact of rollout tuning. The table shows validation loss (MSE) per lead time for lead times up to 12 hours ahead (rows). Columns show persistence and flow baselines as well as Surya at various stages of pretraining. The 5 hour ahead tuned version (rightmost column) corresponds to Surya as released. The best score in each row is marked in bold."

| Lead (h) | Persistence | Flow | 1 hour (no rollout) | 2 hours | 3 hours | 4 hours | 5 hours |
|---|---|---|---|---|---|---|---|
| 1 | 0.59868 | 0.34204 | **0.20133** | 0.20153 | 0.20159 | 0.20169 | 0.20172 |
| 2 | 0.70594 | 0.40124 | 0.22738 | 0.22371 | 0.22352 | 0.22359 | **0.22351** |
| 3 | 0.77862 | 0.45497 | 0.24889 | 0.24182 | 0.23928 | 0.23902 | **0.23889** |
| 4 | 0.83790 | 0.50416 | 0.26662 | 0.25759 | 0.25213 | 0.25125 | **0.25078** |
| 5 | 0.89444 | 0.55291 | 0.27976 | 0.26957 | 0.26295 | 0.26141 | **0.26064** |
| 6 | 0.94441 | 0.59671 | 0.29362 | 0.28111 | 0.27366 | 0.27140 | **0.26995** |
| 7 | 0.99986 | 0.64647 | 0.31217 | 0.29536 | 0.28557 | 0.28113 | **0.27854** |
| 8 | 1.04645 | 0.68856 | 0.32528 | 0.30458 | 0.29519 | 0.29018 | **0.28583** |
| 9 | 1.09013 | 0.72833 | 0.33984 | 0.31209 | 0.30361 | 0.29777 | **0.29216** |
| 10 | 1.12573 | 0.76106 | 0.35257 | 0.31934 | 0.31181 | 0.30522 | **0.29868** |
| 11 | 1.14940 | 0.78280 | 0.37037 | 0.33154 | 0.32443 | 0.31638 | **0.30789** |
| 12 | 1.16841 | 0.79964 | 0.38517 | 0.34304 | 0.33786 | 0.32791 | **0.31651** |

- **Fig. 5** (`figure/rollout_eval.pdf`) plots these curves. Left panel: all 7 series, y from 0.2 to 1.2. Right panel: Surya stages only, y from 0.200 to 0.375. The x-axis is "Hours ahead [h]" from 1 to 12, and the y-axis label is "MSE loss [Model units]". The figure prints no other numbers.
- Text (Sec. 2.5.1): "Rollout tuning improves the performance at 12 hours ahead by **10.9, 12.3, 14.9, and 17.8 %** respectively when compared against the state of the model after phase 1." My arithmetic from Table 9 row 12 gives 10.94 / 12.28 / 14.87 / 17.83 %, which is consistent. The conclusion says "up to 17.8% at 12 hours lead time".
- App. B.1: the best 1 h model is the one before phase 2 (0.20133 vs 0.20172 for the released model).
- **Re-runnable?** Only partly.
  - The **"5 hours" column** can be re-evaluated with the released weights plus S3 data for DOY 15–31 of 2011–2019. That split has 17,710 present files, roughly 10 TB at about 0.59 GB per file (my estimate). The paper's sample count is unknown, so an exact match cannot be confirmed.
  - The **Persistence** column needs data only.
  - The **Flow** column needs a trained flow model. `HelioFlowModel` is in the repo, but there are no weights and no training script.
  - The **1 h–4 h** columns use intermediate checkpoints that were **not released**.
  - The repo's `easy_inference/` (S3 download, rollout, `prediction.nc`) and `compare_prediction_groundtruth.ipynb` (MSE/RMSE/MAE per channel and per step) are the closest released tooling. Note that they compute MSE on whatever date window you pick, not on the Table 9 protocol.

### Repo-only number (not in the paper)
`tests/test_surya.py` covers 2014-01-07 with 8 initialisations (In0 15:00 … 16:24 every 12 min) and 2 steps each. This is the Fig. 14 setting. The test hard-codes `REFERENCE_LOSS` = [0.24890851974487305, 0.31166040152311325, 0.32497961074113846, 0.32198894023895264, 0.3090871721506119, 0.3190929591655731, 0.36053016781806946, 0.33693042397499084] and asserts |mean − ref mean| < 1.0e-4. The README's expected output is "Local loss 0.31665" (my arithmetic: the array mean is 0.316647). The data is `nasa-ibm-ai4science/Surya-1.0_validation_data`: 23 files, 2014-01-07 15:00–19:24, 15.43 GB in total, each file about 670 MB.

---

## 5. Zero-shot "visual flare prediction" and blurring (Sec. 2.5.2–2.5.3, App. B.2)

These results are qualitative. The only numbers are dates, times and figure axes.

| Figure | Event / channel | Init / outputs (from panel titles) | Split status per paper |
|---|---|---|---|
| Fig. 1 | M-class flare 2015-01-13, AIA 94 Å | In 03:12, 04:12 → Out/GT 05:12, 06:12 | DOY 13: in the "temporal buffer", treated as test-like |
| Fig. 6 | St. Patrick's Day event, AIA 131 Å | In 2015-03-14 23:12, 2015-03-15 00:12 → Out/GT 01:12, 02:12 | **training data** (paper says so) |
| Fig. 7 | 2015-01-13 integrated emission, 8 AIA channels | composite of 2-step runs, about 01:48–06:12 UTC; y-axis scale factors 1e7 (AIA94), 1e8 (131, 304, 335, 1600), 1e9 (171, 193, 211) | buffer |
| Fig. 8 | 2015-01-13 overview, AIA 94 | 8 inits, In0 01:48 … 03:12 every 12 min, 2 steps (last output 06:12) | buffer |
| Fig. 12 | 2014-01-07, AIA 94 | In 16:24, 17:24 → 18:24, 19:24 | buffer (DOY 7) |
| Fig. 13 | 2014-01-07, AIA 94 | In 15:24, 16:24 → 17:24, 18:24 | buffer |
| Fig. 14 | 2014-01-07 overview, AIA 94 | 8 inits, In0 15:00 … 16:24, 2 steps (last output 19:24) | buffer |
| Figs. 9, 10 | 2014-01-07 1 h forecast crops, 600×600 px, AIA 171 and HMI_m | — | buffer |

- Fig. 7 text: for all channels except 1600 Å, "the integrated output of Surya tracks the ground truth closely", especially around the rapid changes near 04:24 UTC.
- Rendered Fig. 7 shows (visually, no values) that Surya's AIA 171/193/211/304/1600 sums sit below the ground truth after onset.
- Fig. 8, row 4 (Out 0 = 04:24) shows a visibly brighter, hazier output panel. This is my observation; the paper does not comment on it.
- Sec. 2.5.3: a CRPS noise-injection variant was tried. It gave "improved detail" but "occasional token-level artifacts". No numbers are given.
- **Checkpoint**: not stated. It is presumably the released model.
- **Compute**: not stated.
- **Re-runnable?** Mostly yes, with the released weights.
  - 2014-01-07 (Figs. 9, 10, 12–14): the data is exactly `Surya-1.0_validation_data` (15.43 GB), and the repo test runs it.
  - 2015-01-13 (Figs. 1, 7, 8) and 2015-03-15 (Fig. 6): the data is in S3 (about 23 and 4 files at 12-min cadence, roughly 0.6–0.67 GB each; my estimate).
  - No script for the Fig. 7 integrated-emission plot is released. It would need about 10 lines of your own code (per-channel sum of inverse-transformed outputs).

---

## 6. Ablations (App. A.1, Table 8)

### Table 8 (copied exactly)
Caption: "Architecture ablations & baselines. The table shows MSE loss for 1 hour ahead forecasting in model units of equation 1. All baselines and ablations in this table – including the Surya configuration – were trained on 16 GPUs for 10,000 gradient descent steps."

| Type | Configuration | Parameters | Memory usage [MiB] | Loss (MSE) |
|---|---|---|---|---|
| Baseline | Persistence | N/A | N/A | 0.594044030 |
| | Learned flow | 642 | 24976 | 0.337624282 |
| Ablation | Single timestamp | 361.93 M | 55411 | 0.228553504 |
| | No spectral gating | 210.39 M | 59823 | 0.219618767 |
| | Perceiver | 351.52 M | 52721 | 0.234643415 |
| Surya | | 366.19 M | 56247 | 0.219778508 |

- Text: "Using two timestamps as input improves performance by **3.8%**". My arithmetic from Table 8 gives 3.84%, which is consistent.
- Text: "No spectral gating" gives a "virtually identical loss, yet at 6% reduced GPU memory consumption". Per the table, it is Surya *with* spectral gating that uses less memory: (59823−56247)/59823 = 5.98% (my arithmetic). "No spectral gating" also has the slightly lower loss.
- The Perceiver run had to use a 32×32 token size because of memory.
- "Flow model + transformer residual" ablation: "we found this to perform worse than simply regressing directly onto the target". **No number is given.**
- Data and split are not restated; presumably the same 2011–2019 split.
- **Re-runnable?** No. There is no training script. The model code has switches for `time_embedding.type: perceiver`, `n_spectral_blocks` and the number of input timestamps, so the configurations can be expressed. However, each run needs 16 GPUs × 10k steps on the 257 TB dataset, and no ablation checkpoints are released.

---

## 7. Downstream tasks (Sec. 2.6)

Common setup in the paper: LoRA (Eq. 18–19) plus a head. Pooling options are average, max, attention, transformer or global class token. The paper says LoRA gives "our strongest fine-tuning results". **The paper does not state, for any downstream task:**
- the time split used for the reported numbers (train/validation/test, or 2020–2024),
- the number of evaluation samples,
- the downstream compute (GPUs, hours),
- the LoRA rank,
- seeds or variance.

The split, checkpoint and config details below therefore come from the released repo and HF artefacts, and they may not be what produced the paper's numbers.

The SuryaBench HF datasets all share the same split convention (from the dataset cards; I verified it on the CSVs):
- train: Feb 15 (DOY 46) to Dec 31, 2010–2019
- validation: Jan 15–31 (DOY 15–31), 2011–2019
- leaky_validation: DOY 1–14 and 32–45
- test: all of 2020–2024

The repo `finetune.py` scripts validate on `validation.csv`; none of them reference `test.csv` (my grep). The `infer.py` scripts only print or plot a few random samples (default `--num_samples 3`).

### 7.1 Active-region (PIL) segmentation (Sec. 2.6.2). Table 3 (copied exactly)
Caption: "AR Segmentation results comparing baseline models with Surya"

| Model | Params | IoU | Dice Coeff |
|---|---|---|---|
| Unet | 9.2 M | 0.688 | 0.801 |
| Surya | 4.1 M | 0.768 | 0.853 |

- **Data (paper)**: ARPIL dataset built from HMI LOS magnetograms (4096×4096), following Cai et al. (2020):
  - ±50 G thresholds, regions under 100 px removed, 10-px dilation, intersection of the polarity maps gives the PILs;
  - "yielding **119,454** binary masks spanning January 2011–December 2024".
  - Table 3 "Params" counts trainable parameters ("With just 4.1M trainable parameters").
- Example figure: Fig. 11, 2014-02-01 08:12.
- **Split**: not stated in the paper. The HF `surya-bench-ar-segmentation` card says 128,352 instances, May 13 2010 to Dec 31 2024. My count of rows with `present=1`: train 70,823, validation 3,477, test 41,910, leaky 5,753, total 121,963. None of these equals 119,454.
- **Checkpoint**: Surya backbone with LoRA. Repo `ar_segmentation/config.yaml` has `job_id: AR-segmentation_lora_r32_a64_d0.1`, LoRA r=32, α=64, `pretrained_path: ../../data/Surya-1.0/surya.366m.v1.pt`, bf16, 15 epochs × 2000 iterations, lr 0.001. Fine-tuned weights are released: `nasa-ibm-ai4science/ar_segmentation_surya/ar_segmentation_weights.pth` (1,784,013,352 bytes).
- **Compute**: not reported.
- **Baseline**: UNet code is in the repo (`models.py`: `UNet`, `unet_embed_dim: 128`, `unet_blocks: 3`). No UNet weights are released.
- **Re-runnable?**
  - Fine-tuning: yes in principle (code, labels at 1.31 GB `data.tar.gz`, inputs from S3, LoRA config), but it needs a GPU.
  - Evaluation of the released weights: possible. However, **no script computes IoU/Dice over a whole split**. IoU/Dice appear only as loss classes in `finetune.py`, and `infer.py` only visualises samples. So reproducing 0.768/0.853 needs your own evaluation loop and a guess at the split.
  - The AR model card on HF repeats Table 3 exactly.

### 7.2 Solar flare forecasting (Sec. 2.6.3). Table 4 (copied exactly)
Caption: "The results of flare forecasting evaluation comparing baseline models used in literature Pandey et al. (2023); Pandey et al. (2024) with finetuned Surya-based ones"

| Model | TSS | HSS | F1 |
|---|---|---|---|
| AlexNet | 0.358 | 0.398 | 0.454 |
| ResNet50 | 0.018 | 0.028 | 0.055 |
| Surya | 0.436 | 0.522 | 0.561 |

- **Task**: binary. Positive if the peak X-ray flux of the strongest flare in [t, t+24h) exceeds θ_max = 10^-5 W/m² (M or X class). Metrics are TSS, HSS and F1 (Eq. 21–23).
- The paper notes these are "proof-of-concept studies and are not optimized for end-to-end or operational forecasting use".
- It also says "despite using fewer trainable parameters", but **no parameter counts are given** for this task.
- **Data and split**: not stated in the paper. HF `surya-bench-flare-forecasting` has 128,328 hourly instances; `label_max` = 1 if ≥ M1.0. My counts of `label_max=1`:
  - train 9,051 / 74,760
  - validation 400 / 3,672
  - test 12,903 / 43,848
  - leaky 901 / 6,048
- **Checkpoint/config (repo)**: `solar_flare_forcasting/config.yaml`:
  - 2 input timestamps (−60, 0 min), global class token, LoRA r=8 α=8, 20 epochs × 2000 iterations, lr 1e-4;
  - validation on `validation.csv`, classifier threshold 0.5 (`DistributedClassificationMetrics(threshold=0.5)`);
  - TSS/HSS are computed in `metrics.py`.
- Weights: `nasa-ibm-ai4science/solar_flares_surya/solar_flare_weights.pth` (1,794,022,584 bytes; the model card is empty apart from the license).
- **Compute**: not reported.
- **Baselines**: AlexNet and ResNet50 classifier code is in `models.py` (switch via `model_type`). No baseline weights are released.
- **Re-runnable?**
  - Fine-tuning plus validation metrics: yes in principle (GPU needed). The validation split needs about 3,672 hourly SDO files, roughly 2.2 TB from S3 (my estimate).
  - Evaluating the released weights on a full split: no ready script exists, because `infer.py` prints a few samples only. TSS/HSS would need a loop that reuses `metrics.py`.

### 7.3 EUV spectra modelling (Sec. 2.6.4). Table 5 (copied exactly)
Caption: "EUV spectra modeling results comparing baseline models with Surya"

| Model | MSE | MAE | MAPE |
|---|---|---|---|
| Alexnet | 0.0001311409 | 0.0061858603 | 1.6834715604 |
| ResNet50 | 0.0030414662 | 0.0529725812 | 8.8635520935 |
| Surya | 0.0001260741 | 0.0045114677 | 1.4792510271 |

- **Task**: map SDO images at time t to a 1343-bin EVE irradiance vector ("ranging from 5 nm to 35 nm").
- **Literature comparisons in the text (v2)**:
  - FISM "attains a MAPE of 1.5% and MSE of 0.00031 on the same dataset where Surya is tested, which is 2.46× worse than the Surya prediction" (my arithmetic: 0.00031/0.0001260741 = 2.459 ✓).
  - Szenicer et al. (2019) "obtain a median MAPE of 1.6% and max MAPE of 4.6%".
  - The v1 FISM figures were MAPE 3.4% and 2.33×.
- **Data and split**: not stated in the paper. The two released sources disagree with each other:
  - HF `euv-spectra` card: SDO/EVE MEGS-A, about 6.5–33.3 nm, 12-min cadence, May 2010 to May 2014, 189,344 instances. Splits: train DOY ≥ 46 of 2010–2014; validation Jan 15–31 of 2012 and 2013; test Jan 15–31 of 2011 and 2014.
  - Repo README: a "70-15-15" split. Repo config uses `archive/AIA_EVE_temporally_stratified.nc` (933,907,814 bytes) with `train_time` / `val_time`.
- **Checkpoint/config (repo)**: `euv_spectra_prediction/config.yaml` has `freeze_backbone: True`, **no `use_lora` key**, a class token, 20 epochs, lr 1e-4. The paper says downstream results use LoRA, so it is unclear which configuration produced Table 5.
- Weights: `nasa-ibm-ai4science/euv_spectra_surya/euv_spectra_weights.pth` (1,779,722,720 bytes; the model card is empty apart from the license).
- **Metrics in code**: `finetune.py` computes MAE, RMSE and R² (**no MAPE**). `infer.py` prints MAE, RMSE, R² and spectral correlation for a few samples. The repo README example output (3 samples) is "Mean Absolute Error 0.005164, Root Mean Square Error 0.008369, R² Score 0.985208". That example is not a paper number.
- **Compute**: not reported.
- **Baselines**: AlexNet and ResNet50 code is in the repo; no weights.
- **Re-runnable?** Partly. The code, labels and weights are released. MAPE needs your own code. The split and the LoRA-vs-frozen choice are ambiguous.

### 7.4 Solar wind speed forecasting (Sec. 2.6.5). Tables 6 and 7 (copied exactly)
Text: "AlexNet, trained on 18k steps, ResNet50 for 20k steps, and Surya for 10k steps."

Table 6 caption: "Solar Wind Prediction results comparing baseline models with Surya"

| Model | RMSE (km s⁻¹) | MAE (km s⁻¹) | Validation MSE (km² s⁻²) |
|---|---|---|---|
| AlexNet | 118.6 | 95.7 km | 13839.49 |
| ResNet50 | 93.76 | 74.65 | 8547.924 |
| Surya | 75.92 | 58.06 | 5698.62 |

("95.7 km" is printed with the stray unit in the source TeX.)

Table 7 caption: "Solar wind speed prediction results comparing baseline models with Surya". Surya is printed in bold even though some rows are lower.

| Model | RMSE (km s⁻¹) |
|---|---|
| Surya | 75.92 |
| WindNet 193 (1,4) Upendran et al. (2020) | 84.33 ± 2.31 |
| WindNet 211 (1,4) Upendran et al. (2020) | 85.94 ± 4.67 |
| 27-day Persistence Ahn et al. (2025) | 89 |
| SDO + solar wind speed model Ahn et al. (2025) | 68.6 |
| CH features + solar wind speed model: Collin et al. (2025) | 61 – 71 |
| MHD selected rotations Mayank et al. (2022) | 47 – 85 |
| ESWF using CH features: Reiss et al. (2016) | 108.2 |
| WSA model using photospheric B maps: Reiss et al. (2016) | 99.5 |

- **Task**: predict the scalar speed at L1 at t + 4 days from SDO images at t.
- **Data (paper)**: hourly speed "from 2010-01-01 through 2023-12-31", range 2.4×10² to 8.8×10² km/s (Gloeckler 2023 / ACE).
- The paper says Table 7 is a "qualitative comparison" over different time periods. Models that beat Surya use past wind-speed measurements.
- **Split**: not stated. Table 6 says "Validation MSE". HF `Surya-bench-solarwind` (OMNI) covers May 13 2010 to Dec 31 2024 with 119,225 hourly rows. My row counts: train 69,957, validation 3,429, test 39,975, leaky 5,864.
- **Checkpoint/config (repo)**: `solar_wind_forcasting/config.yaml`:
  - 1 input timestamp, `ds_time_delta_in_out: "4D"`, tolerance 1h;
  - normalisation `ds_scaler: [2.61, 0.09]`;
  - LoRA r=64 α=32, 20 epochs × 2000 iterations, lr 0.001;
  - validates on `validation.csv`.
- Weights: `nasa-ibm-ai4science/solar_wind_surya/solar_wind_weights.pth` (1,805,655,558 bytes; the model card is empty apart from the license).
- **Compute**: steps only (18k / 20k / 10k); no GPUs or hours reported.
- **Re-runnable?**
  - Surya rows of Table 6: code, labels and weights are released; the GPU fine-tune is feasible in principle. `finetune.py` computes MSE/RMSE; `infer.py` prints MAE/RMSE/R² for a few samples.
  - AlexNet/ResNet50: code exists, but there are no weights.
  - Table 7 literature rows are copied from other papers and are not re-runnable here.

### 7.5 Other downstream tasks mentioned but with no results
- The introduction lists "active region emergence prediction" as a demonstrated application. **The paper reports no AR-emergence result.** HF does host `nasa-ibm-ai4science/ar_emergence` (train 37 / validation 9 / test 13 ARs; about 23 MB zip). That belongs to SuryaBench, not this paper.
- `surya-bench-coronal-extrapolation` (36.18 GB) is also SuryaBench-only.

---

## 8. Per-experiment summary matrix

| Experiment | Paper source | Data / split (paper) | Metric | Checkpoint | Compute (paper) | Released artefacts | Re-runnable from release? |
|---|---|---|---|---|---|---|---|
| Phase-1 pretraining | Sec. 2.4.2 | SDO 2011–2019, train DOY ≥ 46 | MSE (model units) | — | 160k steps, 128 A100, bs 128 | final weights only; no training script | **No** |
| Rollout tuning | Sec. 2.4.2 | same | rollout MSE | stages 2–5 h | 20k + 3×4k steps, ~64 GPUs | final (5 h) only | **No** |
| Baselines (Table 2) | Sec. 2.2.2 | same (presumably the Table 8 run) | MSE | persistence; 642-param flow | (16 GPUs, 10k steps per Table 8) | flow code, no weights or script | Persistence: yes (data only). Flow: needs your own training |
| Zero-shot rollout (Table 9 / Fig. 5) | App. B.1 | "validation", test DOY 15–31, 2011–2019; N not given | MSE per lead 1–12 h | 1 h…5 h stages; 5 h = released | — | released 5 h weights, S3 data (~10 TB for the full split, my estimate), easy_inference | 5 h and persistence columns only, and without the paper's sample list |
| Ablations (Table 8) | App. A.1 | same | 1 h MSE, memory MiB | 4 ablation models | 16 GPUs × 10k steps each | none | **No** |
| Visual flares (Figs. 1, 6–8, 12–14) and blur (Figs. 9–10) | Sec. 2.5.2–2.5.3 | 2014-01-07, 2015-01-13 (buffer), 2015-03-15 (train) | visual; integrated emission | presumably released | — | weights; 2014-01-07 data on HF (15.43 GB); others on S3; repo test = Fig. 14 setting | **Yes**, qualitatively (Fig. 7 plot code not released) |
| AR segmentation (Table 3) | Sec. 2.6.2 | ARPIL, 119,454 masks Jan 2011–Dec 2024; split not stated | IoU, Dice | Surya + LoRA (repo r32/α64) | not reported | FT weights, labels, finetune code, UNet code | Fine-tune: GPU only. Eval script for IoU/Dice: not released |
| Flare forecasting (Table 4) | Sec. 2.6.3 | ≥ M1.0 in 24 h; split not stated | TSS, HSS, F1 | Surya + LoRA (repo r8/α8) | not reported | FT weights, labels, finetune + metrics code, AlexNet/ResNet code | Fine-tune/validate: GPU only. No full-split eval script |
| EUV spectra (Table 5) | Sec. 2.6.4 | EVE 1343 bins; split not stated | MSE, MAE, MAPE | repo config: frozen backbone, no LoRA (conflicts with paper) | not reported | FT weights, EVE data (12.06 GB repo; 934 MB file used) | Partly; MAPE not in code |
| Solar wind (Tables 6–7) | Sec. 2.6.5 | OMNI/ACE hourly, +4 d; "Validation MSE" | RMSE, MAE, MSE | Surya + LoRA (repo r64/α32) | steps only (Surya 10k) | FT weights, labels, code | Surya rows: GPU only. Table 7: literature |

---

## 9. Inconsistencies and red flags found while copying

1. **The conclusion calls the baselines MAE but the tables say MSE.** The conclusion says the model "significantly outperforms persistence (MAE ≈ 0.59) and learned-flow baselines (MAE ≈ 0.34)". Tables 2 and 8 in v2 label these values MSE; v1 Table 2 said MAE. The conclusion was not updated.
2. **EUV numbers in the conclusion do not match Table 5.**
   - Conclusion: "MAE of 0.0043 compared to 0.0065 for AlexNet and 0.0256 for ResNet50".
   - Table 5: 0.0045114677, 0.0061858603 and 0.0529725812.
3. **Persistence and flow at 1 h differ between tables.**
   - Table 2/8: 0.594044030 and 0.337624282.
   - Table 9, 1 h row: 0.59868 and 0.34204.
   - The evaluation sets or protocols are evidently different, but neither is described.
   - Surya's 1 h MSE is 0.219778508 in Table 8 (16 GPUs, 10k steps) vs 0.20133 in Table 9 (full training).
4. **The 6% memory sentence in App. A.1 can be misread.** Table 8 shows "No spectral gating" uses **more** memory (59823 vs 56247 MiB) and has a marginally **lower** loss (0.219618767 vs 0.219778508).
5. **Solar wind RMSE does not equal √(Validation MSE)** (my arithmetic):
   - Surya: √5698.62 = 75.49 vs RMSE 75.92
   - AlexNet: √13839.49 = 117.64 vs 118.6
   - ResNet50: √8547.924 = 92.45 vs 93.76
   - So the RMSE and MSE columns were probably computed on different sets or aggregations.
6. **No downstream split is stated.** None of the downstream tables names its split, sample count or seed, and there are no error bars. The repo fine-tune scripts only validate on the DOY 15–31 "validation" split. The 2020–2024 "test" splits on HF are never mentioned in the paper.
7. **Dataset sizes and periods disagree across sources.**
   - Total SDO size: paper 257 TB; HF model card ≈218 TB ("9 years"); core-sdo card ~360 TB.
   - File size: 630 MB (paper) vs ~570 MB (card) vs about 592 MB (S3 Jan-2014 mean, my measurement).
   - Solar wind period: 2010-01-01 to 2023-12-31 (paper) vs May 2010 to Dec 2024 (HF).
   - EVE band range: 5–35 nm (paper) vs ~6.5–33.3 nm (HF/repo).
   - ARPIL masks: 119,454 (paper) vs 128,352 instances (HF card).
   - Downstream READMEs say "Temporal coverage: 2011-2014" for flare and AR, which matches none of the above.
8. **Rollout horizon**: the HF model card says "rollout tuning up to 12 hours". The paper trains rollout up to 5 steps (5 h) and evaluates to 12 h.
9. **Flare "improvement" claims do not appear in the paper.**
   - IBM blog: "16% improvement in solar flare classification accuracy".
   - HF card: "surpasses existing benchmarks by 15%".
   - Repo README: TSS "22%" (my arithmetic: (0.436−0.358)/0.358 = 21.8%).
   - The paper reports no accuracy at all. The blog also says spectral gating "cut memory costs by 5%"; the paper says 6%.
10. **Repo README improvement column** (from "Performance Benchmarks"): TSS 22%, IoU 12%, RMSE 19%, MAPE 12% ("1.48%" vs "1.68% (AlexNet)"). These follow from the paper's tables (my arithmetic: 21.8, 11.6, 19.0, 12.1%).
11. **Broken citations**: the FISM, EVE, SORCE and XPS citations render as "?" in the v2 PDF and as raw keys ("Chamberlin2020", "Woods2010") in the HTML.
12. **EUV config conflicts with the paper**: the repo's EUV config fine-tunes with a frozen backbone and no LoRA, while the paper attributes its best downstream results to LoRA.
13. **AR emergence is announced but not reported** (see §7.5).
14. **Table 7 bolds Surya (75.92)** although three literature rows are lower; the text explains why.

---

## 10. Released-artefact inventory relevant to re-running (sizes from the HF API, 2026-10-15)

| Artefact | Size | Notes |
|---|---|---|
| `nasa-ibm-ai4science/Surya-1.0` | `surya.366m.v1.pt` 1,800,384,309 B; `config.yaml`; `scalers.yaml` 3,835 B; `SuryaFM.pdf` 18,347,597 B | final (5 h rollout) checkpoint only |
| `solar_flares_surya` | `solar_flare_weights.pth` 1,794,022,584 B | card: license only |
| `ar_segmentation_surya` | `ar_segmentation_weights.pth` 1,784,013,352 B | card repeats Table 3 |
| `solar_wind_surya` | `solar_wind_weights.pth` 1,805,655,558 B | card: license only |
| `euv_spectra_surya` | `euv_spectra_weights.pth` 1,779,722,720 B | card: license only |
| dataset `Surya-1.0_validation_data` | 23 × ~670 MB = 15.43 GB | 2014-01-07 15:00–19:24 (repo test / Fig. 14) |
| dataset `core-sdo` | 363.38 GB on HF | indices (`train_index_surya_1_0.csv` 18.6 MB, `valid_index_surya_1_0.csv` 0.96 MB); 15 `infer_data/*.nc` (~0.67 GB each); `test_splitted_tars` 175.8 GB and `validate_splitted_tars` 177.5 GB (January 2011 samples: 352 and 350 files). Full 2010–2024 archive is in `s3://nasa-surya-bench` (public). |
| `surya-bench-flare-forecasting` | 8.76 MB | labels only |
| `surya-bench-ar-segmentation` | 1.31 GB | masks `data.tar.gz` |
| `Surya-bench-solarwind` | 5.33 MB | labels only |
| `euv-spectra` | 12.06 GB | config uses `AIA_EVE_temporally_stratified.nc` (0.93 GB) |
| Not released anywhere I could find | — | intermediate rollout-stage checkpoints (1–4 h); flow-baseline weights; ablation models; AlexNet/ResNet50/UNet baseline weights; pretraining/rollout/ablation training scripts; full-split evaluation scripts for Tables 3–6; the list of samples used for Table 9 |

---

## 11. Gaps and what I could not reach

- **Everything I needed was reachable**: arXiv abs/HTML/PDF/source for v1 and v2, the HF API and cards, the S3 listing, and the repo clone.
- **Not downloaded** (over the 50 MB rule): any model weights; any SDO `.nc` file; the EUV data; the AR masks. No results were re-run for this note. Everything here is the paper's numbers plus metadata.
- The SuryaBench paper (arXiv 2508.14107) was **not** read for this note. Split definitions above come from the HF dataset cards and CSVs.
- I did not check the cited literature papers (Pandey 2023/2024, Upendran 2020, Ahn 2025, Collin 2025, Mayank 2022, Reiss 2016, Szenicer 2019, Chamberlin 2020) to verify the Table 7 and EUV comparison values. They are copied as printed in Surya v2.

Working copies (scratch, not part of the deliverable):
- Paper text: `/tmp/claude-0/-home-user-NJIT-CCSC-Reproduction/cafab427-273b-538d-9e6f-5e9528085d79/scratchpad/arxiv/v2.txt`, `/tmp/claude-0/-home-user-NJIT-CCSC-Reproduction/cafab427-273b-538d-9e6f-5e9528085d79/scratchpad/arxiv/v1.txt`
- TeX source: `/tmp/claude-0/-home-user-NJIT-CCSC-Reproduction/cafab427-273b-538d-9e6f-5e9528085d79/scratchpad/arxiv/src/iclr2025_conference.tex`
- Split CSVs: `/tmp/claude-0/-home-user-NJIT-CCSC-Reproduction/cafab427-273b-538d-9e6f-5e9528085d79/scratchpad/hf/csv/`
