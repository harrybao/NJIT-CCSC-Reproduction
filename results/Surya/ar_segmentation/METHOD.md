# Active-region (PIL) segmentation: method of the partial reproduction

This is a partial reproduction. The released fine-tuned Surya segmentation model is scored on a small, seeded sample of the benchmark images on a CPU. The result can show whether the paper's number is plausible for these images. It cannot show what the full split would give. Script: `scripts/Surya/downstream/ar_segmentation_subset_eval.py`.

## What is evaluated

The model is the released checkpoint `nasa-ibm-ai4science/ar_segmentation_surya`, file `ar_segmentation_weights.pth`, revision `73a6de53`, 1,784,013,352 bytes, SHA-256 `ad391831…56ad`. It is the Surya-1.0 backbone with PEFT LoRA adapters (rank 32, alpha 64, on q/k/v/out_proj, fc1 and fc2) and a linear unembedding head with one output channel. The checkpoint's pickle header holds 40 LoRA tensors with 4,096,000 parameters, which matches the "4.1 M" trainable parameters in the paper.

For each time t the model sees the 13-channel SDO frame at t, normalised with the released scalers, and produces a 4096×4096 map of logits. The target is the `union_with_intersect` mask of the SuryaBench ARPIL dataset (`nasa-ibm-ai4science/surya-bench-ar-segmentation`, revision `615a7df4`) divided by 255. That mask marks active regions that contain a polarity inversion line, built from HMI line-of-sight magnetograms.

The code that touches the numbers is the authors' code, imported unchanged from the clone at commit 78b827e:

- `ArDSDataset` from `dataset.py`, with the arguments `finetune.get_dataloaders` passes for its validation set in `config.yaml`;
- `custom_collate_fn`, `IoULoss` and `DiceLoss` from `finetune.py`;
- `get_model` and `apply_peft_lora` from `infer.py`, using `config_infer.yaml`;
- the checkpoint-loading steps of `infer.load_model`, ending in `load_state_dict(strict=True)`.

## Reference numbers

Table 3 of arXiv 2508.14112 v2 (Sec. 2.6.2) gives IoU 0.768 and Dice 0.853 for Surya, and 0.688 and 0.801 for a UNet. The paper does not give the split, the number of images, the threshold, or how scores were averaged. It also gives no equation. The repository has no code that computes either number: `infer.py` only plots the sigmoid output, and `finetune.py` trains and validates with `BCEWithLogitsLoss`, never calling `IoULoss` or `DiceLoss`.

One clue to the averaging comes from my own arithmetic. For a single image, and for scores pooled over the pixels of many images, Dice always equals 2·IoU/(1+IoU). An IoU of 0.768 would then give a Dice of 0.869, not 0.853; the UNet's 0.688 would give 0.815, not 0.801. So the published pairs cannot be pooled scores. They are consistent with an average of per-image (or per-batch) scores: averaging makes mean Dice smaller than 2·(mean IoU)/(1+mean IoU), because that function is concave.

The primary comparison is therefore the per-image mean. Each image's score uses a hard mask, with sigmoid(logit) > 0.5. Pooled scores and soft scores are reported alongside it.

## Splits and sample design

The test split (all of 2020–2024) is the primary sample, because it was used neither for training nor for picking the released epoch. The validation split (January 15–31 of 2011–2019) is the one `finetune.py` validates on, so it may have been used to choose the released weights; the archive inside the checkpoint is named `epoch_40_loss_0.002`. The validation result is reported separately, as a secondary number.

**Candidates.**
- Test: every row of `test.csv` with `present == 1`, which is 41,910 hourly times.
- Validation: the 3,477 rows of `validation.csv` with `present == 1`, reduced to the 3,357 times whose t and t+60 min are both in core-sdo's `valid_index_surya_1_0.csv`. That is the filter `ArDSDataset` applies when it is built on that index, as `finetune.py` does.

**Strata and blocks.** The sample is spread evenly over years. A remainder goes to years drawn with the seed; with seed 0 and n = 10 that year is 2018. Within a year, the time-ordered candidates are cut into as many equal blocks as the year has images, and one image is drawn per block in a seeded random order.
- Test: 30 images, 6 per year, so each block covers about two months.
- Validation: 10 images, one per year plus a second in 2018.

**Checks and reserves.** A candidate is accepted only if it passes three checks. The frame at t must exist on S3, and so must the frame at t+60 min. The second check exists because the authors' dataset accepts t only when t+60 is in its SDO index, although it never reads that frame. Third, the HDF5 header of the frame at t, read with HTTP range requests, must list all 13 channels at 4096×4096. Each block also gets a reserve, chosen in the same way. The reserve is used only if the primary's mask is missing from the archive, or the primary fails at run time.

**Dry-run outcome (seed 0).** No candidate was skipped in either split. All 80 sampled times (primaries and reserves) passed the join check: the authors' dataset, built for that time alone, returned exactly that time and the expected mask path.

**Compact frames.** Three of the 30 test primaries (2024-05-18 19:00, 2024-10-20 19:00, 2024-12-07 20:00) are about 0.32 GB on S3 instead of about 0.58 GB. In these files the six AIA channels from 131 to 335 Å are stored in 8–24 MB each instead of about 60 MB, while AIA 94 Å, AIA 1600 Å and the HMI channels are about the same size as usual; six of the 30 test reserves are the same. The headers show all 13 channels, and I do not know why these frames compress better. The run records each channel's normalised mean, standard deviation and count of non-finite values for every image, so these frames can be checked.

## Metrics and interval

The hard scores use the sigmoid output thresholded at 0.5. The repository never thresholds, so this threshold is my choice; the run also records scores at thresholds 0.1 to 0.9.
- IoU = TP/(TP+FP+FN), and Dice = 2TP/(2TP+FP+FN).
- An image whose mask and prediction are both empty scores 1. This is the limit of the eps convention in the repository's loss classes.
- The mean is also reported over images with a non-empty mask only. Empty masks are likely in 2020, near solar minimum; 13 of the first 953 masks in the archive (August–September 2010) are empty.

The soft scores are exactly 1 − `IoULoss` and 1 − `DiceLoss`, computed on the sigmoid probabilities.
- eps = 1e-6 and smooth = 1e-5 come from `config.yaml`. YAML reads both as strings, so they are cast to float.
- `IoULoss` averages per-image IoU over a batch, while `DiceLoss` pools all pixels of a batch. With the configured batch size of 1, both are per-image.

Every score is reported two ways: as a mean over images, and pooled over pixels (counts or soft sums added over images first). Each gets a percentile 95% bootstrap interval from 10,000 resamples of the images, seed 0.

The bootstrap ignores the year stratification, which makes it slightly conservative. It covers only the choice of images. It does not cover training randomness, or the paper's unknown protocol.

**Metric checks** (`metric_check.json`):
- On synthetic inputs, the metric code agrees with the authors' `IoULoss` and `DiceLoss`, and with a set-based numpy computation, to within 5e-17.
- A hand-worked two-image example checks the mean and pooled aggregation.
- At run time, the authors' loss classes are also applied to every real output, and the summary reports the largest difference.

## Differences from the paper and the repository

- **Sample size.** The paper presumably scored a whole split; this run scores 30 test and 10 validation images. One float32 forward pass takes about 136 s on this 4-vCPU machine, and each frame is about 0.58 GB, so the 41,910 test images would take more than two months of CPU time.
- **Precision.** The forward pass runs in float32 with no autocast. The repository uses bfloat16 autocast on a GPU. CPU bfloat16 autocast breaks Surya (`results/Surya/bf16_diagnosis`), and float32 is the more exact computation. Pixels with logits near zero may still flip compared with a GPU bfloat16 run.
- **Frame source.** Frames come from the public bucket `s3://nasa-surya-bench`. For the one time compared (2014-01-07 16:00) they decode bitwise-identically to the Hugging Face copies; I assume this holds for all times.
- **No core-sdo index for 2020–2024.** The repository downloads no core-sdo index for those years, so presence on S3 stands in for the index for the test split.
- **The t+60 frame.** It is HEAD-checked and listed in the per-sample index, but never downloaded, because `ArDSDataset` does not read it.
- **Data loading.** Each time stamp gets its own one-sample dataset. The DataLoader uses `num_workers=0`.
- **Checkpoint loading.** `torch.load` uses `weights_only=True` and `mmap=True`. The file contains only an OrderedDict of float tensors, checked from its pickle opcodes, so the loaded weights are the same.
- **Masks.** They are taken from the published `data.tar.gz`, streamed once and never stored, with its SHA-256 checked, and only the sampled members are written. Members are named `YYYY/MM/YYYYMMDD_HHMM.h5`, without the CSV's `data/` prefix, so the README's `tar -xzf data.tar.gz -C data` gives the CSV paths. The archive starts at 2010/08, although `train.csv` lists present masks from 2010-05-13. Within a month the members are not in time order, so the whole archive is read; the run writes a member list and per-year counts.
- **Training config.** The released weights were evidently not trained with the configuration that ships with the code. The checkpoint says epoch 40, while `config.yaml` has 15 epochs. Also, `finetune.get_model` loads the pretrained backbone only for `model_type == "spectformer"`, not for the configured `spectformer_lora`. The weights are evaluated as released, with the shipped model configuration; the strict load confirms the architecture matches.
- **Threshold and averaging.** These are my choices, explained above, since neither the paper nor the repository fixes them.

## What has and has not been run

Done during this session:
- the dry run for both splits: sample selection, S3 HEAD checks, header reads, label and dataset joins, byte totals and time estimates;
- the metric self-test;
- a test of the run-mode code path. It used a stand-in model (not Surya), 256×256 synthetic frames, and a synthetic archive served by a local server that dropped the connection once. It covered archive streaming with reconnect and SHA check, reserve substitution after a failed image, the per-image records, resume, and the summary.

Not done: the real model was not built or run, and no SDO frame, real archive or checkpoint was downloaded, because a 13 GB job was using the machine.

**Estimates.** The test run is estimated at about 92 minutes for 30 forward passes. The validation run is about 35 minutes for 10, or about 5 minutes less if the test run has already left the checkpoint and the masks on disk. Peak disk is about 3.0 GB: the 1.78 GB checkpoint plus at most two frames.

Commands, from the repository root:

    python scripts/Surya/downstream/ar_segmentation_subset_eval.py --dry-run --split test --n 30 --seed 0
    python scripts/Surya/downstream/ar_segmentation_subset_eval.py --split test --n 30 --seed 0
    python scripts/Surya/downstream/ar_segmentation_subset_eval.py --split validation --n 10 --seed 0 --delete-checkpoint

Outputs in this folder:
- `sample_<split>.csv` and `sampling_<split>.json`: the sample, the skips and the estimates;
- `per_sample_<split>.jsonl`: per-image counts, timings and peak memory;
- `arrays/`: 512×512 pooled probability, prediction and mask maps;
- `environment.json`;
- `summary.json`: both splits, with the comparison to Table 3.
