# Flare forecasting (Surya paper, Table 4): partial reproduction on a seeded sample

This is a partial reproduction. It scores the released fine-tuned flare model on a small sample of the benchmark, on a CPU. It does not re-train anything and it does not use the full evaluation split.

The script is `scripts/Surya/downstream/flare_forecasting_subset_eval.py`. The dry run writes `sample_<split>.csv`, `sampling_<split>.json`, `excluded_<split>.csv` (every label row left out, with the reason) and `metric_check.json` (the metric self-test) to this folder. A real run adds `per_sample_<split>.jsonl`, `environment.json` and `summary.json`.

## What is evaluated

The model is the checkpoint `nasa-ibm-ai4science/solar_flares_surya` (revision `ec7c42a`, file `solar_flare_weights.pth`, SHA-256 `a907da5f…`). It runs through the authors' own code from the clone at `78b827e`, imported without changes. The data come from `SolarFlareDataset`, called with the arguments `infer.get_dataloader` passes: inputs at t−60 and t, a target frame at t+60, and the label `label_max`. The network is built with `get_model` and `apply_peft_lora` from `config_infer.yaml` (LoRA r 8, α 8, class-token pooling, one output). The weights are loaded as `infer.load_model` does it, ending in `load_state_dict(strict=True)`.

The task is binary: does a flare of class M1.0 or above occur in [t, t+24 h)? The model returns one logit for the positive class "flare". `infer.py` applies a sigmoid and predicts 1 when the probability exceeds 0.5; there is no softmax. The forward pass runs in float32, without autocast, under `torch.no_grad()`, with `num_workers=0` and one sample at a time.

## Reference numbers and what they imply

Table 4 of arXiv 2508.14112 v2 (Sec. 2.6.3) gives TSS 0.436, HSS 0.522 and F1 0.561 for Surya, using Eq. 21–23. The paper does not name the split, the number of samples or the threshold.

The three values fix the confusion matrix up to scale. Solving Eq. 21–23 for them gives a unique answer: prevalence 0.106, recall 0.456 and false-positive rate 0.020. Over the rounding box of the published values (±0.0005 each), the prevalence stays between 0.103 and 0.109. The script recomputes this in `summary.json` under `implied_by_table4`.

That prevalence matches the SuryaBench validation split, whose candidate set here has prevalence 0.108 (3,243 usable hours). It does not match the 2020–2024 test split (0.297). The repo's `finetune.py` also validates on exactly that split. It computes TSS, HSS and F1 on a fixed shuffled subset of 200 batches per GPU (`iters_per_epoch_valid: 200`, sampler seed 0, no `set_epoch`). So the paper's numbers most likely come from the validation split, possibly from the fine-tuning logs. This is an inference from the numbers; the paper does not say it.

## Split and sample

The primary evaluation uses the test split (all of 2020–2024), because the workflow asked for it. A smaller validation-split sample (Jan 15–31 of 2011–2019) is reported separately. Given the previous section, the validation sample is the closer comparison with Table 4.

A label row at time t is a candidate only if three conditions hold. First, the rows at t−60 and t+60 must also be in the split CSV: the authors' class inner-joins its frame index with the labels, so it cannot load a frame whose row is missing. Second, the frames at t−60, t and t+60 must all be in the S3 bucket listing. Third, for the validation split only, the three frames must have `present == 1` in core-sdo's `valid_index_surya_1_0.csv`, the index `finetune.py` validates on. This leaves:

| Split | Rows in CSV | Candidates | Positives | Prevalence (candidates) | Prevalence (whole CSV) |
|---|---|---|---|---|---|
| Test | 43,848 | 40,260 | 11,945 | 0.2967 | 0.2943 |
| Validation | 3,672 | 3,243 | 350 | 0.1079 | 0.1089 |

Every excluded row is listed with its reason in `excluded_<split>.csv`. In the test split, 2 rows sit at the edge of the split and the other 3,586 lack at least one of their three frames in the bucket. In the validation split, 18 rows sit at the edge, and every other excluded row is missing both from the core-sdo index and from the bucket. The excluded hours have about the same flare fraction as the kept ones (0.267 against 0.297 for test, 0.117 against 0.108 for validation), so the exclusions hardly move the prevalence.

The sample is stratified by class: half positives and half negatives, 20 + 20 for test and 10 + 10 for validation. Within each class, the samples are spread over years in proportion to that class's candidates per year (largest-remainder rule), so the sample is self-weighting within each class. For the test split this gives positives 0, 1, 4, 6 and 9 for 2020 to 2024 (2020 has only 48 positive candidates) and negatives 6, 5, 4, 3 and 2.

Within each class-and-year stratum, the candidates are cut into equal consecutive time blocks, and one hour is drawn per block in a seeded random order. This keeps the draws apart in time. Each draw is checked with HEAD requests on its three frames, and the sizes are compared with the listing. A failing draw would be skipped and recorded; none failed in either dry run.

Each block also has a reserve. It is used only if the primary fails at run time, for example when a frame lacks one of the 13 channels; every downloaded frame is checked locally before the dataset reads it. The run alternates positives and negatives, so an interrupted run still leaves a balanced subset.

## Decision rule and metrics

The repo uses two decision rules, and both are scored from the same logits. The primary one is `infer.py`'s: predict 1 when `sigmoid(logit) > 0.5`, that is, logit > 0. The second comes from `finetune.evaluate_model`, the only code in the repo that computes TSS, HSS and F1. It passes raw logits to `DistributedClassificationMetrics(threshold=0.5)`, so it predicts 1 when logit > 0.5, which is probability > 0.622.

TSS = TPR − FPR does not depend on prevalence, so it is estimated directly from the balanced sample.

HSS and F1 do depend on prevalence. They are re-weighted to a target prevalence π from the sample's TPR and FPR. The expected counts per unit are TP = π·TPR, FN = π·(1−TPR), FP = (1−π)·FPR and TN = (1−π)·(1−FPR), and Eq. 22–23 are applied to these counts. In closed form:

F1 = 2π·TPR / (π(1+TPR) + (1−π)·FPR)

HSS = 2π(1−π)(TPR−FPR) / (π[π(1−TPR) + (1−π)(1−FPR)] + (1−π)[π·TPR + (1−π)·FPR])

π is taken in three ways. The primary choice is the candidate-set prevalence of the split. The second is the prevalence of the whole split CSV. The third is the 0.106 implied by Table 4, which shows what this model's TPR and FPR would give at the paper's apparent prevalence. HSS and F1 on the sample as drawn (prevalence 0.5) are also reported, labelled `raw_sample`; they are not comparable with the paper.

The intervals are 95% percentile bootstraps with 10,000 resamples. Positives and negatives are resampled separately, because the design fixes how many of each there are. Wilson intervals for TPR and FPR, and a ROC AUC (threshold-free, not in the paper), are added for context.

Every metric path is checked against the repo's own code. In `metric_check.json`, our Eq. 21–23 are compared with `DistributedClassificationMetrics` on synthetic predictions (fed both probabilities and raw logits), on a hand-worked example and on a re-weighting identity. The largest difference is 4.8e-8, which is float32 rounding in the repo class. A real run also applies that class to the actual per-sample values and to the re-weighted expected counts.

A simulation in `metric_check.json` shows what 20 + 20 samples can resolve. The bootstrap intervals cover the true value 93–94% of the time for TSS and 90–93% for the re-weighted HSS and F1, so they are slightly too narrow. The TSS interval is about ±0.2 to ±0.25 wide, so a sample of this size can detect only a large departure from 0.436.

## Differences from the paper and the repo

1. **A sample, not the whole split.** Each forward pass of the 4096×4096 model takes about 136 s here, and the full splits would need about 2 TB (validation) to 24 TB (test) of frames.
2. **Re-weighted HSS and F1.** The class-stratified design and the re-weighting are my additions. They give plug-in estimates of the metric at a given prevalence, not the metric computed on the actual split. The simulation shows a small upward bias, about +0.008 for HSS at the paper-like operating point.
3. **Split.** The paper does not state its split. Test is used as primary, as requested, although the Table 4 triple points to the validation split. The test period covers the rising and maximum phase of solar cycle 25. It lies outside the 2011–2019 training years and has about 2.5 times the training split's flare prevalence (0.297 against 0.121). The validation hours lie only two weeks from training data of the same and the previous year, so long-lived active regions can appear in both.
4. **Threshold.** The repo itself is inconsistent about the threshold, so both rules are reported, with `infer.py`'s as primary.
5. **Hardware and precision.** The run is on CPU in float32. `config_infer.yaml` asks for float32, and `infer.py`'s autocast is disabled on CPU anyway. CPU bfloat16 is avoided because it breaks Surya (`results/Surya/bf16_diagnosis`). Small kernel-level differences from a GPU run remain possible.
6. **Loading and data handling.** None of these changes the values the model sees. The checkpoint is read with `torch.load(weights_only=True, mmap=True)` instead of `weights_only=False`; its pickle references only `OrderedDict`, `_rebuild_tensor_v2` and `FloatStorage`, which the dry run checked from a byte-range read of `data.pkl`. The t+60 target frame is downloaded and loaded because the authors' dataset requires it. The unused `batch["forecast"]` tensor is then dropped before the forward pass to save 0.87 GB; with this config, `HelioSpectformer1D` reads only `ts` and `time_delta_input`. Frames come from the public S3 bucket rather than the HF core-sdo archives; their decoded values were bitwise identical for the timestamp checked (`results/Surya/s3_vs_hf_frames.json`).
7. **Edge-of-split hours are excluded.** Hours whose t−60 or t+60 row falls outside the split are dropped: 2 test rows and 18 validation rows. In the authors' validation loop these would fail to load and be silently replaced by the next index through the dataset's retry loop.
8. **Samples are not independent.** Hourly labels are autocorrelated, because neighbouring hours share most of their 24-hour flare window. The time blocks keep the draws apart, but the bootstrap treats the samples as independent. This matters most for the validation split, whose positives all come from short active periods in January of 2011, 2012, 2014 and 2015, so those intervals are optimistic.
9. **Label threshold.** The labels follow the dataset, which counts M1.0 itself as a flare (≥ M1.0); the paper's text says "exceeds".

## Status, cost and how to run

Both dry runs were last repeated on 2026-10-20 with the current script, and both drew the same sample as before. A dry run selects the sample, makes the HEAD checks, builds the authors' dataset for every sampled hour, checks the checkpoint pickle, probes the bandwidth and runs the metric self-test. For every sampled hour the dataset yields exactly that hour and its CSV label.

The dry run also runs a pipeline self-test on synthetic logits, covering the code that only a real run would otherwise reach. It checks that the run planner alternates positives and negatives, that a capped run takes a prefix, and that a resumed run skips scored hours and replaces a failed primary with its reserve. It also checks `summarize()`: its TSS and re-weighted HSS and F1 agree with a direct computation, and with the repo's metric class, to within 2e-8. No weights or frames have been downloaded and no model has been run, so `summary.json` does not exist yet.

The time estimate assumes 136 s per forward pass, about 12 s to read one frame and 150 s to build the model. The next sample's frames download during the current forward pass, so the bandwidth matters little as long as it stays near its usual level. Single 16 MB probes vary a lot here: S3 read at 11.3, 2.3 and 13.9 MB/s in three dry runs within ten minutes. So the dry run reports a compute-only figure, a figure at typical bandwidth, and one at the bandwidth just measured:

| Split | Forward passes | Compute only | At 11 MB/s (S3) and 17 MB/s (HF) | Passes that fit in 90 min | Frames transferred |
|---|---|---|---|---|---|
| Test | 40 | 121 min | 130 min | 27–29, so two runs | 64 GB |
| Validation | 20 | 62 min | 70 min | all 20, one run | 35 GB |

A run does not start a sample that would end after `--max-minutes` (default 90). The next call resumes from `per_sample_<split>.jsonl`. At most 1.8 GB of frames are on disk at once. Peak new disk use is 3.6 GB, including the 1.8 GB checkpoint. A run needs the machine to itself: the forward pass peaks near 13 GB of the 15 GB of RAM.

```
python scripts/Surya/downstream/flare_forecasting_subset_eval.py --dry-run --split test --n 40 --seed 0
python scripts/Surya/downstream/flare_forecasting_subset_eval.py --split test --n 40 --seed 0         # twice: the first call stops at the 90-min budget, the second resumes
python scripts/Surya/downstream/flare_forecasting_subset_eval.py --dry-run --split validation --n 20 --seed 0
python scripts/Surya/downstream/flare_forecasting_subset_eval.py --split validation --n 20 --seed 0
python scripts/Surya/downstream/flare_forecasting_subset_eval.py --summarize-only
```
