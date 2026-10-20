# Solar wind speed 4 days ahead: partial reproduction of Table 6 (arXiv 2508.14112 v2)

This is a partial reproduction. The released fine-tuned Surya checkpoint is scored on a small, seeded sample of the
SuryaBench solar-wind benchmark on a CPU. It does not retrain anything and does not use the full split, so it can say
whether the paper's numbers are plausible for these samples, not what the full split would give.

Script: `scripts/Surya/downstream/solar_wind_subset_eval.py`. Outputs in this folder: `sample_<split>.csv` and
`sampling_<split>.json` (dry run), `metric_check.json` (metric self-test), `per_sample_<split>.jsonl` (one record per
forward pass, appended as the run goes), `environment.json` and `summary.json`.

## What is evaluated and what it is compared with

The task (paper Sec. 2.6.5) is to predict the hourly solar wind speed V at L1 four days after the time t of one SDO
frame (13 channels at 4096x4096). The model is the checkpoint `nasa-ibm-ai4science/solar_wind_surya`,
`solar_wind_weights.pth` (revision 6ac4aa79, sha256 7ff8c190...6a17, checked after download), run with the authors'
code at commit 78b827e. The labels are the SuryaBench CSVs `nasa-ibm-ai4science/Surya-bench-solarwind` (revision
20547a25, checked by git blob id).

The reference is Table 6: Surya RMSE 75.92 km/s, MAE 58.06 km/s, "Validation MSE" 5698.62 km^2/s^2 (ResNet50 93.76,
74.65 and 8547.924; AlexNet 118.6, "95.7 km" and 13839.49). The paper does not say which split, how many samples or
what averaging produced these numbers. The RMSE and MSE columns cannot come from the same set with the same per-sample
mean, because sqrt(5698.62) = 75.49, not 75.92. I compare my RMSE with 75.92, my MAE with 58.06 and my MSE with
5698.62, and report whether each paper number lies inside my 95% interval.

## Splits and sample design

The primary sample is from the test split (test.csv, 2020 to 2024, after every training year). A candidate is a row
T of test.csv whose input time t = T - 4 days is on the hour and in 2020 or later; there are 39,893 such pairs. Thirty
samples are drawn, six per year. Inside each year the candidates, in time order, are cut into six consecutive blocks
of equal count (about two months each), and one pair is drawn per block in a random order fixed by seed 0. A candidate
is used only if the frames at t and t + 60 min are both in the public bucket s3://nasa-surya-bench and hold all 13
channels at 4096x4096. This is checked with a HEAD request and a read of the netCDF header by HTTP range requests.
Otherwise the next candidate in the block is tried, and the skip and its reason are recorded. The test sample had
one skip (2024-10-15 18:00, frame at t + 60 missing). Each block also gets a reserve, the next accepted candidate.
It is run only if the primary fails, or as a second sample of the block with `--include-reserves`.

The secondary sample is from the validation split (validation.csv, 15 to 31 January of 2011 to 2019), the file that
finetune.py evaluates on after every epoch. Here t and t + 60 must also be in core-sdo's `valid_index_surya_1_0.csv`,
the SDO index finetune.py uses for validation, so inputs fall on 15 to 27 January. There are 2,393 candidate pairs.
Ten samples are drawn, one per year, with the remaining one going to a year drawn with the seed (2018). No candidate
had to be skipped. This split is reported separately. It is not strictly held out, because finetune.py measured it every
epoch and may have been used to pick the released epoch (the archive inside the checkpoint is named
`epoch_13_loss_0.001`).

## How the model is run

For each sample the two frames are downloaded to a local cache that holds at most two samples (about 2.4 GB). A
two-row index (t, t + 60) is written in core-sdo's format. The authors' `WindSpeedDSDataset` is then built with the
arguments `finetune.get_dataloaders` gives its validation set in config.yaml, changing only the index path, the frame
root and, for the test split, the label CSV. The dataset forms the pair itself (pd.merge_asof forward, tolerance 1 h).
The script checks that it picked t, the target time t + 4 days and the CSV's V. Batches come from a DataLoader with
the authors' `custom_collate_fn`, batch size 1 and no workers.

The model is built as infer.py builds it. `finetune.get_model` with config_infer.yaml (which first loads the
foundation weights `surya.366m.v1.pt`) is followed by `apply_peft_lora` (rank 64, alpha 32) and
`load_state_dict(strict=True)` of the checkpoint. It runs in eval mode, float32, on the CPU, without autocast, under
`torch.no_grad()`. One forward pass took about 135 s on this machine in the base-model test. The dry run estimates
[[TEST_MIN]] minutes for the 30 test samples and [[VAL_MIN]] minutes for the validation run including the anchors
below. Each run resumes from its jsonl file, so `--max-forward` can split a run into pieces of about 90 minutes.

## De-normalisation

`WindSpeedDSDataset` scales V as (V - min) / range, using the minimum and range of whichever CSV it was given.
finetune.py builds the training set on train.csv (min 240, range 560 km/s) and the validation set on validation.csv
(min 259, range 494), and its loss is the MSE between the raw network output and these scaled targets. The network was
therefore trained to emit (V - 240) / 560, and the prediction used here is 560 y + 240 km/s. infer.py as shipped
rescales with validation.csv's constants instead (494 y + 259), because config_infer.yaml points
`solarwind_train_index` at validation.csv. Pointed at test.csv, it would use that file's constants (min 256, range
750). Both alternatives are recorded and scored, but they are not used for the comparison. The two training-era
choices agree within 2 km/s near y = 0.29, and the gap grows away from it (14 km/s at y = 0.5). The `ds_normalize: true`
and `ds_scaler: [2.61, 0.09]` settings in config.yaml have no effect, because the call to `__scale__` in
`__getitem__` is commented out. The targets are always the observed V from the CSV. The dataset's own scaled target,
un-scaled with its own constants, is checked to equal the CSV value.

## Metrics, intervals and checks

RMSE, MAE and MSE are computed over samples in km/s, as in the summary block of `infer.infer_single_sample`.
`finetune.evaluate_model` computes the same in normalised units. The paper gives no formula, and the repo has no other
metric code for this task. My function was checked against both repo functions on 37 synthetic pairs.
`evaluate_model` was called with a stand-in model in a one-process gloo group, and its relative difference is 4e-8
(its accumulators are float32). The difference from the infer.py printout is 2.4e-5 km/s, within its 4-decimal print.
A hand-worked example also passes (`metric_check.json`). The 95% intervals are percentile bootstrap intervals (10,000
resamples of the samples, seed 0). Model-minus-baseline differences use a paired bootstrap.

## My baselines and how typical the sample is

Three baselines of my own, not in Table 6, are scored on the same samples and, from the CSV alone, on every candidate
pair of the split. They are 4-day persistence V(t), 27-day recurrence V(T - 27 days) and the training-set mean (418.6
km/s). Table 7 quotes a 27-day persistence RMSE of 89 km/s from Ahn et al. (2025) for another period, which is not
comparable. On the whole test split the three give RMSEs of 111.6, 102.4 and 87.0 km/s. On the validation
candidates they give 123.7, 97.9 and 86.6 km/s. Climatology alone is therefore already within about 11 km/s of
Table 6's 75.92 km/s, which sets the scale for judging the model.

The seed-0 test sample turned out to be unusually easy for persistence. Its 29 pairs with V(t) give a persistence
RMSE of 75.1 km/s. That is about the 1st percentile of 2000 samples drawn with the same design (median 110 km/s;
middle 95% from 80 to 143). Its reserves give 119 km/s. For 27-day recurrence it is at the 10th percentile (79 km/s),
and for climatology it is somewhat harder than usual (94.9 km/s, 82nd percentile). Primaries and reserves together
are typical: the 13th, 12th and 37th percentiles. The same simulation shows that, at n = 30, a nominal 95% bootstrap
interval covers the full-split RMSE of these baselines only 86 to 90% of the time. The intervals are therefore a
little too narrow, and a single 30-sample RMSE can sit well away from the full-split value.

The seed was fixed before this was seen and was not changed. Instead, summary.json adds two things. The first is a
regression (control-variate) estimate of the model's full-split MSE and RMSE. It uses the squared errors of
persistence and climatology, whose full-split means are known from the CSV (Cochran, Sampling Techniques, ch. 7).
This is my adjustment, approximate at this n, and it is reported next to the plain sample numbers, not instead of
them. The second is the same design-spread diagnostic for the scored samples. `--include-reserves` doubles the test
sample to two pairs per block (n = 60) over two or three resumable runs. I recommend it here, because the primaries
alone are atypical. The reserves were drawn by the same seeded procedure, so adding them is not a selection made
after seeing results.

## Pipeline anchors

The saved output of the repo's `solar_wind_tutorial.ipynb` is a float32 CPU run of infer.py ("CUDA available: False",
autocast disabled, torch 2.8.0). It covers three validation-period inputs: 2011-01-20 01:00 (printed 393.63 km/s),
2019-01-23 02:00 (391.95) and 2011-01-20 02:00 (383.07), un-normalised with validation.csv's constants. The
validation run scores these three first and compares them with the printed values. The tolerance, 0.5 km/s, was fixed
before any run. They test the whole chain (S3 frames, dataset, model build, LoRA wrapping, strict load, forward pass)
and are kept out of every sample metric. The notebook's ground truths (344, 433, 351 km/s) equal the CSV values the
dataset picks for t + 4 days. Its "Time Target" column shows t + 60 min, the SDO dataset's own target time, not the
real 4-day target.

## Differences from the paper and the repo, and why

The evaluation set is a 30 + 10 sample, not a full split, because a forward pass takes over two minutes on this CPU
and each sample needs 1.2 GB of frames. The paper's own set is unknown.

Only exact pairs T = t + 4 days on the hour are used. The authors' dataset would also accept a later T within one
hour when the exact hour is missing; leaving those out makes every target time unambiguous.

For the test split the authors released no SDO index for 2020 to 2024 (core-sdo covers 2011 to 2019), so presence
on S3 plus the header check plays the role of the index. For the validation split, core-sdo's index and S3 are both
required.

The repo builds one dataset over a whole index; here a one-sample dataset is built per time stamp, so its duplicate
removal between neighbouring times never applies. The frames are the S3 copies, which decode bit-for-bit like the
Hugging Face ones.

The forward pass runs in float32 on a CPU without autocast. Both repo configs set dtype float32, and CPU bfloat16
breaks Surya (results/Surya/bf16_diagnosis). Small float32 differences from a GPU are expected; the anchors measure
them against the authors' own CPU run.

The checkpoint is read with `torch.load(weights_only=True, mmap=True)`, not infer.py's `weights_only=False`. I checked
from the pickle opcodes, without unpickling, that it holds only an OrderedDict of FloatStorage tensors.

Before the forward pass, the t + 60 frame (`batch["forecast"]`) is dropped to save 0.87 GB of RAM; the 1-D model never
reads it. The dataset still reads it, as in the repo, so it is downloaded. finetune.py permutes `batch["ts"]` to
(B, T, C, H, W) and infer.py does not. With one input time the two give identical tensors, which is checked on every
sample. The dataset phase is "valid", as in finetune.py's validation set, where infer.py uses "train"; with channel
masking and HMI dropping both off, the phase changes nothing but a log file name.

The primary de-normalisation uses train.csv's constants, as explained above, not the validation.csv constants infer.py
uses. Both are reported.

The persistence, recurrence and climatology baselines, the representativeness diagnostic, the regression-adjusted
estimate and the notebook anchors are my additions and are labelled as such in summary.json.
