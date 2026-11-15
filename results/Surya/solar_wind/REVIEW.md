# Review of `solar_wind_subset_eval.py` (adversarial, before any real run)

I read the evaluator line by line against the authors' code at commit 78b827e (`dataset.py`, `finetune.py`,
`infer.py`, `models.py`, both configs, `surya/datasets/helio.py`, `surya/models/`). I did not build the model, run a
forward pass or download weights or frames. The checks below used only the label CSVs, the checkpoint's tensor list
(about 0.3 MB of range requests) and netCDF headers.

## What I checked and found correct

The model is built the way infer.py builds it. The model sections of config.yaml and config_infer.yaml are
identical. The strict load will succeed: the checkpoint's 202 tensors match, key for key and shape for shape, what
config_infer.yaml implies (class token, penultimate linear layer, one output, patch embedding at 13 x 1 input planes,
so time_dim 1, LoRA rank 64). Its 367,501,313 parameters, 8,192,000 of them LoRA, are the numbers the notebook printed.
LoRA sits only on `fc1`/`fc2`, because `q_proj`, `k_proj`, `v_proj` and `out_proj` name no module of this backbone.
That was also true in training. LoRA alpha cannot be read from the weights; only the notebook anchors test it. One
observation that does not affect this evaluation: in training, PEFT froze every non-LoRA parameter, including the new
head. The notebook's trainable count equals the LoRA count exactly, so the class token, penultimate linear layer and
output layer stayed at their initial values.

The lead is right. The dataset pairs input t with the first label at or after t + 4 days, within 1 hour. The
evaluator uses only exact pairs T = t + 4 d on the hour, and for every sample it checks that the authors' dataset
picks that t, that T and the CSV's V. The label files are hourly, on the hour, with no duplicates, no NaN and no
OMNI fill values. Missing hours are absent rows, so they can never become targets. 4-day persistence is scored only
where V(t) exists.

De-normalisation is right. train.csv gives min 240 and range 560 over all 69,957 rows, so the network was trained to
emit (V - 240)/560. infer.py as shipped uses validation.csv (259, 494); test.csv would give (256, 750). The primary
uses the training constants, the other two are recorded and scored, and the targets are always the observed km/s.
The inputs reach the network as float32, because the scalers are np.float32, so the result does not depend on
autocast. finetune's permute changes nothing at T = 1. The metric formulas match the repository's and were checked
against its functions. The bootstrap and the seeded sampling are sound. The baselines are mine and labelled as mine.

## Issues found and fixed

1. Medium. The script imported its HTTP helpers from the AR-segmentation script, which other reviews were editing at
   the same time. An edit there could break this run at start-up or change a helper's return value. The helpers are
   now defined in this script.
2. Medium. Resume and the summary used every record in `per_sample_<split>.jsonl`, even records from another
   `--n`/`--seed` or an earlier draw. They now keep only rows of the current sample file, and the dry run warns if a
   redraw differs from the previous one.
3. Medium. Compact input frames went unnoticed. Seven of the 30 test primaries (five of the six in 2024) have an input
   frame whose AIA 131-335 channels are quantised and stored in about a third of the usual space. These samples are
   still scored. Their stored sizes are now recorded, and summary.json gives the metrics with and without them.
4. Medium. There was no RAM check, even though another job held 11 GB. The run now refuses to start below 12 GB
   MemAvailable (`--min-ram-gb`) and makes itself the preferred victim of the OOM killer.
5. Low. The summary crashed (KeyError) when exactly one sample was scored, for example after a validation run with
   `--max-forward 4`. Fixed.
6. Low. Frames left behind by an interrupted run were never deleted. They are now cleared at start.
7. Low. The time stamps of the batch were not checked. With a one-sample dataset the authors' retry loop cannot skip
   to another time, but the input and SDO-target times are now asserted anyway.
8. Low. scalers.yaml was copied unchecked, and the core-sdo index was not hash-checked. Both now come from core-sdo at
   a pinned revision and are checked by git blob id.
9. Low. The strict-load precondition is now checked in the dry run, anchor results are logged as soon as each anchor
   runs, and every baseline entry carries an "applicant's own" label.

## Remaining risks

The paper's split and aggregation are unknown (and sqrt(5698.62) is not 75.92). The test inputs have no authors' SDO
index, so their quality flags are unknown, and the compact frames come on top of that. The seed-0 test primaries
are unusually easy for persistence (1st percentile), so `--include-reserves` is advisable. Only the validation
anchors test LoRA alpha and the pipeline end to end, so run validation first. A frame that is corrupt but has the
right length would make the authors' `__getitem__` retry 100 times before failing. The timings are taken from the
base-model run and are not yet measured for this model.
