# Review of `flare_forecasting_subset_eval.py` (2026-11-14/15)

This was an adversarial review of the flare-forecasting evaluator, checked line by line against the authors' code at `78b827e`. No model was built, no forward pass was run, and no checkpoint or frame was downloaded. The checks used the label CSVs, byte-range reads, the local foundation-model file, a stub model and stub data.

## What is right

The network, the inputs and the labels all come from the authors' own code, called the way `infer.py` calls it.

- **Model.** `get_model` and `apply_peft_lora` run with `config_infer.yaml`, whose model section is identical to `config.yaml`'s. That gives `HelioSpectformer1D` with a class token, `time_dim` 2, one output, and LoRA with r 8 and α 8. The weights are loaded as `infer.load_model` loads them, ending in `load_state_dict(strict=True)`.
- **Checkpoint keys.** The checkpoint's 202 keys, all prefixed `base_model.model.`, are exactly what that construction produces. They are the foundation model without its decoder, plus a class token, a 1280×1280 penultimate layer, a 1×1280 head and rank-8 LoRA on the 20 `fc1`/`fc2` layers. The attention blocks use `qkv`/`proj`, so the config's `q/k/v/out_proj` targets match nothing, in the authors' training as well as here. The patch embedding is [1280, 26, 16, 16]: 13 channels times 2 input times.
- **Inputs.** The inputs are the frames at t−60 and t, with channels in the config's order. The scalers are float32, so `ts` is float32. There is no masking, flipping or pooling. The t+60 target is loaded, as the class requires, and then dropped before the forward pass.
- **Label.** The label is `label_max` at the reference time t, exactly as `SolarFlareDataset` attaches it. All 80 sampled rows pass the join check.
- **Output.** The output is one logit for "flare"; training used `BCEWithLogitsLoss`.
- **Precision.** float32 is faithful. Both configs say `dtype: float32`, so the authors' autocast was a no-op in training and inference as well.
- **Metrics.** TSS, HSS and F1 match `metrics.py` to 5e-8. I re-derived the re-weighting algebra and the closed forms, and both are correct. The class-stratified bootstrap and the year-proportional, block-systematic sampling make the sample self-weighting within each class, so the sample's TPR and FPR estimate those of the candidate set.

## Issues found and fixed

1. **Medium: real-run failures the dry run could not see.** The dry run never imported `infer`/`finetune`/`models`, and never compared the checkpoint with the config. An import or construction failure would only have appeared after a 1.8 GB download and a model build. The dry run now imports those modules (checking that they come from the clone) and checks the checkpoint's key names and shapes against the foundation model and `config_infer.yaml`. It fails if either check fails.
2. **Medium: a redraw could mix two samples.** `--dry-run` overwrote `sample_<split>.csv` even when scored records existed. The planner and `summarize()` would then have combined records from two different samples, because the block keys collide. A redraw that differs from the existing sample is now refused while records exist. `run()` and `summarize()` also ignore records whose time stamp is not in the current sample.
3. **Medium: samples could be dropped silently.** `summary.json` did not list blocks without a scored sample, failed records or reserve use. A NaN logit counted as a "no flare" forecast, because `sigmoid(nan) > 0.5` is false. All of these are now listed, and a non-finite logit is a recorded failure.
4. **Medium, operational: no memory check.** Nothing stopped a real run from starting next to the other ~11 GB job, which would get one of the two OOM-killed. A preflight check now refuses to start below 12 GiB of available RAM (`--min-free-gb`) or with too little disk.
5. **Low.**
   - The frame times returned by the dataset are now asserted. With a one-sample dataset its retry loop cannot move to another time, but the check is now explicit.
   - Reserves added after failures could exceed `--max-forward`.
   - A reserve that had already failed was re-run on every resume.
   - Frames and partial downloads left by an interrupted run were never deleted; they are now swept.
   - The year allocation could silently fall short of n/2.
   - The pipeline self-test crashed for small n.
   - Same-named modules could shadow the authors' `dataset`/`infer`.

The new real-run paths (failure, reserve substitution, the cap, resume, stale records, the sweep) were run end to end with a stub model and stub data. Both dry runs pass and draw the same samples as before.

## Not changed, but note

`infer.py`'s rule, `sigmoid(logit) > 0.5`, stays primary. However, the only repo code that computes TSS, HSS and F1 is `finetune.evaluate_model`, which applies the 0.5 threshold to raw logits. The prevalence implied by Table 4 (0.106) suggests the paper's numbers come from those validation logs. If so, the `logit > 0.5` rule on the validation split is the closer comparison. Both rules are reported, and the final log line now prints both.

## Remaining risks

- **Untested parts.** The real strict load, the forward pass and its peak RSS cannot be checked without running the model.
- **Runtime.** The estimate assumes 136 s per forward pass. At 3–5 min per pass, the test split needs three runs instead of two.
- **Retry stalls.** A failure that the local channel check does not catch, such as a corrupt compressed chunk, makes the authors' `__getitem__` retry the same sample 100 times, about an hour. The file length is verified, but no checksum is.
- **Numerics.** CPU float32 can differ slightly from GPU kernels, which may use TF32 for convolutions. This matters only for logits near the threshold.
- **Sample size.** With 20+20 samples, the TSS interval is about ±0.25. Hourly labels are autocorrelated. The test split's prevalence (0.297) is far from the paper's implied 0.106. Replacing a failed primary with its reserve depends on the failure.
