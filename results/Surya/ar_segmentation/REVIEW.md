# Review of `ar_segmentation_subset_eval.py`

An adversarial review of the AR (PIL) segmentation evaluator against the authors' code at commit 78b827e (`downstream_examples/ar_segmentation/{dataset,finetune,infer,models}.py`, both configs, `surya/`). The review did not build the model, run a forward pass, or download the checkpoint, a frame or the mask archive. It read about 1 MB of the checkpoint and 6 MB of the archive with range requests.

## Issues found and fixed

1. **Crash: `shutil` was never imported** (critical). The real run would have stopped with a NameError before any image was scored. Fixed.
2. **Crash: `check_resources` was called but never defined** (critical). I implemented it as a pre-flight check that runs before anything large is fetched. It needs disk for the checkpoint plus the two largest frames plus 0.5 GB, and at least 12 GB of RAM available. One float32 pass peaks at 10–13 GB, and the threshold can be changed with `--min-ram-gb`.
3. **Resume could silently drop blocks** (high). After the archive step, `run()` read `per_sample_<split>.jsonl` a second time without the `records_in_sample` filter. So an "ok" record left over from a sample drawn with another `--n`, another `--seed`, or a redrawn sample marked a (year, block) as done. That block was then never scored, and the summary's filter quietly removed it. Fixed, and tested with a stale record.
4. **Transient network errors changed the seeded sample** (high, found by this review's dry run). One proxy error during a netCDF header read was recorded as a skip. In block 2024/2, the primary 2024-05-18 19:00 was replaced by its reserve. Header reads now retry on network errors and then stop the selection instead of skipping. Only content problems (a channel missing or with the wrong shape) count as skips. After the fix, the dry run reproduced the original sample.
5. **Disk bound not kept after failed downloads** (medium). When every retry failed, `fetch_s3` left a `.part` file of up to 0.6 GB that `FrameCache` does not track. Such files are now removed from the cache that failed.
6. **No check that the scored image is the requested one** (medium). `ArDSDataset.__getitem__` replaces the parent's retry loop, and each dataset holds one time, so another time could not be substituted in practice. The loop now asserts it anyway: the returned `timestamps_input` must be t, the joined mask path must match the sample, the input must be 1×13×1×4096×4096, and the output must have the mask's shape.
7. **Scalers were not verified** (low). The script copied `scalers.yaml` from the Surya-1.0 folder. It now fetches core-sdo's file at the pinned revision, as `download_data.sh` does, and checks its SHA-256. The two copies are byte-identical (a944e8f1…).
8. **Wasted work on a finished run** (low). The 1.31 GB archive was streamed even when every block was already scored. That no longer happens.
9. **Latent bug in `RangeFile`** (low). It took `Content-Length` from a HEAD request without following redirects. HF answers HEAD with a 302 whose length is 1,113 bytes. Only S3 URLs went through it before. It now follows redirects.
10. **Reporting gaps** (low). `summary.json` now lists images with non-finite inputs or logits and masks whose values are not 0/1 after the /255 scaling. Blocks with no usable candidate now count as planned but not scored. The dry run also gives a slow-case time (300 s per pass) and the RAM available.

The dry run now also checks the checkpoint's pickle header against the config before the 1.8 GB download. It found 199 float32 tensors, all under `base_model.model.`, with LoRA layers stored as `…base_layer.weight` (the layout PEFT 0.17 produces). There are 40 LoRA tensors of rank 32, on `fc1`/`fc2` only, holding 4,096,000 parameters, plus a 256×1280×1×1 unembedding convolution. Surya has no `q_proj`/`k_proj`/`v_proj`/`out_proj`, so those config entries match nothing. Enumerating the modules from the source code (the persistent `pos_embed` buffer, `qkv` without bias, 2 spectral and 8 attention blocks) gives the same key set, so `load_state_dict(strict=True)` should succeed.

## Checked and found faithful

- **Model.** The model sections of `config.yaml` and `config_infer.yaml` are identical. The script uses `infer.get_model` and `infer.apply_peft_lora`, and the `dtype` argument is unused by HelioSpectFormer.
- **Input.** Only the frame at t, the 13 channels in config order, the signum-log transform and standardisation with float32 scalers, so the input is float32. No pooling and no flips. The linear embedding ignores `time_delta_input`.
- **Masks.** `union_with_intersect` is uint8 with values {0, 255}, 4096×4096. Archive members are named `YYYY/MM/…` and the CSV paths carry a `data/` prefix; the script handles the difference. Dividing by 255 gives {0, 1}.
- **Orientation.** The target is `forecast.unsqueeze(1)` compared with the raw model output, with no transpose, exactly as in `finetune.py`. The orientation is therefore the training orientation by construction.
- **Output.** The model emits logits (it was trained with `BCEWithLogitsLoss`). The script applies a sigmoid and a 0.5 threshold, which is logit > 0.
- **Metrics.** IoU and Dice agree with the repository's loss classes and with set-based numpy to 5e-17. Per-image and pooled aggregation and the percentile bootstrap over images are correct.
- **Collation.** `custom_collate_fn` falls back to a list for the `datetime64` metadata, because torch 2.10 raises a TypeError, which it catches.
- **Memory.** Large tensors are released after each image. In the zero-shot runs on this machine, peak RSS stayed at about 10.5 GB across samples.

## Remaining risks

- The real model path (build, strict load, forward) has not been run. A key mismatch would stop the first run loudly, not silently.
- Forward passes run in float32, while the authors used bf16 autocast on a GPU. Pixels with logits near zero may flip.
- The paper's split, threshold and averaging are unknown. The sample has 30 images, so intervals are wide. In the per-image mean, an image with an empty mask scores 0 if the model predicts any positive pixel.
- Some 2024 frames have quantised AIA channels ("compact frames"). The summary reports a sensitivity analysis without them.
- The full 1.31 GB stream through the HF/xet CDN was not exercised. A mid-file Range request returns 206, so reconnects should work.
- The test split has no core-sdo index, so presence on S3 stands in for it.
