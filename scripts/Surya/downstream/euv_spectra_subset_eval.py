"""EUV spectra (SDO/EVE MEGS-A, 1343 bins) with the released fine-tuned Surya model, on a seeded sample of SuryaBench.

PARTIAL REPRODUCTION. Table 5 of arXiv 2508.14112 v2 (Sec. 2.6.4, "Solar EUV spectra modeling") reports for Surya
MSE 0.0001260741, MAE 0.0045114677 and MAPE 1.4792510271 (AlexNet 0.0001311409 / 0.0061858603 / 1.6834715604,
ResNet50 0.0030414662 / 0.0529725812 / 8.8635520935). The paper gives no split, no sample count and no formula for
any of the three metrics. This script scores the released checkpoint nasa-ibm-ai4science/euv_spectra_surya on a small,
seeded, time-stratified sample of the EVE evaluation times on a CPU and reports MSE, MAE and MAPE with bootstrap 95%
intervals over samples. It can say whether the paper's numbers are plausible for these samples; it cannot say what the
full split would give.

Data the authors' code actually uses. finetune.py / infer.py / config*.yaml read the netCDF file
archive/AIA_EVE_temporally_stratified.nc of the HF dataset nasa-ibm-ai4science/euv-spectra (933,907,814 bytes,
revision aafccf9). It holds train_time/train_spectra (165,430 rows, 2010-05-13..2014-05-26), val_time/val_spectra
(3,275 rows, Jan 15-28 of 2012 and 2013) and test_time/test_spectra (4,880 rows, Jan 15-28 of 2011 and 2014), at a
1-minute cadence with gaps (time stamps at :04/:05 s), plus the attribute split_rules ("test = weeks 3-4 (2011, 2014
only); val = weeks 3-4 (2012, 2013 only); train = weeks 7-52"). The SuryaBench 2020-2024 "test" split of the other
tasks does not exist here: EVE MEGS-A stopped in May 2014. --split test therefore means test_time/test_spectra and
--split validation means val_time/val_spectra, which is what finetune.get_dataloaders validates on (each epoch, at most
iters_per_epoch_valid = 200 batches of a shuffled DistributedSampler) and what infer.py always uses.

Targets. EVEDSDataset (downstream_examples/euv_spectra_prediction/dataset.py) replaces zeros (none exist in this file)
by the per-wavelength minimum of the split, takes log10 and min-max scales with the fixed bounds -9.00 and -1.96:
y = (log10(E) + 9) / 7.04, so y is in [0, 1] (observed 0.0024..0.98; the floor 0.0024 is EVE's 1.04e-9 W/m^2/nm, met
only in a few bins near 32.4-33.2 nm). The network ends in a sigmoid and is trained with MSELoss on y. The paper's MSE
and MAE are on this normalised scale (finetune.evaluate_model pools |e| and e^2 over all elements; the repo README and
the tutorial notebook print MAE 0.005164 / 0.003811 on the same scale).

MAPE is not implemented anywhere in the repo. Here MAPE = (100 / N) * sum_k |yhat_k - y_k| / |y_k| over all
N = n_samples x 1343 elements on the normalised scale, in percent: Table 5 prints 1.479 and the text of Sec. 2.6.4 says
"FISM ... attains a MAPE of 1.5%" and the repo README lists "1.48%", so the unit is percent; a fraction of 1.48 would
need errors of about 150% of the target while MAE is 0.0045 on targets of about 0.6. No target is zero (minimum
0.0024), so no epsilon or clipping is used; because the few floor-valued elements can still dominate the mean, two
variants fixed before any run are reported next to it (elements with y >= 0.05 only; and MAPE on the de-normalised
irradiance 10^(7.04 y - 9) in W/m^2/nm), plus the median absolute percentage error.

Evaluation set (the authors' join, unchanged). EVEDSDataset is built on core-sdo's valid_index_surya_1_0.csv (the SDO
index finetune.py validates with; Jan 15-31 of 2011-2019): an SDO time t is valid when t and t+60 min are indexed
(time_delta_input_minutes [0], time_delta_target_minutes +60, rollout_steps 0); pd.merge_asof(direction="forward")
pairs t with the first EVE time >= t, duplicates of an EVE time keep the closest t, and pairs further apart than 6
minutes are dropped (ds_time_tolerance "6m", hard-coded in finetune.py and infer.py). That gives 355 (t, EVE) pairs for
val_time and 517 for test_time; every candidate here is one of those pairs, and for every sampled pair a one-sample
EVEDSDataset (index with t and t+60 only) is checked to give the same EVE time and a bit-identical target vector.

What is the authors' code (imported unchanged from the clone, nothing copied or edited): EVEDSDataset; custom_collate_fn
and get_model (finetune.py, as imported by infer.py) with config_infer.yaml (infer.py's default), then the checkpoint
loading of infer.load_model with load_state_dict(strict=True). The EUV config has freeze_backbone True and no use_lora
key; infer.py applies no LoRA/peft, so neither does this script. The metric code it is checked against:
finetune.evaluate_model (MAE, RMSE, R^2, mean MSELoss; it reads a module-global `config` that only exists under
__main__, so the check sets finetune.config before calling it) and infer.infer_samples (MAE, RMSE, R^2, mean spectral
correlation), both called on synthetic inputs (metric_check.json).

Differences from infer.py (none changes the network or its weights): torch.load with weights_only=True and mmap=True
(infer.py: weights_only=False; --allow-full-unpickle restores that if the pickle needs it); float32 without autocast
(infer.py's autocast(cpu, float32) disables itself; CPU bfloat16 breaks Surya, results/Surya/bf16_diagnosis);
DataLoader with num_workers=0 and a one-sample dataset per time stamp; batch["forecast"] (the t+60 frame, which
HelioSpectformer1D never reads) is dropped before the forward pass to save 0.87 GB of RAM.

Pipeline anchors. The saved output of euv_spectra_tutorial.ipynb (committed 2025-09-26, after the current checkpoint
was uploaded on 2025-08-26) is a float32 CPU run of infer.run_inference on two val_time pairs (SDO 2012-01-15 00:48 and
2013-01-28 18:12) printing per-sample prediction and ground-truth ranges (4 decimals) and MAE 0.003811, RMSE 0.013978,
R^2 0.966645, mean spectral correlation 0.983955 (6 decimals). The ground-truth ranges are checked in the dry run; in a
real run the two anchors are run first (once, whatever the split), scored with infer.infer_samples itself (replaying
our predictions) against those printed numbers with tolerances fixed in advance, and kept out of the sample metrics
unless they are also drawn as samples of the validation split.

Baseline (MY OWN, not in Table 5): the mean normalised training spectrum (train_spectra through the dataset's
transform), scored on the same samples and on every candidate pair of the split (no model). It also measures how
representative the sample is.

Sampling. Candidates are the authors' (t, EVE) pairs of the split. The sample size is allocated to the two years in
proportion to their candidate counts (largest remainder, ties by the seed); inside a year the time-sorted candidates are
cut into equal consecutive blocks, one sample per block, drawn in a seeded random order. A candidate is skipped (and the
next one in the block tried) if its frame at t or t+60 is not on S3 or lacks one of the 13 channels at 4096x4096
(netCDF header read with HTTP range requests). Every skip is recorded with its reason. Each block also gets a reserve,
used only if the primary fails at run time.

Usage (from the repository root):
  python scripts/Surya/downstream/euv_spectra_subset_eval.py --dry-run --split test --n 30 --seed 0
  python scripts/Surya/downstream/euv_spectra_subset_eval.py --split test --n 30 --seed 0
  python scripts/Surya/downstream/euv_spectra_subset_eval.py --dry-run --split validation --n 20 --seed 0
  python scripts/Surya/downstream/euv_spectra_subset_eval.py --split validation --n 20 --seed 0
  python scripts/Surya/downstream/euv_spectra_subset_eval.py --summarize-only
A rerun skips time stamps already in per_sample_<split>.jsonl / anchors.jsonl, so an interrupted run resumes;
--max-forward K limits one invocation to K forward passes (the dry run prints the K that fits in about 90 minutes).
"""
import argparse
import concurrent.futures as cf
import contextlib
import gc
import io
import json
import os
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import surya_common as C  # noqa: E402  (sets HF_HUB_DISABLE_XET=1 before huggingface_hub is imported)
# our own helpers from the sibling AR script (HTTP range reader for netCDF headers, small-file fetch, sha256, HEAD retry)
from ar_segmentation_subset_eval import fetch_small, remote_nc_channels, s3_exists_retry, sha256_file  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")  # infer.infer_samples saves a matplotlib figure

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
UPSTREAM = "/home/user/upstream/surya/Surya"
TASK_DIR = os.path.join(UPSTREAM, "downstream_examples", "euv_spectra_prediction")
FM_DIR = "/home/user/work/surya_run/data/Surya-1.0"
WORK = "/home/user/work/surya_ft/euv_spectra"
# config.yaml's paths are relative ("./assets/..."), and HelioNetCDFDataset writes logs/data/ relative to the cwd, so
# the script runs with WORK as the working directory and lays out WORK/assets/ the way download_data.sh would.
ASSETS = os.path.join(WORK, "assets")

HF = "https://huggingface.co"
EVE_REPO, EVE_REV = "nasa-ibm-ai4science/euv-spectra", "aafccf9c1364bd39eaf49e058e39f24f70719657"
EVE_FILE, EVE_SIZE = "archive/AIA_EVE_temporally_stratified.nc", 933907814
EVE_SHA256 = "6fe7786c2710b5362885f3f779bbe09df5ff5db74060c15501095722e0c56458"  # HF LFS oid at EVE_REV
CKPT_REPO, CKPT_REV = "nasa-ibm-ai4science/euv_spectra_surya", "6ed3bc13eeece973653422cc83e583aabd31dae0"
CKPT_NAME, CKPT_SIZE = "euv_spectra_weights.pth", 1779722720
CKPT_SHA256 = "f064836d449b52b5573603435cee87130bc9fc5cd89e8dec7581dd20dcb6159f"  # HF LFS oid at CKPT_REV
CKPT_HISTORY = {  # HF commit list of the model repo (fetched 2026-10-20)
    "current": {"rev": CKPT_REV, "date": "2025-08-26T15:42:52Z", "bytes": CKPT_SIZE, "sha256": CKPT_SHA256},
    "replaced": {"rev": "253bf1091d3b71393591fd06899c244784ba1a44", "date": "2025-08-18T22:41:44Z",
                 "bytes": 1776981576, "sha256": "dd71f11b23f5c84ae661a96ad5bc30c4c5ebff3dbfc68f0aa65bcda1a81ae580"},
    "note": "arXiv v1 2025-08-18, v2 2025-08-21; Table 5 is identical in both. Which upload produced Table 5 is not "
            "stated; the notebook anchors were committed 2025-09-26, after the current upload."}
CORE_SDO_REPO, CORE_SDO_REV = "nasa-ibm-ai4science/core-sdo", "9c10abe52c1f206d1b66715745229e29ef543cc6"

PAPER_TABLE5 = {  # arXiv 2508.14112 v2, Sec. 2.6.4, Table 5 (copied exactly)
    "Alexnet": {"mse": 0.0001311409, "mae": 0.0061858603, "mape": 1.6834715604},
    "ResNet50": {"mse": 0.0030414662, "mae": 0.0529725812, "mape": 8.8635520935},
    "Surya": {"mse": 0.0001260741, "mae": 0.0045114677, "mape": 1.4792510271},
}
PAPER_TEXT = {"FISM_sec_2_6_4": {"mape_pct": 1.5, "mse": 0.00031},
              "conclusion_MAE": {"Surya": 0.0043, "AlexNet": 0.0065, "ResNet50": 0.0256},
              "repo_README_MAPE": "1.48% (Surya) vs 1.68% (AlexNet)"}
# euv_spectra_tutorial.ipynb, cell 10 (infer.run_inference, config_infer.yaml, "CUDA available: False", torch 2.8.0+cu126,
# "CPU Autocast ... Disabling autocast": a float32 CPU run of the same pipeline). infer.get_dataloader with the 15-frame
# infer_index_surya_1_0.csv gives a dataset of 2 val_time pairs; Subset([0, 1]).
NOTEBOOK_ANCHORS = [
    {"timestamp": "2012-01-15 00:48:00", "pred_range": (0.3569, 0.9730), "gt_range": (0.0096, 0.9711)},
    {"timestamp": "2013-01-28 18:12:00", "pred_range": (0.3111, 0.9690), "gt_range": (0.0097, 0.9699)},
]
NOTEBOOK_SUMMARY = {"mae": 0.003811, "rmse": 0.013978, "r2": 0.966645, "avg_corr": 0.983955, "n": 2}
# Tolerances fixed before any run: ground-truth ranges must agree at the 4 printed decimals (data path is deterministic);
# predictions may differ by float32 kernel differences across CPUs and torch 2.8 vs 2.10 (sigmoid outputs, so ~1e-5)
# plus print rounding (5e-5 at 4 decimals, 5e-7 at 6 decimals).
ANCHOR_TOL = {"gt_range": 5e-5 + 1e-9, "pred_range": 2e-4, "mae": 2e-5, "rmse": 2e-5, "r2": 2e-4, "avg_corr": 2e-4}
SPLIT_TIME = {"test": "test_time", "validation": "val_time"}
SPLIT_SPECTRA = {"test": "test_spectra", "validation": "val_spectra"}
STRATA_YEARS = {"test": [2011, 2014], "validation": [2012, 2013]}
DEFAULT_N = {"test": 30, "validation": 20}
LOG_MIN, LOG_MAX = -9.00, -1.96  # dataset.py global_min / global_max (de-normalisation; checked against the dataset)
NEAR_ZERO = 0.05  # MAPE variant: elements with normalised target below this are left out (fixed before any run)
N_BOOT = 10000
# Timing assumptions for the estimate (the dry run cannot time the model): one fp32 forward pass of the 4096x4096
# Surya backbone took 136 s on this machine (results/Surya/validation_test, 271.9 s for two passes); reading and
# normalising one 13-channel frame is taken as 30 s (two frames per sample: t and t+60); scoring 2 s; building the
# model, loading the foundation weights (get_model needs them) and then the fine-tuned checkpoint 150 s.
SEC_FORWARD, SEC_LOAD_FRAME, SEC_SCORE, SEC_MODEL = 136, 30, 2, 150
DISK_MARGIN_GB = 1.0  # free space kept on / at all times during a run
FRAME_BUDGET_GB = 3.0  # frames on disk at once (task budget)


# ----------------------------------------------------------------------------------------------- small utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hf_url(repo, rev, path, kind="datasets"):
    return f"{HF}/datasets/{repo}/resolve/{rev}/{path}" if kind == "datasets" else f"{HF}/{repo}/resolve/{rev}/{path}"


def stamp(t):
    return pd.Timestamp(t).strftime("%Y%m%d_%H%M")


def jnum(x):
    """float for JSON, None for NaN/None."""
    return None if x is None or not np.isfinite(x) else float(x)


def free_gb(path=WORK):
    return shutil.disk_usage(path).free / 1e9


def denorm(y):
    """Normalised target -> irradiance in W/m^2/nm (inverse of dataset.py's (log10(E) - LOG_MIN) / (LOG_MAX - LOG_MIN))."""
    return 10.0 ** (np.asarray(y, dtype=np.float64) * (LOG_MAX - LOG_MIN) + LOG_MIN)


# --------------------------------------------------------------------------------------------------- metrics
def sample_stats(p, y, wavelengths=None):
    """Per-sample sums from which every pooled metric is built (so pooling over elements = the authors' pooling)."""
    p, y = np.asarray(p, dtype=np.float64).ravel(), np.asarray(y, dtype=np.float64).ravel()
    e = p - y
    ape = np.abs(e) / np.abs(y)
    keep = y >= NEAR_ZERO
    pp, yp = denorm(p), denorm(y)
    ape_phys = np.abs(pp - yp) / yp
    k = int(np.argmax(ape))
    corr = np.corrcoef(p, y)[0, 1] if np.std(p) > 0 and np.std(y) > 0 else np.nan
    return {"n": int(e.size), "sse": float(np.sum(e ** 2)), "sae": float(np.sum(np.abs(e))),
            "sape": float(np.sum(ape)), "n_ge": int(keep.sum()), "sape_ge": float(np.sum(ape[keep])),
            "sape_phys": float(np.sum(ape_phys)), "sy": float(np.sum(y)), "syy": float(np.sum(y ** 2)),
            "corr": float(corr), "max_ape_pct": float(100 * ape[k]), "max_ape_bin": k,
            "max_ape_wavelength_nm": None if wavelengths is None else float(wavelengths[k]),
            "max_ape_target": float(y[k]), "max_ape_pred": float(p[k]),
            "median_ape_pct": float(100 * np.median(ape)),
            "mse": float(np.mean(e ** 2)), "mae": float(np.mean(np.abs(e))), "mape_pct": float(100 * np.mean(ape)),
            "pred_min": float(p.min()), "pred_max": float(p.max()), "target_min": float(y.min()),
            "target_max": float(y.max()), "target_mean": float(y.mean())}


STAT_KEYS = ("n", "sse", "sae", "sape", "n_ge", "sape_ge", "sape_phys", "sy", "syy", "corr")


def stack_stats(stats):
    return {k: np.array([s[k] for s in stats], dtype=np.float64) for k in STAT_KEYS}


def pooled(S, idx=None):
    """Pooled metrics from stacked per-sample sums; idx (B, n) gives bootstrap resamples (returns arrays of length B)."""
    g = (lambda a: a[idx].sum(-1)) if idx is not None else (lambda a: a.sum())
    gm = (lambda a: np.nanmean(a[idx], -1)) if idx is not None else (lambda a: np.nanmean(a))
    n = g(S["n"])
    mse = g(S["sse"]) / n
    sst = g(S["syy"]) - g(S["sy"]) ** 2 / n
    return {"mse": mse, "rmse": np.sqrt(mse), "mae": g(S["sae"]) / n, "mape_pct": 100 * g(S["sape"]) / n,
            f"mape_pct_targets_ge_{NEAR_ZERO}": 100 * g(S["sape_ge"]) / g(S["n_ge"]),
            "mape_pct_physical_irradiance": 100 * g(S["sape_phys"]) / n,
            "r2_infer_py": 1 - g(S["sse"]) / sst, "mean_spectral_corr_infer_py": gm(S["corr"])}


def pooled_arrays(P, Y):
    """Same metrics computed directly from (n, K) arrays, written independently of sample_stats/pooled."""
    P, Y = np.asarray(P, np.float64), np.asarray(Y, np.float64)
    E = P - Y
    keep = Y >= NEAR_ZERO
    corrs = [np.corrcoef(P[i], Y[i])[0, 1] for i in range(len(P))]
    return {"mse": E.__pow__(2).mean(), "rmse": np.sqrt(E.__pow__(2).mean()), "mae": np.abs(E).mean(),
            "mape_pct": 100 * (np.abs(E) / np.abs(Y)).mean(),
            f"mape_pct_targets_ge_{NEAR_ZERO}": 100 * (np.abs(E)[keep] / Y[keep]).mean(),
            "mape_pct_physical_irradiance": 100 * (np.abs(denorm(P) - denorm(Y)) / denorm(Y)).mean(),
            "r2_infer_py": 1 - (E ** 2).sum() / ((Y - Y.mean()) ** 2).sum(),
            "mean_spectral_corr_infer_py": float(np.mean([c for c in corrs if not np.isnan(c)]))}


def scored(stats, n_boot=N_BOOT, seed=0):
    """Point metrics, bootstrap 95% percentile intervals over samples, and the bootstrap SD."""
    S = stack_stats(stats)
    m = {k: float(v) for k, v in pooled(S).items()}
    n = len(stats)
    m["n_samples"] = n
    m["n_elements"] = int(S["n"].sum())
    m[f"fraction_elements_target_lt_{NEAR_ZERO}"] = float(1 - S["n_ge"].sum() / S["n"].sum())
    m["median_ape_pct_median_over_samples"] = float(np.median([s["median_ape_pct"] for s in stats]))
    if n >= 2:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        b = pooled(S, idx)
        m["ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in b.items()}
        m["bootstrap_sd"] = {k: float(np.nanstd(v)) for k, v in b.items()}
    return m


def weighted_point(stats, weights):
    """Block-weighted estimate (each sample stands for its block's share of the candidates)."""
    w = np.asarray(weights, float) / np.sum(weights)
    S = stack_stats(stats)
    n = S["n"]
    return {"mse": float(np.sum(w * S["sse"] / n)), "mae": float(np.sum(w * S["sae"] / n)),
            "mape_pct": float(100 * np.sum(w * S["sape"] / n))}


class _FakeModel:
    """Stands in for the network in the metric checks: returns preset outputs, one batch after another."""

    def __init__(self, outputs):
        import torch
        self.out = [torch.as_tensor(np.asarray(o, dtype=np.float32)).reshape(1, -1) for o in outputs]
        self.i = 0

    def eval(self):
        return self

    def __call__(self, batch):
        o = self.out[self.i]
        self.i += 1
        return o


def _grab(text, label):
    m = re.search(re.escape(label) + r"\s*\|\s*([-0-9.naN]+)", text)
    return float(m.group(1)) if m else float("nan")


def repo_infer_samples(P, Y, timestamps, save_path):
    """Run the authors' infer.infer_samples on given predictions/targets (replayed through a fake model) and return
    its printed summary, its per-sample printed ranges and the arrays it returns."""
    import torch
    sys.path.insert(0, TASK_DIR)
    import infer as I
    md = [{"timestamps_input": np.array([[np.datetime64(pd.Timestamp(t).strftime("%Y-%m-%dT%H:%M"))]]),
           "timestamps_targets": np.array([[np.datetime64((pd.Timestamp(t) + pd.Timedelta(minutes=60))
                                                          .strftime("%Y-%m-%dT%H:%M"))]])} for t in timestamps]
    loader = [({"label": torch.as_tensor(np.asarray(Y[i], np.float32)).reshape(1, -1),
                "ts": torch.zeros(1, 1, 1, 1, 1)}, md[i]) for i in range(len(Y))]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        preds, gts, _ = I.infer_samples(model=_FakeModel(P), local_rank=torch.device("cpu"), validation_loader=loader,
                                        gpu=True, dtype=torch.float32, save_path=save_path, device_type="cpu")
    text = buf.getvalue()
    ranges = re.findall(r"\|\s*\[([-0-9.]+), ([-0-9.]+)\]\s*\|\s*\[([-0-9.]+), ([-0-9.]+)\]", text)
    printed = {"mae": _grab(text, "Mean Absolute Error"), "rmse": _grab(text, "Root Mean Square Error"),
               "r2": _grab(text, "R² Score"), "avg_corr": _grab(text, "Average Spectral Correlation")}
    return printed, [tuple(float(v) for v in r) for r in ranges], np.asarray(preds), np.asarray(gts), text


def metric_self_test(out_dir):
    """Checks my metrics against finetune.evaluate_model and infer.infer_samples on synthetic inputs, checks MAPE on a
    hand example and against a plain loop, and checks my reading of the target transform against EVEDSDataset."""
    import torch
    import torch.distributed as dist
    sys.path.insert(0, TASK_DIR)
    import finetune as FT  # authors' evaluate_model
    rng = np.random.default_rng(0)
    n, K = 7, 1343
    Y = rng.uniform(0.25, 0.97, (n, K)).astype(np.float32)
    Y[0, -5:] = [0.0024, 0.0096, 0.012, 0.03, 0.049]  # floor-like targets, as in the real data near 33 nm
    P = np.clip(Y + rng.normal(0, 0.008, (n, K)), 0, 1).astype(np.float32)
    P[0, -5:] = 0.36
    report = {"n_synthetic_samples": n, "bins": K}
    mine = pooled(stack_stats([sample_stats(P[i], Y[i]) for i in range(n)]))
    direct = pooled_arrays(P, Y)
    report["sample_stats_vs_direct_max_rel_diff"] = max(abs(float(mine[k]) - float(direct[k])) / abs(float(direct[k]))
                                                         for k in direct)

    # (a) finetune.evaluate_model in a one-process gloo group (it calls dist.all_reduce). It reads the module-global
    #     `config` (set only under __main__ in finetune.py), so the keys it uses are set as finetune.main would have them.
    batches = [({"ts": torch.zeros(1, 1, 1, 1, 1), "label": torch.from_numpy(Y[i:i + 1])}, {}) for i in range(n)]
    tmp = tempfile.NamedTemporaryFile(prefix="gloo_", dir=WORK, delete=False).name
    os.remove(tmp)
    had = hasattr(FT, "config")
    FT.config = {"iters_per_epoch_valid": 200, "wandb_log_train_after": 10, "dtype": torch.float32}
    dist.init_process_group("gloo", init_method=f"file://{tmp}", rank=0, world_size=1)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            mae, rmse, r2, avg_loss = FT.evaluate_model(batches, 0, _FakeModel(P), "cpu", None, torch.nn.MSELoss())
    finally:
        dist.destroy_process_group()
        if os.path.exists(tmp):
            os.remove(tmp)
        if not had:
            del FT.config
    report["finetune_evaluate_model"] = {"repo": {"mae": mae, "rmse": rmse, "r2": r2, "avg_MSELoss": avg_loss},
                                         "mine": {"mae": float(mine["mae"]), "rmse": float(mine["rmse"]),
                                                  "r2": float(mine["r2_infer_py"]), "mse": float(mine["mse"])}}
    rel_a = max(abs(mae - mine["mae"]) / mine["mae"], abs(rmse - mine["rmse"]) / mine["rmse"],
                abs(r2 - mine["r2_infer_py"]) / abs(mine["r2_infer_py"]), abs(avg_loss - mine["mse"]) / mine["mse"])
    report["finetune_evaluate_model_max_rel_diff"] = float(rel_a)  # its accumulators are float32 tensors

    # (b) infer.infer_samples (prints 6 decimals and returns the arrays it scored)
    m3 = pooled(stack_stats([sample_stats(P[i], Y[i]) for i in range(3)]))
    printed, ranges, preds, gts, _ = repo_infer_samples(P[:3], Y[:3], ["2012-01-15 00:48"] * 3,
                                                        os.path.join(WORK, "tmp", "metric_check_infer_samples.png"))
    mine_b = {"mae": float(m3["mae"]), "rmse": float(m3["rmse"]), "r2": float(m3["r2_infer_py"]),
              "avg_corr": float(m3["mean_spectral_corr_infer_py"])}
    absd_b = max(abs(printed[k] - mine_b[k]) for k in mine_b)
    rng_b = max(max(abs(r[0] - P[i].min()), abs(r[1] - P[i].max()), abs(r[2] - Y[i].min()), abs(r[3] - Y[i].max()))
                for i, r in enumerate(ranges)) if len(ranges) == 3 else float("inf")
    report["infer_infer_samples"] = {"repo_printed_6dp": printed, "mine": mine_b, "max_abs_diff": absd_b,
                                     "printed_ranges_max_abs_diff": rng_b,
                                     "returned_arrays_equal_inputs": bool(np.array_equal(preds, P[:3])
                                                                          and np.array_equal(gts, Y[:3]))}

    # (c) hand example: targets 0.5, 0.25; predictions 0.55, 0.20 -> APE 10%, 20% -> MAPE 15%; MSE 0.0025, MAE 0.05
    h = sample_stats([0.55, 0.20], [0.5, 0.25])
    hand = max(abs(h["mape_pct"] - 15.0), abs(h["mse"] - 0.0025), abs(h["mae"] - 0.05))
    report["hand_example"] = {"mape_pct": h["mape_pct"], "mse": h["mse"], "mae": h["mae"],
                              "expected": {"mape_pct": 15.0, "mse": 0.0025, "mae": 0.05}}

    # (d) MAPE against a plain Python loop over every element (percent)
    tot, cnt = 0.0, 0
    for i in range(n):
        for k in range(K):
            tot += abs(float(P[i, k]) - float(Y[i, k])) / abs(float(Y[i, k]))
            cnt += 1
    loop_mape = 100.0 * tot / cnt
    report["mape_loop"] = {"loop": loop_mape, "mine": float(mine["mape_pct"]),
                           "rel_diff": abs(loop_mape - float(mine["mape_pct"])) / loop_mape}

    # (e) bootstrap sanity: the interval brackets the estimate; a resample of identical samples has zero width
    s = scored([sample_stats(P[i], Y[i]) for i in range(n)], n_boot=2000)
    s1 = scored([sample_stats(P[1], Y[1])] * 5, n_boot=200)
    boot_ok = all(s["ci95"][k][0] <= s[k] <= s["ci95"][k][1] for k in ("mse", "mae", "mape_pct")) and \
        all(abs(s1["ci95"]["mse"][j] - s1["mse"]) < 1e-15 for j in (0, 1))
    report["bootstrap_sanity_ok"] = bool(boot_ok)

    # (f) my reading of the target transform vs EVEDSDataset on a synthetic EVE file (with zeros, which the dataset
    #     replaces by the per-wavelength minimum of the split before log10)
    tr = check_target_transform()
    report["target_transform_vs_EVEDSDataset"] = tr

    report["passed"] = bool(report["sample_stats_vs_direct_max_rel_diff"] < 1e-10 and rel_a < 1e-5 and absd_b < 1e-6
                            and rng_b < 1e-4 and report["infer_infer_samples"]["returned_arrays_equal_inputs"]
                            and hand < 1e-12 and report["mape_loop"]["rel_diff"] < 1e-9 and boot_ok and tr["ok"])
    report["mape_definition"] = ("MAPE = 100/N * sum |yhat - y| / |y| over all N = n_samples x 1343 elements, "
                                 "normalised targets, percent; not in the repo; Table 5 / Sec. 2.6.4 of arXiv "
                                 "2508.14112 v2 name it without a formula")
    json.dump(report, open(os.path.join(out_dir, "metric_check.json"), "w"), indent=1, default=float)
    if not report["passed"]:
        raise AssertionError(f"metric self-test failed: {json.dumps(report, default=float)[:3000]}")
    return report


def check_target_transform():
    """Build EVEDSDataset on a small synthetic EVE netCDF and compare its normalised targets with
    (log10(E') - LOG_MIN) / (LOG_MAX - LOG_MIN), E' = E with zeros replaced by the per-wavelength minimum."""
    import xarray as xr
    sys.path.insert(0, TASK_DIR)
    from dataset import EVEDSDataset
    from surya.utils.data import build_scalers
    d = os.path.join(WORK, "tmp", "synthetic_eve")
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(1)
    t0 = pd.Timestamp("2012-01-15 00:48")
    eve_t = [t0 + pd.Timedelta(minutes=12 * i, seconds=4) for i in range(3)]
    E = (10 ** rng.uniform(-8.9, -2.5, (3, 1343))).astype(np.float32)
    E[0, 5] = 0.0
    E[2, 7] = 0.0
    nc = os.path.join(d, "eve.nc")
    xr.Dataset({"val_spectra": (("val_time", "wavelength"), E)},
               coords={"val_time": pd.DatetimeIndex(eve_t), "wavelength": np.arange(1343.0)}
               ).to_netcdf(nc, engine="h5netcdf")
    idx = os.path.join(d, "index.csv")
    C.write_index([t0 + pd.Timedelta(minutes=12 * i) for i in range(9)], idx)
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))
    scalers = build_scalers(info=yaml.safe_load(open(os.path.join(FM_DIR, "scalers.yaml"))))
    with contextlib.redirect_stdout(io.StringIO()):
        ds = EVEDSDataset(sdo_data_root_path=d, index_path=idx,
                          time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
                          time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
                          n_input_timestamps=cfg["model"]["time_embedding"]["time_dim"],
                          rollout_steps=cfg["rollout_steps"], channels=cfg["data"]["channels"],
                          drop_hmi_probablity=cfg["drop_hmi_probablity"],
                          num_mask_aia_channels=cfg["num_mask_aia_channels"],
                          use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"], scalers=scalers,
                          phase="train", ds_eve_index_path=nc, ds_time_column="val_time", ds_time_tolerance="6m",
                          ds_match_direction="forward")
    Ez = E.astype(np.float32).copy()
    Ez[Ez == 0] = np.nan
    mn = np.nanmin(Ez, axis=0)
    Ez = np.where(np.isnan(Ez), mn[None, :], Ez)
    mine = ((np.log10(Ez) - LOG_MIN) / (LOG_MAX - LOG_MIN)).astype(np.float32)
    got = np.stack(list(ds.df_valid_indices["normalized_spectrum"]))
    eve_got = list(pd.to_datetime(ds.df_valid_indices["ds_index"]))
    nonzero = E > 0
    inv = denorm(got)
    out = {"dataset_len": len(ds), "eve_times_match": eve_got == eve_t,
           "max_abs_diff_normalised": float(np.max(np.abs(got - mine))) if got.shape == mine.shape else None,
           "denorm_max_rel_diff_nonzero": float(np.max(np.abs(inv[nonzero] - E[nonzero]) / E[nonzero]))
           if got.shape == mine.shape else None,
           "zero_replaced_by_per_wavelength_min": bool(abs(inv[0, 5] - mn[5]) / mn[5] < 1e-5)
           if got.shape == mine.shape else None,
           "target_dtype": str(got.dtype)}
    out["ok"] = bool(len(ds) == 3 and out["eve_times_match"] and out["max_abs_diff_normalised"] is not None
                     and out["max_abs_diff_normalised"] <= 1e-7 and out["denorm_max_rel_diff_nonzero"] < 1e-5
                     and out["zero_replaced_by_per_wavelength_min"])
    return out


# -------------------------------------------------------------------------------------------- data and labels
def ensure_small_inputs(allow_eve_download=True):
    """core-sdo validation index (pinned revision), scalers.yaml, and the EVE netCDF where config.yaml expects them."""
    os.makedirs(ASSETS, exist_ok=True)
    sc_dst = os.path.join(ASSETS, "scalers.yaml")
    if not os.path.exists(sc_dst):
        # core-sdo's scalers.yaml (config.yaml scalers_path) and Surya-1.0's are byte-identical (sha256 a944e8f1...)
        with open(os.path.join(FM_DIR, "scalers.yaml"), "rb") as a, open(sc_dst, "wb") as b:
            b.write(a.read())
    fetch_small(hf_url(CORE_SDO_REPO, CORE_SDO_REV, "valid_index_surya_1_0.csv"),
                os.path.join(ASSETS, "valid_index_surya_1_0.csv"))
    return ensure_eve_file(allow_eve_download)


def ensure_eve_file(allow_download=True):
    """archive/AIA_EVE_temporally_stratified.nc (934 MB, the labels) at the pinned revision, sha256-checked once."""
    path = os.path.join(ASSETS, "euv-spectra", EVE_FILE)
    if not os.path.exists(path):
        if not allow_download:
            raise FileNotFoundError(path)
        if free_gb() - EVE_SIZE / 1e9 < 3.0:
            raise IOError(f"downloading {EVE_FILE} would leave less than 3 GB free on /")
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=EVE_REPO, repo_type="dataset", filename=EVE_FILE, revision=EVE_REV,
                               local_dir=os.path.join(ASSETS, "euv-spectra"))
    if os.path.getsize(path) != EVE_SIZE:
        raise IOError(f"{path}: {os.path.getsize(path)} bytes, expected {EVE_SIZE}")
    marker = path + ".sha256_ok"
    if not (os.path.exists(marker) and open(marker).read().strip() == f"{EVE_SHA256} {os.path.getmtime(path)}"):
        sha = sha256_file(path)
        if sha != EVE_SHA256:
            raise IOError(f"{path}: sha256 {sha}, expected {EVE_SHA256}")
        open(marker, "w").write(f"{EVE_SHA256} {os.path.getmtime(path)}")
    return {"path": path, "bytes": EVE_SIZE, "sha256": EVE_SHA256, "repo": f"{EVE_REPO}@{EVE_REV}"}


def eve_file_facts():
    """Variables, time coverage and split rules of the EVE file (recorded in sampling_<split>.json)."""
    import h5py
    import xarray as xr
    p = os.path.join(ASSETS, "euv-spectra", EVE_FILE)
    out = {}
    with h5py.File(p, "r") as f:
        out["variables"] = {k: [list(f[k].shape), str(f[k].dtype)] for k in f.keys()}
        out["attrs"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in f.attrs.items()
                        if not k.startswith("_")}
        out["attrs"] = {k: (v.decode() if isinstance(v, bytes) else (v[0] if isinstance(v, list) and len(v) == 1
                                                                       else v)) for k, v in out["attrs"].items()}
    with xr.open_dataset(p, engine="h5netcdf", cache=False, lock=False) as ds:
        for k in ("train_time", "val_time", "test_time"):
            t = pd.Series(pd.to_datetime(ds[k].values))
            out[k] = {"rows": int(len(t)), "first": str(t.min()), "last": str(t.max()),
                      "rows_per_year": {int(y): int(c) for y, c in t.dt.year.value_counts().sort_index().items()},
                      "share_of_1_minute_steps": float((t.diff() == pd.Timedelta(minutes=1)).mean())}
        wl = ds["wavelength"].values
        out["wavelength_nm"] = {"n": int(len(wl)), "first": float(wl[0]), "last": float(wl[-1])}
    return out


def load_configs():
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))  # finetune.py's config (dataset arguments)
    cfg_inf = yaml.safe_load(open(os.path.join(TASK_DIR, "config_infer.yaml")))  # infer.py's default (model)
    diff = {}
    for sec in ("model", "data"):
        for k in sorted(set(cfg[sec]) | set(cfg_inf[sec])):
            if cfg[sec].get(k) != cfg_inf[sec].get(k):
                diff[f"{sec}.{k}"] = (cfg[sec].get(k), cfg_inf[sec].get(k))
    for k in sorted(set(cfg) | set(cfg_inf)):
        if k not in ("model", "data") and cfg.get(k) != cfg_inf.get(k):
            diff[k] = (cfg.get(k), cfg_inf.get(k))
    return cfg, cfg_inf, diff


def import_dataset_class():
    sys.path.insert(0, TASK_DIR)
    import dataset as D
    if os.path.dirname(os.path.abspath(D.__file__)) != TASK_DIR:
        raise ImportError(f"'dataset' resolved to {D.__file__}, not the EUV task's dataset.py")
    return D.EVEDSDataset


def build_eve_dataset(cfg, scalers, split, index_path, sdo_root, quiet=True):
    """The authors' EVEDSDataset with finetune.get_dataloaders' validation arguments (config.yaml): phase "train",
    tolerance "6m", direction "forward"; ds_time_column val_time (validation) or test_time (test)."""
    EVEDSDataset = import_dataset_class()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        ds = EVEDSDataset(
            sdo_data_root_path=sdo_root,
            index_path=index_path,
            time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
            n_input_timestamps=cfg["model"]["time_embedding"]["time_dim"],
            rollout_steps=cfg["rollout_steps"],
            channels=cfg["data"]["channels"],
            drop_hmi_probablity=cfg["drop_hmi_probablity"],
            num_mask_aia_channels=cfg["num_mask_aia_channels"],
            use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"],
            scalers=scalers,
            phase="train",
            ds_eve_index_path=cfg["data"]["valid_solar_data_path"],
            ds_time_column=SPLIT_TIME[split],
            ds_time_tolerance="6m",
            ds_match_direction="forward",
        )
    return ds


def one_sample_dataset(cfg, scalers, split, t, sdo_root):
    t = pd.Timestamp(t)
    index_path = os.path.join(WORK, "index", f"{split}_{stamp(t)}.csv")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    C.write_index([t, t + pd.Timedelta(minutes=cfg["data"]["time_delta_target_minutes"])], index_path)
    return build_eve_dataset(cfg, scalers, split, index_path, sdo_root)


_FULL = {}


def full_split(cfg, scalers, split):
    """All (t, EVE) pairs the authors' code forms for the split over core-sdo's validation index, with targets, raw
    EVE values and the train-mean baseline per pair. Cached per split."""
    if split in _FULL:
        return _FULL[split]
    import xarray as xr
    ds = build_eve_dataset(cfg, scalers, split, cfg["data"]["valid_data_path"], os.path.join(WORK, "frames", "a"))
    df = ds.df_valid_indices.reset_index()
    with xr.open_dataset(cfg["data"]["valid_solar_data_path"], engine="h5netcdf", cache=False, lock=False) as x:
        times = pd.to_datetime(x[SPLIT_TIME[split]].values)
        raw = x[SPLIT_SPECTRA[split]].values
        wl = x["wavelength"].values.astype(np.float64)
    pos = pd.Series(np.arange(len(times)), index=times)
    Y = np.stack(list(df["normalized_spectrum"])).astype(np.float32)
    R = raw[pos.reindex(df["ds_index"]).values.astype(int)]
    base = train_mean_spectrum(cfg)
    stats = [sample_stats(base, Y[i]) for i in range(len(Y))]
    cand = pd.DataFrame({"t": pd.to_datetime(df["valid_indices"]), "eve_time": pd.to_datetime(df["ds_index"]),
                         "eve_minus_sdo_s": (pd.to_datetime(df["ds_index"]) - pd.to_datetime(df["valid_indices"]))
                         .dt.total_seconds()})
    cand["year"] = cand["t"].dt.year
    cand["target_mean"] = Y.mean(1)
    cand["target_min"] = Y.min(1)
    cand["n_bins_target_lt_near_zero"] = (Y < NEAR_ZERO).sum(1)
    cand["baseline_mse"] = [s["mse"] for s in stats]
    cand["baseline_mae"] = [s["mae"] for s in stats]
    cand["baseline_mape_pct"] = [s["mape_pct"] for s in stats]
    cand = cand.sort_values("t").reset_index(drop=True)
    order = {t: i for i, t in enumerate(pd.to_datetime(df["valid_indices"]))}
    rows = np.array([order[t] for t in cand["t"]])
    _FULL[split] = {"cand": cand, "Y": Y[rows], "raw": R[rows], "stats": [stats[i] for i in rows], "wl": wl,
                    "n_rows_eve_split": int(len(times)), "baseline": base,
                    "denorm_vs_raw_max_rel_diff": float(np.max(np.abs(denorm(Y) - R) / R)),
                    "transform_vs_mine_max_abs_diff": float(np.max(np.abs(
                        Y - ((np.log10(R.astype(np.float32)) - LOG_MIN) / (LOG_MAX - LOG_MIN)).astype(np.float32))))}
    return _FULL[split]


def train_mean_spectrum(cfg):
    """MY baseline: mean normalised training spectrum (train_spectra through the dataset's transform: zeros -> the
    per-wavelength minimum, log10, fixed min-max), read in row chunks and cached."""
    cache = os.path.join(WORK, "meta", f"train_mean_spectrum_{EVE_SHA256[:12]}.npz")
    if os.path.exists(cache):
        return np.load(cache)["mean"]
    import h5py
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with h5py.File(cfg["data"]["valid_solar_data_path"], "r") as f:
        d = f["train_spectra"]
        nrow = d.shape[0]
        mn = np.full(d.shape[1], np.inf)
        for a in range(0, nrow, 20000):  # pass 1: per-wavelength minimum of the non-zero values
            x = d[a:a + 20000].astype(np.float32)
            mn = np.minimum(mn, np.where((x != 0) & ~np.isnan(x), x, np.inf).min(0))
        acc = np.zeros(d.shape[1])
        for a in range(0, nrow, 20000):  # pass 2: mean of the normalised spectra
            x = d[a:a + 20000].astype(np.float32)
            x = np.where((x == 0) | np.isnan(x), mn[None, :].astype(np.float32), x)
            acc += ((np.log10(x) - LOG_MIN) / (LOG_MAX - LOG_MIN)).astype(np.float64).sum(0)
    mean = acc / nrow
    np.savez(cache, mean=mean, n_rows=nrow)
    return mean


# ------------------------------------------------------------------------------------------------- sampling
def allocate(counts, n, rng):
    """Largest-remainder allocation of n proportional to counts (dict year -> count); ties broken by rng."""
    years = sorted(counts)
    tot = sum(counts.values())
    q = {y: n * counts[y] / tot for y in years}
    a = {y: int(np.floor(q[y])) for y in years}
    rem = n - sum(a.values())
    order = sorted(years, key=lambda y: (-(q[y] - a[y]), rng.random()))
    for y in order[:rem]:
        a[y] += 1
    return a


def design_blocks(cand, n, seed):
    rng = np.random.default_rng(seed)
    counts = {int(y): int(c) for y, c in cand["year"].value_counts().items()}
    alloc = allocate(counts, n, rng)
    blocks = []
    for y in sorted(alloc):
        c = cand[cand["year"] == y].sort_values("t").reset_index(drop=True)
        k = alloc[y]
        edges = np.linspace(0, len(c), k + 1).round().astype(int)
        for b in range(k):
            sub = c.iloc[edges[b]:edges[b + 1]]
            blocks.append({"year": y, "block": b, "first": str(sub["t"].iloc[0]), "last": str(sub["t"].iloc[-1]),
                           "n_candidates": int(len(sub)), "order": [str(t) for t in sub["t"].iloc[rng.permutation(len(sub))]]})
    return alloc, blocks


def head_check(t):
    """HEAD check of the frames at t and t+60 (EVEDSDataset indexes and reads both)."""
    t = pd.Timestamp(t)
    extra = {}
    for which, ts in (("t", t), ("t+60", t + pd.Timedelta(minutes=60))):
        ok, nb = s3_exists_retry(ts)
        if not ok:
            return False, f"frame at {which} not on S3" + (" (EVEDSDataset indexes and reads t+60)" if which != "t"
                                                           else ""), extra
        extra["frame_bytes" if which == "t" else "frame_t60_bytes"] = nb
    return True, "", extra


def check_candidates(times, check_channels, tp, pp):
    """HEAD checks (threads) and 13-channel netCDF header checks of both frames (spawned processes: h5py holds a
    global lock, so header reads in threads would run one at a time). Returns [(ok, reason, extra)] in input order."""
    heads = list(tp.map(head_check, times))
    out = []
    futs = {}
    if check_channels:
        for j, h in enumerate(heads):
            if h[0]:
                t = pd.Timestamp(times[j])
                futs[j] = (pp.submit(remote_nc_channels, C.s3_key(t)),
                           pp.submit(remote_nc_channels, C.s3_key(t + pd.Timedelta(minutes=60))))
    for j, (ok, why, extra) in enumerate(heads):
        extra = dict(extra)
        if ok and check_channels:
            nread = 0
            for which, f in zip(("t", "t+60"), futs[j]):
                try:
                    shapes, nb, _ = f.result()
                except Exception as e:
                    ok, why = False, f"could not read the netCDF header of the frame at {which}: {e!r}"
                    break
                nread += nb
                bad = [c for c in C.CHANNELS if shapes.get(c) != (4096, 4096)]
                if bad:
                    ok, why = False, f"frame at {which} lacks channels {bad}"
                    break
            extra["header_bytes_read"] = nread
        out.append((ok, why, extra))
    return out


def select_sample(cfg, scalers, split, n, seed, check_channels=True, workers=4, want_per_block=2, max_tries=30):
    """Seeded, year-proportional, time-blocked sample with one primary and one reserve per block (module docstring).
    Candidates are tried in each block's seeded order, in rounds; the assignment is deterministic."""
    import multiprocessing as mp
    F = full_split(cfg, scalers, split)
    cand = F["cand"].set_index("t")
    alloc, blocks = design_blocks(F["cand"], n, seed)
    state = [{"pos": 0, "found": [], "skipped": []} for _ in blocks]
    ctx = mp.get_context("spawn")
    anchors, anchor_skips = [], []
    vcand = full_split(cfg, scalers, "validation")["cand"].set_index("t")
    with cf.ThreadPoolExecutor(8) as tp, cf.ProcessPoolExecutor(workers, mp_context=ctx) as pp:
        while True:
            batch = []  # (block index, candidate time)
            for bi, (blk, st) in enumerate(zip(blocks, state)):
                need = want_per_block - len(st["found"])
                while need > 0 and st["pos"] < min(len(blk["order"]), max_tries):
                    batch.append((bi, blk["order"][st["pos"]]))
                    st["pos"] += 1
                    need -= 1
            if not batch:
                break
            res = check_candidates([ts for _, ts in batch], check_channels, tp, pp)
            for (bi, ts), (ok, why, extra) in zip(batch, res):  # in permutation order -> deterministic
                if ok and len(state[bi]["found"]) < want_per_block:
                    state[bi]["found"].append((ts, extra))
                elif not ok:
                    state[bi]["skipped"].append({"timestamp": ts, "year": blocks[bi]["year"],
                                                 "block": blocks[bi]["block"], "reason": why})
        valid_pairs = set(vcand.index)
        a_times = [a["timestamp"] for a in NOTEBOOK_ANCHORS]
        a_res = check_candidates(a_times, check_channels, tp, pp)
    rows, skipped = [], []
    for b, st in zip(blocks, state):
        skipped += st["skipped"]
        for role, (ts, extra) in zip(("primary", "reserve"), st["found"]):
            rows.append(sample_row(cand, ts, split, role, b, extra, check_channels))
    for i, (ts, (ok, why, extra)) in enumerate(zip(a_times, a_res)):
        if pd.Timestamp(ts) not in valid_pairs:
            anchor_skips.append({"timestamp": ts, "reason": "not one of the authors' val_time pairs"})
        elif not ok:
            anchor_skips.append({"timestamp": ts, "reason": why})
        else:
            b = {"year": pd.Timestamp(ts).year, "block": -(i + 1), "first": ts, "last": ts, "n_candidates": 1}
            anchors.append(sample_row(vcand, ts, "validation", "anchor", b, extra, check_channels))
    for b, st in zip(blocks, state):
        b["n_found"] = len(st["found"])
        b["n_tried"] = st["pos"]
    return alloc, blocks, pd.DataFrame(rows), pd.DataFrame(anchors), skipped, anchor_skips


def sample_row(cand, ts, split, role, b, extra, check_channels):
    t = pd.Timestamp(ts)
    c = cand.loc[t]
    return {"split": split, "role": role, "stratum": int(b["year"]), "block": int(b["block"]),
            "block_first": b["first"], "block_last": b["last"], "block_candidates": int(b["n_candidates"]),
            "timestamp": str(t), "eve_time": str(c["eve_time"]), "eve_minus_sdo_s": float(c["eve_minus_sdo_s"]),
            "target_mean": float(c["target_mean"]), "target_min": float(c["target_min"]),
            "n_bins_target_lt_near_zero": int(c["n_bins_target_lt_near_zero"]),
            "baseline_mse": float(c["baseline_mse"]), "baseline_mae": float(c["baseline_mae"]),
            "baseline_mape_pct": float(c["baseline_mape_pct"]),
            "frame_key": C.s3_key(t), "frame_bytes": int(extra.get("frame_bytes", 0)),
            "frame_t60_key": C.s3_key(t + pd.Timedelta(minutes=60)),
            "frame_t60_bytes": int(extra.get("frame_t60_bytes", 0)), "channels_checked": bool(check_channels),
            "header_bytes_read": int(extra.get("header_bytes_read", 0))}


def check_joins(sample, cfg, scalers):
    """For every sampled row, a one-sample EVEDSDataset (index with t and t+60 only; no frame is read) must select t,
    pair it with the same EVE time as the full-split dataset and give a bit-identical target vector."""
    out = []
    for r in sample.itertuples():
        F = full_split(cfg, scalers, r.split)
        i = int(np.flatnonzero(F["cand"]["t"].values == np.datetime64(pd.Timestamp(r.timestamp)))[0])
        ds = one_sample_dataset(cfg, scalers, r.split, r.timestamp, os.path.join(WORK, "frames", "a"))
        ok = len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == pd.Timestamp(r.timestamp)
        row = ds.df_valid_indices.iloc[0] if len(ds) else None
        same_eve = row is not None and pd.Timestamp(row["ds_index"]) == pd.Timestamp(r.eve_time)
        y = np.asarray(row["normalized_spectrum"]) if row is not None else None
        same_y = y is not None and np.array_equal(y, F["Y"][i])
        rec = {"timestamp": r.timestamp, "split": r.split, "role": r.role, "dataset_len": len(ds),
               "eve_time_from_dataset": str(row["ds_index"]) if row is not None else None,
               "same_eve_time_as_full_split": bool(same_eve), "target_bit_identical_to_full_split": bool(same_y),
               "target_dtype": str(y.dtype) if y is not None else None,
               "target_range_4dp": [round(float(y.min()), 4), round(float(y.max()), 4)] if y is not None else None,
               "sdo_paths_read_by_dataset": list(ds.index["path"])}
        rec["ok"] = bool(ok and same_eve and same_y)
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------ model
def ensure_checkpoint():
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=CKPT_REPO, filename=CKPT_NAME, revision=CKPT_REV, local_dir=WORK)
    if os.path.getsize(path) != CKPT_SIZE:
        raise IOError(f"{path}: {os.path.getsize(path)} bytes, expected {CKPT_SIZE}")
    marker = path + ".sha256_ok"
    if not (os.path.exists(marker) and open(marker).read().strip() == f"{CKPT_SHA256} {os.path.getmtime(path)}"):
        sha = sha256_file(path)
        if sha != CKPT_SHA256:
            raise IOError(f"{path}: sha256 {sha}, expected {CKPT_SHA256}")
        open(marker, "w").write(f"{CKPT_SHA256} {os.path.getmtime(path)}")
    return path


def ensure_pretrained_link(cfg_inf):
    """finetune.get_model raises unless config pretrained_path (./assets/surya.366m.v1.pt) exists; it loads the
    foundation weights, which the strict load of the fine-tuned checkpoint then overwrites. Link the copy on disk."""
    dst = os.path.join(WORK, cfg_inf["pretrained_path"])
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(os.path.join(FM_DIR, "surya.366m.v1.pt"), dst)
    return dst


def load_model(cfg_inf, ckpt_path, allow_full_unpickle=False):
    """infer.load_model, step by step, with a safe unpickler and memory-mapped weights."""
    import torch
    sys.path.insert(0, TASK_DIR)
    import infer as I  # authors' get_model (imported by infer.py from finetune.py)
    from surya.utils.distributed import set_global_seed
    set_global_seed(42)  # as infer.main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model = I.get_model(cfg_inf, wandb_logger=None)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        how = "weights_only=True, mmap=True"
    except Exception as e:
        if not allow_full_unpickle:
            raise RuntimeError(f"weights_only load failed ({e!r}); rerun with --allow-full-unpickle to load it the way "
                               "infer.py does (weights_only=False)") from e
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        how = "weights_only=False (as infer.py)"
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        model_state = checkpoint["state_dict"]
    else:
        model_state = checkpoint
    if any(k.startswith("module.") for k in model_state.keys()):
        model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    model.load_state_dict(model_state, strict=True)
    n_keys = len(model_state)
    del checkpoint, model_state
    gc.collect()
    model.to(torch.device("cpu"))
    model.eval()
    model.requires_grad_(False)
    return model, {"n_parameters": int(n_total), "n_trainable_after_get_model": int(n_trainable),
                   "state_dict_keys": int(n_keys), "load_state_dict_strict": True, "torch_load": how,
                   "get_model_stdout": buf.getvalue()[-2000:]}


def verify_frames(paths, channels):
    """Fail fast before the authors' __getitem__ (which retries a failing index up to 100 times, re-reading frames):
    each file must open and hold every channel at 4096x4096."""
    import xarray as xr
    for p in paths:
        with xr.open_dataset(p, engine="h5netcdf", chunks=None, cache=False) as ds:
            bad = [c for c in channels if c not in ds or tuple(ds[c].shape) != (4096, 4096)]
        if bad:
            raise IOError(f"{os.path.basename(p)} lacks channels {bad}")


# ---------------------------------------------------------------------------------------------- bookkeeping
def per_sample_path(out_dir, split):
    return os.path.join(out_dir, f"per_sample_{split}.jsonl")


def anchors_path(out_dir):
    return os.path.join(out_dir, "anchors.jsonl")


def read_records(path):
    recs = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["timestamp"]] = r  # the last record for a time stamp wins
    return recs


def anchor_check(arecs):
    """Our anchor predictions through the authors' infer.infer_samples vs the notebook's printout."""
    by_t = {a["timestamp"]: a for a in NOTEBOOK_ANCHORS}
    ok_recs = [arecs[t] for t in by_t if t in arecs and arecs[t].get("status") == "ok"]
    out = {"n_expected": len(NOTEBOOK_ANCHORS), "n_run": len(ok_recs), "tolerances": ANCHOR_TOL, "per_sample": []}
    if not ok_recs:
        return out
    P = np.array([r["pred"] for r in ok_recs], dtype=np.float32)
    Y = np.array([r["target"] for r in ok_recs], dtype=np.float32)
    printed, ranges, _, _, _ = repo_infer_samples(P, Y, [r["timestamp"] for r in ok_recs],
                                                  os.path.join(WORK, "tmp", "anchors_infer_samples.png"))
    all_ok = True
    for r, rg in zip(ok_recs, ranges):
        nb = by_t[r["timestamp"]]
        dp = max(abs(rg[0] - nb["pred_range"][0]), abs(rg[1] - nb["pred_range"][1]))
        dg = max(abs(rg[2] - nb["gt_range"][0]), abs(rg[3] - nb["gt_range"][1]))
        ok = dp <= ANCHOR_TOL["pred_range"] and dg <= ANCHOR_TOL["gt_range"]
        all_ok &= ok
        out["per_sample"].append({"timestamp": r["timestamp"], "eve_time": r["eve_time"],
                                  "ours_printed_by_infer_samples": rg, "notebook_pred_range": nb["pred_range"],
                                  "notebook_gt_range": nb["gt_range"], "pred_range_max_abs_diff": dp,
                                  "gt_range_max_abs_diff": dg, "within_tolerance": bool(ok)})
    out["summary_ours_printed_by_infer_samples"] = printed
    out["summary_notebook"] = NOTEBOOK_SUMMARY
    if len(ok_recs) == len(NOTEBOOK_ANCHORS):
        d = {k: abs(printed[k] - NOTEBOOK_SUMMARY[k]) for k in ("mae", "rmse", "r2", "avg_corr")}
        out["summary_abs_diff"] = d
        all_ok &= all(d[k] <= ANCHOR_TOL[k] for k in d)
    else:
        all_ok = False
    out["all_within_tolerance"] = bool(all_ok)
    return out


def compare_with_paper(m):
    ref = PAPER_TABLE5["Surya"]
    out = {}
    for ours, key in (("mse", "mse"), ("mae", "mae"), ("mape_pct", "mape")):
        lo, hi = m.get("ci95", {}).get(ours, [None, None])
        out[key] = {"ours": m[ours], "ci95": [lo, hi], "paper_surya": ref[key],
                    "ratio_ours_over_paper": m[ours] / ref[key],
                    "ratio_ci95": [lo / ref[key], hi / ref[key]] if lo is not None else None,
                    "paper_inside_ci95": bool(lo <= ref[key] <= hi) if lo is not None else None}
    return out


def summarize(out_dir, seed=0):
    import xarray as xr
    summary = {"what": "PARTIAL REPRODUCTION of arXiv 2508.14112 v2 Table 5 (EUV spectra, Surya row) on a small "
                       "seeded sample, CPU float32",
               "paper_table5": PAPER_TABLE5, "paper_text": PAPER_TEXT,
               "metric_definitions": {
                   "mse": "mean of (yhat - y)^2 over all n_samples x 1343 elements, normalised targets "
                          "y = (log10 E + 9) / 7.04 (pooling as finetune.evaluate_model)",
                   "mae": "mean of |yhat - y| over the same elements",
                   "mape_pct": "100 * mean of |yhat - y| / |y| over the same elements (percent; not in the repo; "
                               "Table 5 / Sec. 2.6.4 name MAPE without a formula); no epsilon, min target 0.0024",
                   f"mape_pct_targets_ge_{NEAR_ZERO}": f"same, elements with y >= {NEAR_ZERO} only (variant)",
                   "mape_pct_physical_irradiance": "same on 10^(7.04 y - 9) W/m^2/nm (variant)",
                   "r2_infer_py": "1 - SSE / SST over all elements (infer.py and finetune.evaluate_model)",
                   "mean_spectral_corr_infer_py": "mean over samples of Pearson r across the 1343 bins (infer.py)",
                   "ci95": f"percentile bootstrap over samples, {N_BOOT} resamples, seed {seed}"},
               "checkpoint": CKPT_HISTORY, "splits": {}}
    wl = None
    p = os.path.join(ASSETS, "euv-spectra", EVE_FILE)
    if os.path.exists(p):
        with xr.open_dataset(p, engine="h5netcdf", cache=False, lock=False) as x:
            wl = x["wavelength"].values.astype(float)
    arecs = read_records(anchors_path(out_dir))
    if arecs:
        summary["notebook_anchor_check"] = anchor_check(arecs)
    for split in ("test", "validation"):
        sp = {"ds_time_column": SPLIT_TIME[split]}
        sj = os.path.join(out_dir, f"sampling_{split}.json")
        if os.path.exists(sj):
            sjd = json.load(open(sj))
            sp["n_candidates_full_split"] = sjd.get("n_candidates_authors_join")
            sp["baseline_train_mean_on_all_candidates"] = sjd.get("baseline_train_mean_on_all_candidates")
            sp["sample_design"] = {k: sjd.get(k) for k in ("n_requested", "seed", "allocation", "n_primary",
                                                           "n_reserve", "n_skipped")}
        recs = read_records(per_sample_path(out_dir, split))
        ok = [r for r in recs.values() if r.get("status") == "ok" and r.get("role") in ("primary", "reserve")]
        # one sample per block: the primary if it was scored, else the reserve
        by_block = {}
        for r in sorted(ok, key=lambda r: r["role"] != "primary"):
            by_block.setdefault((r["stratum"], r["block"]), r)
        use = sorted(by_block.values(), key=lambda r: r["timestamp"])
        sp["n_records"] = len(recs)
        sp["n_failed"] = sum(1 for r in recs.values() if r.get("status") != "ok")
        sp["failed"] = [{"timestamp": r["timestamp"], "error": r.get("error", "")[:300]}
                        for r in recs.values() if r.get("status") != "ok"]
        sp["n_samples_scored"] = len(use)
        if not use:
            sp["status"] = "no samples scored yet"
            summary["splits"][split] = sp
            continue
        P = np.array([r["pred"] for r in use], dtype=np.float32)
        Y = np.array([r["target"] for r in use], dtype=np.float32)
        stats = [sample_stats(P[i], Y[i], wl) for i in range(len(use))]
        m = scored(stats, seed=seed)
        mb = scored([sample_stats(np.asarray(r["baseline_pred"], np.float64), Y[i], wl) for i, r in enumerate(use)],
                    seed=seed) if all("baseline_pred" in r for r in use) else None
        sp["metrics"] = {"model": m, "baseline_train_mean_spectrum_same_samples": mb,
                         "model_block_weighted_point": weighted_point(stats, [r["block_candidates"] for r in use])}
        sp["comparison_with_paper_table5_surya"] = compare_with_paper(m)
        if mb:
            sp["model_vs_baseline_mse_ratio"] = m["mse"] / mb["mse"]
        sp["worst_elements_for_ape"] = sorted(
            [{"timestamp": r["timestamp"], "max_ape_pct": s["max_ape_pct"], "wavelength_nm": s["max_ape_wavelength_nm"],
              "target": s["max_ape_target"], "pred": s["max_ape_pred"]} for r, s in zip(use, stats)],
            key=lambda d: -d["max_ape_pct"])[:5]
        sp["per_sample"] = [{"timestamp": r["timestamp"], "eve_time": r["eve_time"], "role": r["role"],
                             "mse": s["mse"], "mae": s["mae"], "mape_pct": s["mape_pct"], "corr": s["corr"]}
                            for r, s in zip(use, stats)]
        sp["seconds"] = {k: [float(np.median([r["seconds"][k] for r in use])), float(max(r["seconds"][k] for r in use))]
                         for k in ("dataset_and_load", "forward", "total")}
        sp["seconds_note"] = "[median, max] per sample"
        sp["peak_rss_gb"] = max(r.get("peak_rss_gb", 0) for r in use)
        sp["target_checks_all_ok"] = all(r.get("target_checks", {}).get("ok", False) for r in use)
        if wl is not None:  # per-wavelength profile over the scored samples
            E = P.astype(np.float64) - Y.astype(np.float64)
            pd.DataFrame({"wavelength_nm": wl, "mean_target": Y.mean(0), "mean_pred": P.mean(0),
                          "mse": (E ** 2).mean(0), "mae": np.abs(E).mean(0),
                          "mape_pct": 100 * (np.abs(E) / Y).mean(0)}).to_csv(
                os.path.join(out_dir, f"per_wavelength_{split}.csv"), index=False, float_format="%.6g")
        summary["splits"][split] = sp
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    return summary


# ---------------------------------------------------------------------------------------------- sample I/O
def sample_csv(out_dir, split):
    return os.path.join(out_dir, f"sample_{split}.csv")


def get_sample(args, out_dir, cfg, scalers):
    """Read sample_<split>.csv written by the dry run (same seed and n), or select it now."""
    p = sample_csv(out_dir, args.split)
    sj = os.path.join(out_dir, f"sampling_{args.split}.json")
    if os.path.exists(p) and os.path.exists(sj):
        meta = json.load(open(sj))
        if meta.get("seed") == args.seed and meta.get("n_requested") == args.n:
            return pd.read_csv(p), meta
    log("no matching sample file; selecting the sample now (as --dry-run would)")
    return dry_run(args, out_dir, run_mode=True)


def bandwidth_probe(key):
    """8 MB range read of one S3 frame (the dry run reads nothing of the checkpoint; its download time is estimated
    with the same rate)."""
    out = {}
    try:
        t0 = time.time()
        r = requests.get(f"{C.S3}/{key}", headers={"Range": "bytes=0-8388607"}, timeout=120)
        r.raise_for_status()
        dt = max(1e-3, time.time() - t0)
        out["s3_frame"] = {"bytes": len(r.content), "seconds": round(dt, 2),
                           "MB_per_s": round(len(r.content) / 1e6 / dt, 1)}
    except Exception as e:
        out["s3_frame"] = {"error": repr(e)[:300]}
    return out


def dry_run(args, out_dir, run_mode=False):
    from surya.utils.data import build_scalers
    t_start = time.time()
    eve = ensure_small_inputs()
    cfg, cfg_inf, diff = load_configs()
    scalers = build_scalers(info=yaml.safe_load(open(cfg["data"]["scalers_path"])))
    mc = metric_self_test(out_dir)
    log(f"metric self-test passed: evaluate_model rel diff {mc['finetune_evaluate_model_max_rel_diff']:.1e}, "
        f"infer_samples abs diff {mc['infer_infer_samples']['max_abs_diff']:.1e}")
    facts = eve_file_facts()
    F = full_split(cfg, scalers, args.split)
    cand = F["cand"]
    log(f"{args.split}: {F['n_rows_eve_split']} EVE rows in {SPLIT_TIME[args.split]}, {len(cand)} (t, EVE) pairs "
        f"from the authors' join over core-sdo's validation index")
    alloc, blocks, sample, anchors, skipped, anchor_skips = select_sample(
        cfg, scalers, args.split, args.n, args.seed, not args.no_channel_check, args.header_workers)
    if args.no_anchors:
        anchors = anchors.iloc[0:0]
    allrows = pd.concat([sample, anchors], ignore_index=True)
    joins = check_joins(allrows, cfg, scalers)
    allrows["dataset_join_ok"] = [j["ok"] for j in joins]
    anchor_gt = []
    for a in NOTEBOOK_ANCHORS:
        j = next((j for j in joins if j["timestamp"] == a["timestamp"]), None)
        if j is not None:
            d = max(abs(j["target_range_4dp"][0] - a["gt_range"][0]), abs(j["target_range_4dp"][1] - a["gt_range"][1]))
            anchor_gt.append({"timestamp": a["timestamp"], "eve_time": j["eve_time_from_dataset"],
                              "dataset_target_range_4dp": j["target_range_4dp"], "notebook_gt_range": a["gt_range"],
                              "max_abs_diff": d, "equal_at_4dp": bool(d <= ANCHOR_TOL["gt_range"])})
    allrows.to_csv(sample_csv(out_dir, args.split), index=False)
    pd.DataFrame(skipped + anchor_skips).to_csv(os.path.join(out_dir, f"skipped_{args.split}.csv"), index=False)

    prim = allrows[allrows["role"] == "primary"]
    anc = allrows[allrows["role"] == "anchor"]
    anc_extra = anc[~anc["timestamp"].isin(prim["timestamp"])] if args.split == "validation" else anc
    run_rows = pd.concat([anc_extra, prim])
    frames_bytes = int((run_rows["frame_bytes"] + run_rows["frame_t60_bytes"]).sum())
    per_sample_bytes = int((run_rows["frame_bytes"] + run_rows["frame_t60_bytes"]).max()) if len(run_rows) else 0
    n_pass = len(run_rows)
    est_s = SEC_MODEL + n_pass * (SEC_FORWARD + 2 * SEC_LOAD_FRAME + SEC_SCORE)
    bw = bandwidth_probe(run_rows["frame_key"].iloc[0]) if len(run_rows) else {}
    mbps = bw.get("s3_frame", {}).get("MB_per_s")
    dl_s = per_sample_bytes / 1e6 / mbps if mbps else None
    per_sample_s = max(SEC_FORWARD + 2 * SEC_LOAD_FRAME + SEC_SCORE, dl_s or 0)  # downloads overlap (prefetch)
    est_s_bw = SEC_MODEL + (dl_s or 0) + n_pass * per_sample_s
    ckpt_s = CKPT_SIZE / 1e6 / mbps if mbps and not os.path.exists(os.path.join(WORK, CKPT_NAME)) else None
    ckpt_present = os.path.exists(os.path.join(WORK, CKPT_NAME))
    free = free_gb()
    need_ckpt = 0 if ckpt_present else CKPT_SIZE / 1e9
    prefetch_ok = free - need_ckpt - 2 * per_sample_bytes / 1e9 >= DISK_MARGIN_GB and \
        2 * per_sample_bytes / 1e9 <= FRAME_BUDGET_GB
    fit90 = int(max(0, (90 * 60 - SEC_MODEL - (ckpt_s or 0)) // per_sample_s))
    base_all = scored(F["stats"], seed=args.seed)
    rep = {}
    for col in ("target_mean", "baseline_mse", "eve_minus_sdo_s"):
        full_mean, full_sd = float(cand[col].mean()), float(cand[col].std())
        smean = float(prim[col].mean()) if len(prim) else None
        fpc = np.sqrt(max(0.0, (len(cand) - len(prim)) / max(1, len(cand) - 1)))
        se = full_sd / np.sqrt(max(1, len(prim))) * fpc
        rep[col] = {"full_split_mean": full_mean, "sample_mean": smean,
                    "z_sample_vs_full": (smean - full_mean) / se if smean is not None and se > 0 else None}
    meta = {
        "split": args.split, "ds_time_column": SPLIT_TIME[args.split], "seed": args.seed, "n_requested": args.n,
        "eve_file": eve, "eve_file_facts": facts,
        "n_eve_rows_in_split": F["n_rows_eve_split"],
        "n_candidates_authors_join": int(len(cand)),
        "candidates_per_year": {int(y): int(c) for y, c in cand["year"].value_counts().sort_index().items()},
        "candidate_eve_minus_sdo_seconds": {"min": float(cand["eve_minus_sdo_s"].min()),
                                            "median": float(cand["eve_minus_sdo_s"].median()),
                                            "max": float(cand["eve_minus_sdo_s"].max())},
        "authors_join": "EVEDSDataset over core-sdo valid_index_surya_1_0.csv (t and t+60 present), merge_asof forward "
                        "to the first EVE time >= t, duplicates keep the closest t, tolerance 6 min",
        "finetune_validation_note": "finetune.py validates on val_time with StatefulDistributedSampler(shuffle=True, "
                                    "seed 0, drop_last) and stops after iters_per_epoch_valid=200 batches of 1",
        "target_checks_full_split": {"denorm_vs_raw_max_rel_diff": F["denorm_vs_raw_max_rel_diff"],
                                     "dataset_vs_my_transform_max_abs_diff": F["transform_vs_mine_max_abs_diff"],
                                     "normalised_target_min": float(F["Y"].min()),
                                     "normalised_target_max": float(F["Y"].max()),
                                     f"fraction_elements_lt_{NEAR_ZERO}": float((F["Y"] < NEAR_ZERO).mean())},
        "allocation": alloc, "blocks": [{k: v for k, v in b.items() if k != "order"} for b in blocks],
        "n_primary": int((allrows["role"] == "primary").sum()), "n_reserve": int((allrows["role"] == "reserve").sum()),
        "n_anchor": int(len(anc)), "n_skipped": len(skipped), "anchor_skips": anchor_skips,
        "channels_checked": not args.no_channel_check,
        "dataset_join_all_ok": bool(all(j["ok"] for j in joins)), "dataset_join_checks": joins,
        "notebook_anchor_ground_truth_check": anchor_gt,
        "baseline_train_mean_on_all_candidates": base_all,
        "representativeness_primary_vs_full_split": rep,
        "config_differences_config_vs_config_infer": diff,
        "plan": {"forward_passes_this_split": n_pass, "anchors_in_plan": int(len(anc_extra)),
                 "frames_to_download_bytes": frames_bytes, "max_frame_bytes_per_sample": per_sample_bytes,
                 "peak_frames_on_disk_gb_with_prefetch": round(2 * per_sample_bytes / 1e9, 2),
                 "checkpoint_bytes": CKPT_SIZE, "checkpoint_present": ckpt_present,
                 "free_disk_gb_now": round(free, 2), "prefetch_possible_now": bool(prefetch_ok),
                 "peak_disk_gb_new": round(need_ckpt + (2 if prefetch_ok else 1) * per_sample_bytes / 1e9, 2),
                 "bandwidth_probe": bw, "est_minutes_compute_only": round(est_s / 60, 1),
                 "est_minutes_with_bandwidth": round((est_s_bw + (ckpt_s or 0)) / 60, 1),
                 "est_checkpoint_download_minutes": round(ckpt_s / 60, 1) if ckpt_s else None,
                 "max_forward_that_fits_90_min": fit90,
                 "timing_assumptions_s": {"forward": SEC_FORWARD, "load_frame": SEC_LOAD_FRAME, "score": SEC_SCORE,
                                          "model_build_and_load": SEC_MODEL}},
        "dry_run_seconds": round(time.time() - t_start, 1),
    }
    json.dump(meta, open(os.path.join(out_dir, f"sampling_{args.split}.json"), "w"), indent=1, default=float)
    log(f"{args.split}: allocation {alloc}; {meta['n_primary']} primaries, {meta['n_reserve']} reserves, "
        f"{len(anc)} anchors, {len(skipped)} candidates skipped; dataset joins all ok: {meta['dataset_join_all_ok']}")
    for a in anchor_gt:
        log(f"anchor {a['timestamp']} -> EVE {a['eve_time']}: target range {a['dataset_target_range_4dp']} vs notebook "
            f"{list(a['notebook_gt_range'])} equal at 4 dp: {a['equal_at_4dp']}")
    log(f"baseline (train-mean spectrum) on all {len(cand)} pairs: MSE {base_all['mse']:.3e} MAE {base_all['mae']:.4f} "
        f"MAPE {base_all['mape_pct']:.2f}%")
    log(f"plan: {n_pass} forward passes, {frames_bytes / 1e9:.1f} GB of frames in total, "
        f"{2 * per_sample_bytes / 1e9:.2f} GB on disk at once with prefetch; free disk {free:.1f} GB; "
        f"estimate {meta['plan']['est_minutes_with_bandwidth']} min (compute only "
        f"{meta['plan']['est_minutes_compute_only']} min); {fit90} passes fit in 90 min")
    if not meta["dataset_join_all_ok"]:
        raise RuntimeError("dataset join check failed; see sampling json")
    if run_mode:
        return allrows, meta
    summarize(out_dir, args.seed)
    return allrows, meta


# ---------------------------------------------------------------------------------------------------- run
def run(args, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from surya.utils.data import build_scalers
    sys.path.insert(0, TASK_DIR)
    from finetune import custom_collate_fn

    ensure_small_inputs()
    cfg, cfg_inf, _ = load_configs()
    cfg_inf["data"]["scalers"] = yaml.safe_load(open(cfg_inf["data"]["scalers_path"]))  # as infer.main
    cfg_inf["dtype"] = torch.float32  # as infer.main for dtype "float32"; HelioSpectFormer ignores it
    scalers = build_scalers(info=yaml.safe_load(open(cfg["data"]["scalers_path"])))
    metric_self_test(out_dir)
    sample, meta = get_sample(args, out_dir, cfg, scalers)
    if not meta.get("dataset_join_all_ok", False):
        raise RuntimeError("dataset join check failed in the sample file; rerun --dry-run")
    wl = full_split(cfg, scalers, args.split)["wl"]

    recs_path, anc_path = per_sample_path(out_dir, args.split), anchors_path(out_dir)
    previous, prev_anc = read_records(recs_path), read_records(anc_path)
    done = {t for t, r in previous.items() if r.get("status") == "ok"}
    done_anc = {t for t, r in prev_anc.items() if r.get("status") == "ok"}
    blocks, anchors = {}, []
    for r in sample.to_dict("records"):
        if r["role"] == "anchor":
            anchors.append(r)
        else:
            blocks.setdefault((int(r["stratum"]), int(r["block"])), {})[r["role"]] = r
    ok_blocks = {(int(r["stratum"]), int(r["block"])) for r in previous.values()
                 if r.get("status") == "ok" and r.get("role") in ("primary", "reserve")}
    work = []
    for key, b in sorted(blocks.items()):
        if key in ok_blocks:  # one sample per block: a block is finished once its primary or reserve is scored
            continue
        prim, res = b.get("primary"), b.get("reserve")
        failed_before = prim is not None and previous.get(prim["timestamp"], {}).get("status") == "failed"
        if prim is not None and not failed_before:
            work.append(prim)
        elif res is not None:
            work.append(res)
    if not args.no_anchors:  # notebook anchors first (an early end-to-end check), once overall
        work = [a for a in anchors if a["timestamp"] not in done_anc] + work
    pending = [r for r in work if not (r["role"] != "anchor" and r["timestamp"] in done)]
    # a time that is both an anchor and a validation sample is run once; the record is copied to the other file
    reuse = {}
    for r in pending:
        if r["role"] == "anchor" and r["timestamp"] in done and args.split == "validation":
            reuse[r["timestamp"]] = previous[r["timestamp"]]
        if r["role"] != "anchor" and r["timestamp"] in done_anc and args.split == "validation":
            reuse[r["timestamp"]] = prev_anc[r["timestamp"]]
    for t, src in reuse.items():
        for r in [p for p in pending if p["timestamp"] == t]:
            rec = dict(src, role=r["role"], stratum=r["stratum"], block=r["block"],
                       block_candidates=r["block_candidates"], copied_from_existing_record=True)
            with open(anc_path if r["role"] == "anchor" else recs_path, "a") as fp:
                fp.write(json.dumps(rec, default=float) + "\n")
            pending.remove(r)
    pending = pending[: args.max_forward]
    log(f"{len(blocks)} blocks, {len(ok_blocks)} already scored; {len(pending)} forward passes to run now")
    if not pending:
        s = summarize(out_dir, args.seed)
        report(s, args.split)
        return

    per_sample_gb = max((r["frame_bytes"] + r["frame_t60_bytes"]) for r in pending) / 1e9
    ckpt_present = os.path.exists(os.path.join(WORK, CKPT_NAME))
    free = free_gb() - (0 if ckpt_present else CKPT_SIZE / 1e9)
    prefetch = not args.no_prefetch and free - 2 * per_sample_gb >= DISK_MARGIN_GB and \
        2 * per_sample_gb <= FRAME_BUDGET_GB
    if free - per_sample_gb < DISK_MARGIN_GB * 0.5:
        raise IOError(f"not enough disk: {free:.1f} GB free after the checkpoint, one sample needs {per_sample_gb:.1f}")
    log(f"free disk after checkpoint {free:.1f} GB; prefetch {'on' if prefetch else 'off'}")

    ckpt = ensure_checkpoint()
    ensure_pretrained_link(cfg_inf)
    t0 = time.time()
    model, minfo = load_model(cfg_inf, ckpt, args.allow_full_unpickle)
    minfo["seconds_build_and_load"] = round(time.time() - t0, 1)
    env = {"torch": torch.__version__, "threads": torch.get_num_threads(), "precision": "float32, no autocast",
           "upstream_head": os.popen(f"git -C {UPSTREAM} rev-parse --short HEAD").read().strip(),
           "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}", "checkpoint_sha256": CKPT_SHA256,
           "labels": f"{EVE_REPO}@{EVE_REV}/{EVE_FILE}", "model": minfo, "model_training_flag": model.training,
           "prefetch": prefetch, "peak_rss_gb_after_model_load": C.peak_rss_gb()}
    json.dump(env, open(os.path.join(out_dir, "environment.json"), "w"), indent=1, default=str)
    log(f"model ready in {minfo['seconds_build_and_load']} s, peak RSS {C.peak_rss_gb()} GB")

    caches = [C.FrameCache(os.path.join(WORK, "frames", "a")), C.FrameCache(os.path.join(WORK, "frames", "b"))]
    pool = cf.ThreadPoolExecutor(1)

    def frames_of(row):
        t = pd.Timestamp(row["timestamp"])
        return [t, t + pd.Timedelta(minutes=cfg["data"]["time_delta_target_minutes"])]

    fut = pool.submit(caches[0].ensure, frames_of(pending[0]))
    j = 0
    while j < len(pending):
        row = pending[j]
        split = row["split"]
        t = pd.Timestamp(row["timestamp"])
        cache = caches[j % 2] if prefetch else caches[0]
        rec = {k: row.get(k) for k in ("split", "role", "stratum", "block", "block_candidates", "timestamp",
                                       "eve_time", "eve_minus_sdo_s", "frame_key", "frame_bytes", "frame_t60_key",
                                       "frame_t60_bytes")}
        t_start = time.time()
        ds = loader = batch = x = out = None
        try:
            dl_s, err = fut.result(), None
        except Exception as e:
            dl_s, err = None, e
        if prefetch and j + 1 < len(pending):  # prefetch the next sample's two frames into the other cache
            fut = pool.submit(caches[(j + 1) % 2].ensure, frames_of(pending[j + 1]))
        try:
            if err is not None:
                raise err
            t1 = time.time()
            verify_frames([os.path.join(cache.root, C.s3_key(ts)) for ts in frames_of(row)], cfg["data"]["channels"])
            F = full_split(cfg, scalers, split)
            i_full = int(np.flatnonzero(F["cand"]["t"].values == np.datetime64(t))[0])
            ds = one_sample_dataset(cfg, scalers, split, t, cache.root)
            assert len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == t, "dataset did not select the time stamp"
            eve_t = pd.Timestamp(ds.df_valid_indices.iloc[0]["ds_index"])
            assert eve_t == pd.Timestamp(row["eve_time"]), f"dataset EVE time {eve_t} != {row['eve_time']}"
            loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
            batch, metadata = next(iter(loader))
            t2 = time.time()
            fc = batch.pop("forecast")  # the t+60 frame; HelioSpectformer1D never reads it
            rec["forecast_frame"] = {"shape": list(fc.shape), "nonfinite": int((~torch.isfinite(fc)).sum())}
            del fc
            x = batch["ts"]
            rec["input"] = {"shape": list(x.shape), "dtype": str(x.dtype), "nonfinite": int((~torch.isfinite(x)).sum()),
                            "time_delta_input": batch["time_delta_input"].tolist(), "per_channel_normalised": {}}
            for ci, ch in enumerate(cfg["data"]["channels"]):  # mean / std / non-finite count after the scalers
                v = x[0, ci, 0].double()
                m = float(torch.nanmean(v))
                rec["input"]["per_channel_normalised"][ch] = [
                    round(m, 4), round(float(torch.nanmean((v - m) ** 2)) ** 0.5, 4), int((~torch.isfinite(v)).sum())]
                del v
            y = batch["target"][0].numpy()
            lab = batch["label"][0].numpy()
            rec["target_checks"] = {
                "dtype": str(batch["target"].dtype),
                "label_equals_target": bool(np.array_equal(y, lab)),
                "bit_identical_to_full_split_dataset": bool(np.array_equal(y, F["Y"][i_full])),
                "denorm_vs_raw_eve_max_rel_diff": float(np.max(np.abs(denorm(y) - F["raw"][i_full]) / F["raw"][i_full]))}
            rec["target_checks"]["ok"] = bool(rec["target_checks"]["label_equals_target"]
                                              and rec["target_checks"]["bit_identical_to_full_split_dataset"]
                                              and rec["target_checks"]["denorm_vs_raw_eve_max_rel_diff"] < 1e-5)
            with torch.no_grad():
                out = model(batch)
            t3 = time.time()
            pred = out.reshape(-1).numpy().astype(np.float32)
            rec["output"] = {"shape": list(out.shape), "dtype": str(out.dtype),
                             "nonfinite": int((~np.isfinite(pred)).sum())}
            rec["pred"] = [float(v) for v in pred]
            rec["target"] = [float(v) for v in y.astype(np.float32)]
            rec["baseline_pred"] = [float(v) for v in F["baseline"]]
            st = sample_stats(pred, y, wl)
            rec["sample_metrics"] = {k: st[k] for k in ("mse", "mae", "mape_pct", "median_ape_pct", "corr",
                                                         "max_ape_pct", "max_ape_wavelength_nm", "max_ape_target",
                                                         "max_ape_pred", "pred_min", "pred_max", "target_min",
                                                         "target_max", "target_mean")}
            try:
                rec["metadata"] = {k: [str(v) for v in np.ravel(metadata[k])] for k in metadata}
            except Exception:
                rec["metadata"] = str(metadata)[:500]
            rec["seconds"] = {"frame_download": dl_s, "dataset_and_load": round(t2 - t1, 1),
                              "forward": round(t3 - t2, 1), "score": round(time.time() - t3, 1),
                              "total": round(time.time() - t_start, 1)}
            rec["status"] = "ok"
            log(f"[{j + 1}/{len(pending)}] {row['role']} {row['timestamp']} (EVE {row['eve_time']}): MSE "
                f"{st['mse']:.3e} MAE {st['mae']:.4f} MAPE {st['mape_pct']:.2f}% r {st['corr']:.4f}, "
                f"forward {rec['seconds']['forward']} s")
        except Exception as e:  # record, substitute the block's reserve once, carry on
            rec["status"] = "failed"
            rec["error"] = repr(e)[:2000]
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} FAILED: {e!r}")
            res = blocks.get((int(row["stratum"]), int(row["block"])), {}).get("reserve") \
                if row["role"] == "primary" else None
            if res is not None and res["timestamp"] not in done and all(p["timestamp"] != res["timestamp"]
                                                                        for p in pending):
                pending.append(res)
                if prefetch and j + 1 == len(pending) - 1:  # nothing was prefetched for it yet
                    fut = pool.submit(caches[(j + 1) % 2].ensure, frames_of(res))
        finally:
            ds = loader = batch = x = out = None  # release the input tensors before the next sample
        rec["peak_rss_gb"] = C.peak_rss_gb()
        with open(anc_path if row["role"] == "anchor" else recs_path, "a") as fp:
            fp.write(json.dumps(rec, default=float) + "\n")
        if row["role"] == "anchor" and rec["status"] == "ok" and args.split == "validation":
            same = [p for p in sample.to_dict("records") if p["timestamp"] == row["timestamp"] and p["role"] != "anchor"]
            for p in same:  # also a stratified sample of this split: record it there without a second pass
                with open(recs_path, "a") as fp:
                    fp.write(json.dumps(dict(rec, role=p["role"], stratum=p["stratum"], block=p["block"],
                                             block_candidates=p["block_candidates"],
                                             copied_from_existing_record=True), default=float) + "\n")
                pending = [q for q in pending if not (q["timestamp"] == p["timestamp"] and q["role"] != "anchor")]
        gc.collect()
        j += 1
        if not prefetch and j < len(pending):
            cache.clear()
            fut = pool.submit(caches[0].ensure, frames_of(pending[j]))
    pool.shutdown(wait=True)
    for c in caches:
        c.clear()
    if args.delete_checkpoint:
        os.remove(ckpt)
    s = summarize(out_dir, args.seed)
    report(s, args.split)


def report(s, split):
    if "notebook_anchor_check" in s:
        a = s["notebook_anchor_check"]
        log(f"notebook anchors: {a['n_run']} run, all within tolerance: {a.get('all_within_tolerance')}, "
            f"summary diffs {a.get('summary_abs_diff')}")
    sp = s["splits"].get(split, {})
    if "metrics" in sp:
        m = sp["metrics"]["model"]
        ci = m.get("ci95", {})
        log(f"{split}: n={sp['n_samples_scored']} MSE {m['mse']:.3e} {ci.get('mse')} MAE {m['mae']:.5f} {ci.get('mae')} "
            f"MAPE {m['mape_pct']:.3f}% {ci.get('mape_pct')} (paper 1.2607e-04 / 0.0045115 / 1.4793)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="select and check the sample; no weights, frames or model")
    ap.add_argument("--split", choices=["test", "validation"], default="test",
                    help="test = test_time (Jan 15-28 of 2011, 2014); validation = val_time (Jan 15-28 of 2012, 2013)")
    ap.add_argument("--n", type=int, default=None, help="samples (default 30 test, 20 validation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "Surya", "euv_spectra"))
    ap.add_argument("--max-forward", type=int, default=40, help="forward passes allowed in this run")
    ap.add_argument("--no-channel-check", action="store_true", help="skip the netCDF header check of each frame")
    ap.add_argument("--header-workers", type=int, default=4, help="threads reading netCDF headers during selection")
    ap.add_argument("--no-anchors", action="store_true", help="leave out the two notebook anchor samples")
    ap.add_argument("--no-prefetch", action="store_true", help="keep only one sample's frames on disk (~1.2 GB)")
    ap.add_argument("--delete-checkpoint", action="store_true", help="remove the 1.8 GB checkpoint at the end")
    ap.add_argument("--allow-full-unpickle", action="store_true",
                    help="if weights_only loading fails, load the checkpoint with weights_only=False as infer.py does")
    ap.add_argument("--summarize-only", action="store_true", help="recompute summary.json from the jsonl files")
    args = ap.parse_args()
    args.n = args.n or DEFAULT_N[args.split]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(WORK, "tmp"), exist_ok=True)
    os.chdir(WORK)  # config paths are ./assets/...; HelioNetCDFDataset writes logs/data/ relative to the cwd
    if args.summarize_only:
        report(summarize(out_dir, args.seed), args.split)
        return
    if args.dry_run:
        dry_run(args, out_dir)
    else:
        run(args, out_dir)


if __name__ == "__main__":
    main()
