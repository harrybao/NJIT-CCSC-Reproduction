"""Solar wind speed (V at L1, 4 days ahead) with the released fine-tuned Surya model, on a seeded sample of SuryaBench.

PARTIAL REPRODUCTION. Table 6 of arXiv 2508.14112 v2 (Sec. 2.6.5) reports for Surya RMSE 75.92 km/s, MAE 58.06 km/s
and "Validation MSE" 5698.62 km^2/s^2 (AlexNet 118.6 / 95.7 / 13839.49, ResNet50 93.76 / 74.65 / 8547.924). The paper
does not say which split, how many samples, or how the numbers were aggregated; sqrt(5698.62) = 75.49, not 75.92, so
the RMSE and the MSE columns were not computed on the same set. This script scores the released checkpoint
nasa-ibm-ai4science/solar_wind_surya on a small, seeded, year-stratified sample of the SuryaBench solar-wind TEST
split (inputs in 2020-2024) or VALIDATION split (Jan 15-31 of 2011-2019, the set finetune.py validates on), on a CPU,
and reports RMSE, MAE and MSE in km/s with bootstrap 95% intervals over samples. It can say whether the paper's numbers
are plausible for these samples; it cannot say what the full split would give.

What is the authors' code (imported unchanged from the clone, nothing copied or edited):
  * WindSpeedDSDataset (downstream_examples/solar_wind_forcasting/dataset.py) with the arguments that
    finetune.get_dataloaders gives its validation set (config.yaml): one input frame at t (time_delta_input_minutes [0]);
    t is only accepted when t+60 min is also indexed (time_delta_target_minutes +60, rollout_steps 0) and, unlike the
    AR dataset, the t+60 frame IS read (into batch["forecast"], which the 1-D model never uses), so both frames are
    downloaded; the label is V at the first CSV time >= t + 4 days (pd.merge_asof forward, tolerance 1 h,
    ds_time_delta_in_out "4D"), min-max scaled with the min and range of the CSV the dataset is given.
  * custom_collate_fn, get_model and apply_peft_lora (finetune.py, as imported by infer.py) with config_infer.yaml
    (infer.py's default config), then the checkpoint loading of infer.load_model with load_state_dict(strict=True).
  * The metric code it is checked against: finetune.evaluate_model (MAE, RMSE, R^2, mean MSELoss) and the summary
    block of infer.infer_single_sample (MAE, RMSE, R^2 in km/s), both called on synthetic inputs (metric_check.json).

De-normalisation (the important repo quirk). WindSpeedDSDataset.un_norm_output(x) = x * ptp(V) + min(V) of whatever
CSV that dataset object was built on. finetune.py trains on train.csv (min 240, range 560), so the network learned to
output (V - 240) / 560; but infer.py rescales with the dataset it builds, which config_infer.yaml points at
validation.csv (min 259, range 494), and finetune.evaluate_model compares the output with validation targets scaled by
validation.csv's own constants. Primary predictions here use the TRAINING constants (what the model was trained to
emit). Predictions with the validation.csv constants (infer.py as shipped) and with the evaluated split's own constants
(infer.py pointed at that CSV; test.csv: min 256, range 750) are recorded and scored as well. The targets are always
the observed V in km/s (the dataset's normalised target is un-normalised with its own constants, checked to equal the
CSV value). ds_scaler [2.61, 0.09] / logsign is dead code in dataset.py (the call is commented out) and plays no role.

Differences from infer.py (none changes the network or its weights): torch.load uses weights_only=True and mmap=True
(the pickle of solar_wind_weights.pth holds only an OrderedDict of FloatStorage tensors, checked from its opcodes); the
forward pass runs in float32 without autocast (infer.py's autocast(cpu, float32) disables itself; CPU bfloat16 breaks
Surya, results/Surya/bf16_diagnosis); DataLoader with num_workers=0 and a one-sample dataset per time stamp;
batch["forecast"] (the unused t+60 frame) is dropped before the forward pass to save 0.87 GB of RAM. finetune.py's
evaluate_model permutes batch["ts"] to (B, T, C, H, W) and infer.py does not; with one input time the patch embedding's
flatten(1, 2) gives identical tensors either way (checked on every sample).

Pipeline anchors (validation split only, --no-anchors to drop them). The saved output of solar_wind_tutorial.ipynb
is a float32 CPU run of infer.py on three validation-period time stamps (2011-01-20 01:00 and 02:00, 2019-01-23
02:00), with predictions printed in km/s (validation.csv constants). They are run first, scored against those printed
numbers with a tolerance fixed in advance (0.5 km/s), and kept out of every sample metric.

Baselines (MY OWN, not in Table 6): 4-day persistence V(t) (the speed at the input time), 27-day recurrence V(T - 27 d)
(Table 7 quotes a "27-day Persistence" of 89 km/s from Ahn et al. 2025 on another period) and the training-set mean.
They are scored on the same samples and on every candidate pair of the split (CSV only, no model).

Sampling. Candidates are the target rows T of <split>.csv with input time t = T - 4 d such that, for the test split,
t >= 2020-01-01 (the input lies in the test period too), and, for the validation split, t and t+60 are present in
core-sdo's valid_index_surya_1_0.csv (the SDO index finetune.py uses for validation). Only exact matches T = t + 4 d
are used (in the full dataset the merge_asof/de-duplication would also accept a later T within 1 h when the exact hour
is missing; such pairs are left out). The sample is split evenly over years of t (remainder to years drawn with the
seed); inside a year the candidates are cut into equal consecutive time blocks, one sample per block, drawn in a
seeded random order. A candidate is skipped (and the next one in the block tried) if its frame at t or t+60 is not on
S3 or lacks one of the 13 channels (netCDF header read with HTTP range requests). Every skip is recorded with its
reason. Each block also gets a reserve, used only if the primary fails.

Usage (from the repository root):
  python scripts/Surya/downstream/solar_wind_subset_eval.py --dry-run --split test --n 30 --seed 0
  python scripts/Surya/downstream/solar_wind_subset_eval.py --split test --n 30 --seed 0
  python scripts/Surya/downstream/solar_wind_subset_eval.py --dry-run --split validation --n 10 --seed 0
  python scripts/Surya/downstream/solar_wind_subset_eval.py --split validation --n 10 --seed 0
  python scripts/Surya/downstream/solar_wind_subset_eval.py --summarize-only
A rerun skips time stamps already in per_sample_<split>.jsonl, so an interrupted run resumes; --max-forward K limits
one invocation to K forward passes (the dry run prints the K that fits in about 90 minutes).
"""
import argparse
import concurrent.futures as cf
import contextlib
import functools
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import re
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import surya_common as C  # noqa: E402  (sets HF_HUB_DISABLE_XET=1 before huggingface_hub is imported)
# our own helpers from the sibling AR script (HTTP range reader for netCDF headers, small-file fetch, sha256, HEAD retry)
from ar_segmentation_subset_eval import fetch_small, remote_nc_channels, s3_exists_retry, sha256_file  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
UPSTREAM = "/home/user/upstream/surya/Surya"
TASK_DIR = os.path.join(UPSTREAM, "downstream_examples", "solar_wind_forcasting")
FM_DIR = "/home/user/work/surya_run/data/Surya-1.0"
WORK = "/home/user/work/surya_ft/solar_wind"
# config.yaml's solarwind_*_index, scalers_path and pretrained_path are relative ("./assets/..."), and
# HelioNetCDFDataset writes logs/data/ relative to the cwd, so the script runs with WORK as the working directory and
# lays out WORK/assets/ the way download_data.sh would.
ASSETS = os.path.join(WORK, "assets")
LABEL_DIR = os.path.join(ASSETS, "Surya-bench-solarwind")

HF = "https://huggingface.co"
LABEL_REPO, LABEL_REV = "nasa-ibm-ai4science/Surya-bench-solarwind", "20547a25cbd3aaa359e9cda18be913b94c225e7b"
LABEL_BLOBS = {  # git blob ids from the HF tree API at LABEL_REV (sha1 of "blob <size>\0<content>")
    "train.csv": "0bd3047d8a57b6e81dd37c3576da08bf9a18baae",
    "validation.csv": "472f3cc74a72c64cc14eba5c30b4ced6d45c657d",
    "test.csv": "4b894121b91c27b5facd890507f265bb03304a81",
    "leaky_validation.csv": "c622eec6b79bb8d81a75f1cb46e15072b6a8f19e",
}
CKPT_REPO, CKPT_REV = "nasa-ibm-ai4science/solar_wind_surya", "6ac4aa79a96f8785a1ab300842d13727e52b58a5"
CKPT_NAME, CKPT_SIZE = "solar_wind_weights.pth", 1805655558
CKPT_SHA256 = "7ff8c19024fc8a8c4cb5c73278081ba242ba7e871b832e1caab11be8e3166a17"  # HF LFS oid
CORE_SDO_REPO, CORE_SDO_REV = "nasa-ibm-ai4science/core-sdo", "9c10abe52c1f206d1b66715745229e29ef543cc6"

PAPER_TABLE6 = {  # arXiv 2508.14112 v2, Sec. 2.6.5, Table 6 (copied exactly; "95.7 km" has a stray unit in the source)
    "AlexNet": {"rmse_km_s": 118.6, "mae_km_s": 95.7, "validation_mse_km2_s2": 13839.49},
    "ResNet50": {"rmse_km_s": 93.76, "mae_km_s": 74.65, "validation_mse_km2_s2": 8547.924},
    "Surya": {"rmse_km_s": 75.92, "mae_km_s": 58.06, "validation_mse_km2_s2": 5698.62},
}
PAPER_TABLE7_27DAY_PERSISTENCE = 89  # "27-day Persistence Ahn et al. (2025)", RMSE km/s, other period
# Pipeline anchors: the saved output of solar_wind_tutorial.ipynb (cell 9: infer.run_inference with config_infer.yaml,
# "CUDA available: False", torch 2.8.0+cu126, "CPU Autocast ... Disabling autocast", i.e. a float32 CPU run of the same
# checkpoint). infer.py un-normalises with validation.csv's constants (min 259, range 494). (input time, printed
# prediction km/s, printed ground truth km/s); all three input times lie in the validation period (DOY 20 and 23).
# They are scored separately and never enter the sample metrics.
NOTEBOOK_ANCHORS = [("2011-01-20 01:00:00", 393.63, 344.00), ("2019-01-23 02:00:00", 391.95, 433.00),
                    ("2011-01-20 02:00:00", 383.07, 351.00)]
ANCHOR_TOL_KM_S = 0.5  # fixed before any run: 0.005 km/s print rounding plus float32 differences across CPUs/torch
SPLIT_CSV = {"test": "test.csv", "validation": "validation.csv"}
STRATA_YEARS = {"test": list(range(2020, 2025)), "validation": list(range(2011, 2020))}
DEFAULT_N = {"test": 30, "validation": 10}
LEAD = pd.Timedelta(days=4)  # config.yaml ds_time_delta_in_out "4D"
LAG27 = pd.Timedelta(days=27)
N_BOOT = 10000
# Timing assumptions for the estimate (the dry run cannot time the model): one fp32 forward pass of the 4096x4096
# Surya backbone took 136 s on this machine (results/Surya/validation_test, 271.9 s for two passes); reading and
# normalising one 13-channel frame is taken as 30 s (two frames per sample here); scoring 2 s; building the model,
# loading the foundation weights (get_model needs them) and then the fine-tuned checkpoint 150 s.
SEC_FORWARD, SEC_LOAD_FRAME, SEC_SCORE, SEC_MODEL = 136, 30, 2, 150


# ----------------------------------------------------------------------------------------------- small utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hf_url(repo, rev, path, kind="datasets"):
    return f"{HF}/datasets/{repo}/resolve/{rev}/{path}" if kind == "datasets" else f"{HF}/{repo}/resolve/{rev}/{path}"


def git_blob_sha1(path):
    data = open(path, "rb").read()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def stamp(t):
    return pd.Timestamp(t).strftime("%Y%m%d_%H%M")


def jnum(x):
    """float for JSON, None for NaN."""
    return None if x is None or not np.isfinite(x) else float(x)


# --------------------------------------------------------------------------------------------------- metrics
def errors_metrics(pred, obs):
    """RMSE, MAE, MSE, bias (mean pred - obs), R^2 = 1 - SSE/SST (infer.py / finetune.evaluate_model), Pearson r."""
    pred, obs = np.asarray(pred, dtype=np.float64), np.asarray(obs, dtype=np.float64)
    e = pred - obs
    mse = float(np.mean(e ** 2))
    sst = float(np.sum((obs - obs.mean()) ** 2))
    return {"rmse": mse ** 0.5, "mae": float(np.mean(np.abs(e))), "mse": mse, "bias": float(np.mean(e)),
            "r2": 1.0 - float(np.sum(e ** 2)) / sst if sst > 0 else float("nan"),
            "pearson_r": float(np.corrcoef(pred, obs)[0, 1]) if len(e) > 2 and np.std(pred) > 0 else float("nan"),
            "n": int(len(e))}


def bootstrap_ci(stat, n, n_boot=N_BOOT, seed=0):
    """Percentile 95% interval of stat(idx) over resamples of n samples; stat takes a (B, n) index array."""
    rng = np.random.default_rng(seed)
    vals = stat(rng.integers(0, n, size=(n_boot, n)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def scored(pred, obs, n_boot=N_BOOT, seed=0):
    """Point metrics plus bootstrap intervals for RMSE, MAE, MSE and bias."""
    pred, obs = np.asarray(pred, dtype=np.float64), np.asarray(obs, dtype=np.float64)
    e = pred - obs
    m = errors_metrics(pred, obs)
    n = len(e)
    if n >= 2:
        m["ci95"] = {"rmse": bootstrap_ci(lambda i: np.sqrt(np.mean(e[i] ** 2, 1)), n, n_boot, seed),
                     "mae": bootstrap_ci(lambda i: np.mean(np.abs(e[i]), 1), n, n_boot, seed),
                     "mse": bootstrap_ci(lambda i: np.mean(e[i] ** 2, 1), n, n_boot, seed),
                     "bias": bootstrap_ci(lambda i: np.mean(e[i], 1), n, n_boot, seed)}
    return m


def paired_rmse_diff(pred_a, pred_b, obs, n_boot=N_BOOT, seed=0):
    """RMSE(a) - RMSE(b) on the same samples, with a paired bootstrap 95% interval."""
    ea, eb = np.asarray(pred_a, float) - np.asarray(obs, float), np.asarray(pred_b, float) - np.asarray(obs, float)
    n = len(ea)
    est = float(np.sqrt(np.mean(ea ** 2)) - np.sqrt(np.mean(eb ** 2)))
    ci = bootstrap_ci(lambda i: np.sqrt(np.mean(ea[i] ** 2, 1)) - np.sqrt(np.mean(eb[i] ** 2, 1)), n, n_boot, seed)
    return {"estimate": est, "ci95": ci, "n": int(n)}


class _FakeModel:
    """Stands in for the network in the metric self-test: returns preset outputs, one batch after another."""

    def __init__(self, outputs):
        import torch
        self.out = [torch.tensor([[float(v)]], dtype=torch.float32) for v in outputs]
        self.i = 0

    def eval(self):
        return self

    def __call__(self, batch):
        o = self.out[self.i]
        self.i += 1
        return o


class _Loader(list):
    """A list of (batch, metadata) that also has the .dataset attribute infer.infer_single_sample prints."""

    @property
    def dataset(self):
        return self


def metric_self_test(out_dir):
    """Checks errors_metrics against finetune.evaluate_model and infer.infer_single_sample on synthetic inputs."""
    import torch
    import torch.distributed as dist
    sys.path.insert(0, TASK_DIR)
    import finetune as FT  # authors' evaluate_model
    import infer as I  # authors' infer_single_sample
    from dataset import WindSpeedDSDataset

    rng = np.random.default_rng(0)
    n = 37
    obs_n = rng.uniform(0.05, 0.9, n)  # targets in normalised units, as in the dataset
    pred_n = (obs_n + rng.normal(0, 0.12, n)).astype(np.float32).astype(np.float64)  # the model emits float32
    report = {"n_synthetic": n}

    # (a) finetune.evaluate_model in a one-process gloo group (it calls dist.all_reduce); targets float64 as the
    #     dataset returns them, outputs float32 as the model returns them; MSELoss as in finetune.main
    def batches():
        return [({"ts": torch.zeros(1, 1, 1, 1, 1), "target": torch.tensor([[o]], dtype=torch.float64),
                  "ds_time": torch.tensor([0.0], dtype=torch.float64)}, {}) for o in obs_n]

    tmp = tempfile.NamedTemporaryFile(prefix="gloo_", dir=WORK, delete=False).name
    os.remove(tmp)
    dist.init_process_group("gloo", init_method=f"file://{tmp}", rank=0, world_size=1)
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            mae, rmse, r2, avg_loss = FT.evaluate_model(batches(), 0, _FakeModel(pred_n), "cpu", None,
                                                        {"iters_per_epoch_valid": 200, "dtype": torch.float32},
                                                        torch.nn.MSELoss())
    finally:
        dist.destroy_process_group()
        if os.path.exists(tmp):
            os.remove(tmp)
    mine = errors_metrics(pred_n, obs_n)
    report["finetune_evaluate_model_normalised_units"] = {
        "repo": {"mae": mae, "rmse": rmse, "r2": r2, "avg_loss_MSELoss": avg_loss},
        "mine": {"mae": mine["mae"], "rmse": mine["rmse"], "r2": mine["r2"], "mse": mine["mse"]}}
    rel = max(abs(mae - mine["mae"]) / mine["mae"], abs(rmse - mine["rmse"]) / mine["rmse"],
              abs(r2 - mine["r2"]) / abs(mine["r2"]), abs(avg_loss - mine["mse"]) / mine["mse"])
    report["finetune_evaluate_model_max_rel_diff"] = rel  # its accumulators are float32 tensors

    # (b) infer.infer_single_sample: un-normalises with dataset.un_norm_output and prints MAE/RMSE/R^2 (4 decimals);
    #     here the dataset object carries the train.csv constants (min 240, range 560)
    ns = type("NormOnly", (), {})()
    ns.un_norm_min, ns.un_norm_ptp = 240.0, 560.0
    ns.un_norm_output = functools.partial(WindSpeedDSDataset.un_norm_output, ns)
    md = {"timestamps_input": np.array([[np.datetime64("2020-01-01T00:00")]]),
          "timestamps_targets": np.array([[np.datetime64("2020-01-01T01:00")]])}
    loader = _Loader([(b, md) for b, _ in batches()])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        I.infer_single_sample(model=_FakeModel(pred_n), local_rank=torch.device("cpu"), validation_loader=loader,
                              gpu=True, dtype=torch.float32, save_path=os.path.join(WORK, "unused.png"),
                              device_type="cpu", dataset=ns)
    text = buf.getvalue()

    def grab(label):
        m = re.search(re.escape(label) + r"\s*\|\s*([-0-9.naN]+)", text)
        return float(m.group(1)) if m else float("nan")

    printed = {"mae": grab("Mean Absolute Error"), "rmse": grab("Root Mean Square Error"), "r2": grab("R² Score")}
    pred_k, obs_k = pred_n * 560.0 + 240.0, obs_n * 560.0 + 240.0
    mine_k = errors_metrics(pred_k, obs_k)
    report["infer_single_sample_km_s"] = {"repo_printed_4dp": printed,
                                          "mine": {k: mine_k[k] for k in ("mae", "rmse", "r2")}}
    absd = max(abs(printed[k] - mine_k[k]) for k in ("mae", "rmse", "r2"))
    report["infer_single_sample_max_abs_diff"] = absd  # bounded by the 4-decimal printout
    unnorm_diff = float(np.max(np.abs(np.array([ns.un_norm_output(v) for v in pred_n]) - pred_k)))
    report["un_norm_output_vs_mine_max_abs_diff"] = unnorm_diff

    # (c) hand example: errors +3 and -4 km/s -> MAE 3.5, MSE 12.5, RMSE 3.5355339, bias -0.5
    h = errors_metrics([403.0, 396.0], [400.0, 400.0])
    expect = {"mae": 3.5, "mse": 12.5, "rmse": 12.5 ** 0.5, "bias": -0.5}
    report["hand_example"] = {k: {"got": h[k], "expected": v} for k, v in expect.items()}
    hand = max(abs(h[k] - v) for k, v in expect.items())

    # (d) bootstrap sanity: interval brackets the estimate; paired difference of a forecast with itself is 0
    s = scored(pred_k, obs_k, n_boot=2000)
    pd0 = paired_rmse_diff(pred_k, pred_k, obs_k, n_boot=500)
    boot_ok = all(s["ci95"][k][0] <= s[k] <= s["ci95"][k][1] for k in ("rmse", "mae", "mse")) and \
        pd0["estimate"] == 0 and pd0["ci95"] == [0.0, 0.0]
    report["bootstrap_sanity_ok"] = bool(boot_ok)

    # (e) my regression (control-variate) estimator: exact when the target is linear in the control variates, and
    #     equal to the plain mean when the sample means of the control variates equal their full-split means
    Xe = rng.gamma(2.0, 1000.0, (30, 2))
    ra1 = regression_adjusted(3.0 + 2.0 * Xe[:, 0] - 0.5 * Xe[:, 1], Xe, np.array([1500.0, 2500.0]), n_boot=200)
    se0 = rng.gamma(2.0, 3000.0, 30)
    ra2 = regression_adjusted(se0, Xe, Xe.mean(0), n_boot=200)
    reg = {"linear_target_rel_err": abs(ra1["mse"] - (3.0 + 2.0 * 1500.0 - 0.5 * 2500.0)) / 1750.0,
           "no_shift_equals_mean_rel_err": abs(ra2["mse"] - se0.mean()) / se0.mean()}
    report["regression_estimator_checks"] = reg
    report["passed"] = bool(rel < 1e-5 and absd < 6e-5 and unnorm_diff < 1e-9 and hand < 1e-12 and boot_ok
                            and max(reg.values()) < 1e-9)
    json.dump(report, open(os.path.join(out_dir, "metric_check.json"), "w"), indent=1)
    if not report["passed"]:
        raise AssertionError(f"metric self-test failed: {json.dumps(report)[:2000]}")
    return report


# -------------------------------------------------------------------------------------------- data and labels
def ensure_small_inputs():
    """Label CSVs (pinned revision, git-blob checked), core-sdo validation index, scalers.yaml where config expects."""
    os.makedirs(LABEL_DIR, exist_ok=True)
    out = {}
    for name, blob in LABEL_BLOBS.items():
        p = fetch_small(hf_url(LABEL_REPO, LABEL_REV, name), os.path.join(LABEL_DIR, name))
        got = git_blob_sha1(p)
        if got != blob:
            raise IOError(f"{p}: git blob {got}, expected {blob}")
        out[name] = {"bytes": os.path.getsize(p), "git_blob_sha1": got}
    sc_dst = os.path.join(ASSETS, "scalers.yaml")
    if not os.path.exists(sc_dst):
        # core-sdo's scalers.yaml (config.yaml scalers_path) and Surya-1.0's are byte-identical (sha256 a944e8f1...)
        with open(os.path.join(FM_DIR, "scalers.yaml"), "rb") as a, open(sc_dst, "wb") as b:
            b.write(a.read())
    fetch_small(hf_url(CORE_SDO_REPO, CORE_SDO_REV, "valid_index_surya_1_0.csv"),
                os.path.join(ASSETS, "valid_index_surya_1_0.csv"))
    return out


@functools.lru_cache(maxsize=1)
def labels():
    """All four SuryaBench solar-wind CSVs: per split frame and one V series over every time stamp."""
    frames = {}
    for name in LABEL_BLOBS:
        d = pd.read_csv(os.path.join(LABEL_DIR, name))
        d["timestamp"] = pd.to_datetime(d["timestamp"])
        frames[name[:-4]] = d
    allv = pd.concat([d.set_index("timestamp")["V"] for d in frames.values()])
    if allv.index.duplicated().any():
        dup = allv[allv.index.duplicated(keep=False)]
        if dup.groupby(level=0).nunique().max() > 1:
            raise ValueError("a time stamp has different V in two split files")
        allv = allv[~allv.index.duplicated()]
    return frames, allv.sort_index()


def norm_constants():
    """min and ptp of V per CSV, computed as WindSpeedDSDataset does (np.min / np.ptp of the V column)."""
    frames, _ = labels()
    return {k: {"min": float(np.min(d["V"])), "ptp": float(np.ptp(d["V"])), "max": float(np.max(d["V"])),
                "mean": float(np.mean(d["V"])), "rows": int(len(d))} for k, d in frames.items()}


def candidates(split):
    """All (t, T = t + 4 d) pairs the authors' dataset would form for this split, with my baselines' inputs."""
    frames, allv = labels()
    lab = frames[split[:-4] if split.endswith(".csv") else split]
    T = lab["timestamp"]
    df = pd.DataFrame({"T": T.values, "V_T": lab["V"].values})
    df["t"] = df["T"] - LEAD
    info = {"rows_in_csv": int(len(lab))}
    if split == "test":
        keep = df["t"] >= pd.Timestamp("2020-01-01")
        info["rows_with_t_in_test_period"] = int(keep.sum())
    else:
        idx = pd.read_csv(os.path.join(ASSETS, "valid_index_surya_1_0.csv"))
        have = set(pd.to_datetime(idx.loc[idx["present"] == 1, "timestep"]))
        keep = df["t"].map(lambda t: t in have and t + pd.Timedelta(minutes=60) in have)
        info["rows_with_t_and_t60_in_core_sdo_valid_index"] = int(keep.sum())
    df = df[keep & (df["t"].dt.minute == 0)].copy()
    df["V_t"] = allv.reindex(df["t"]).values  # 4-day persistence input
    df["V_T27"] = allv.reindex(df["T"] - LAG27).values  # 27-day recurrence input
    df = df.sort_values("t").reset_index(drop=True)
    info["candidates"] = int(len(df))
    info["candidates_per_year"] = {int(k): int(v) for k, v in df.groupby(df["t"].dt.year).size().items()}
    info["candidates_with_V_t"] = int(df["V_t"].notna().sum())
    info["candidates_with_V_T27"] = int(df["V_T27"].notna().sum())
    return df, info


def design_blocks(cands, split, n, rng):
    """Year allocation (n split evenly, the remainder to years drawn with rng) and the consecutive equal time blocks
    inside each year, as arrays of positions in cands (sorted by t, default index)."""
    yr = cands["t"].dt.year.to_numpy()
    years = [y for y in STRATA_YEARS[split] if (yr == y).any()]
    base, extra = divmod(n, len(years))
    alloc = {y: base for y in years}
    for y in sorted(int(v) for v in rng.choice(years, size=extra, replace=False)):
        alloc[y] += 1
    parts = []
    for y in years:
        if alloc[y] > 0:
            parts += [(y, b, part) for b, part in enumerate(np.array_split(np.flatnonzero(yr == y), alloc[y]))]
    return alloc, parts


def plan_blocks(cands, split, n, seed):
    """Blocks with their candidates in a seeded random order (the first that passes the S3 checks is the primary)."""
    rng = np.random.default_rng(seed)
    alloc, parts = design_blocks(cands, split, n, rng)
    blocks = []
    for y, b, part in parts:
        order = rng.permutation(part)
        blocks.append({"stratum": y, "block": b, "block_first": str(cands["t"].iloc[part[0]]),
                       "block_last": str(cands["t"].iloc[part[-1]]), "block_candidates": int(len(part)),
                       "order": cands.iloc[order].to_dict("records")})
    return alloc, blocks


def design_spread(split, n, seed, sample_rmse, per_block=1, reps=2000, reps_cov=500, n_boot=2000):
    """CSV only, no model: how typical the drawn sample is. For each of my baselines, the RMSE of `reps` random
    samples drawn with the same design (same allocation and blocks, per_block uniformly random pairs per block), the
    percentile of the drawn sample's RMSE among them, and how often a percentile-bootstrap 95% interval from such a
    sample covers the RMSE over every pair of the split."""
    cands, _ = candidates(split)
    _, parts = design_blocks(cands, split, n, np.random.default_rng(seed))
    rng = np.random.default_rng(seed + 1)
    obs = cands["V_T"].to_numpy(float)
    preds = {"persistence_4d_V_at_t": cands["V_t"].to_numpy(float),
             "recurrence_27d_V_at_T_minus_27d": cands["V_T27"].to_numpy(float),
             "climatology_train_mean": np.full(len(cands), norm_constants()["train"]["mean"])}
    picks = np.concatenate([p[rng.integers(0, len(p), (reps, per_block))] for _, _, p in parts], 1)
    out = {"design": f"{len(parts)} blocks x {per_block} pair(s), {reps} draws (coverage: {reps_cov} draws x "
                     f"{n_boot} bootstrap resamples)"}
    for name, pr in preds.items():
        se = (pr - obs) ** 2
        full = float(np.sqrt(np.nanmean(se)))
        with np.errstate(invalid="ignore"):
            r = np.sqrt(np.nanmean(se[picks], 1))
        cover = 0
        for k in range(reps_cov):
            e = se[picks[k]]
            e = e[np.isfinite(e)]
            bs = np.sqrt(np.mean(e[rng.integers(0, len(e), (n_boot, len(e)))], 1))
            lo, hi = np.percentile(bs, [2.5, 97.5])
            cover += lo <= full <= hi
        sv = sample_rmse.get(name)
        out[name] = {"full_split_rmse": full, "design_rmse_median": float(np.median(r)),
                     "design_rmse_2.5_97.5": [float(v) for v in np.percentile(r, [2.5, 97.5])],
                     "drawn_sample_rmse": jnum(sv) if sv is not None else None,
                     "drawn_sample_percentile": float(100 * np.mean(r <= sv)) if sv is not None else None,
                     "bootstrap95_coverage_of_full_split_rmse": cover / reps_cov}
    return out


def regression_adjusted(se_model, X, x_full_mean, n_boot=N_BOOT, seed=0):
    """Regression (control-variate) estimator of the full-split MSE: mean(se_model) - b.(mean(X) - x_full_mean),
    b from least squares of se_model on X in the sample (Cochran, Sampling Techniques, 3rd ed., Ch. 7). X holds my
    baselines' squared errors, whose means over every pair of the split are known from the CSV. Approximately
    unbiased only for large n; with ~30 samples it is a sensitivity check, not a replacement for the plain mean."""
    se_model, X = np.asarray(se_model, float), np.asarray(X, float)
    n, k = X.shape

    def est(idx):
        S, Xs = se_model[idx], X[idx]
        Xc, Sc = Xs - Xs.mean(1, keepdims=True), S - S.mean(1, keepdims=True)
        b = np.linalg.solve(np.einsum("bnk,bnl->bkl", Xc, Xc) + 1e-9 * np.eye(k),
                            np.einsum("bnk,bn->bk", Xc, Sc)[..., None])[..., 0]
        return S.mean(1) - np.einsum("bk,bk->b", b, Xs.mean(1) - x_full_mean), b

    m, b = est(np.arange(n)[None])
    bm, _ = est(np.random.default_rng(seed).integers(0, n, (n_boot, n)))
    with np.errstate(invalid="ignore"):
        return {"mse": float(m[0]), "rmse": float(np.sqrt(m[0])) if m[0] > 0 else None,
                "ci95_mse": [float(v) for v in np.percentile(bm, [2.5, 97.5])],
                "ci95_rmse": [float(np.sqrt(max(v, 0))) for v in np.percentile(bm, [2.5, 97.5])],
                "b": [float(v) for v in b[0]], "n": int(n),
                "corr_with_model_sq_err": [float(np.corrcoef(se_model, X[:, j])[0, 1]) for j in range(k)]}


def head_check(t):
    ok0, b0 = s3_exists_retry(t)
    if not ok0:
        return False, "input frame at t not on S3", {}
    ok1, b1 = s3_exists_retry(t + pd.Timedelta(minutes=60))
    if not ok1:
        return False, "frame at t+60 min not on S3 (the authors' dataset indexes and reads t+60)", {}
    return True, "", {"frame_bytes": b0, "frame_t60_bytes": b1}


def frame_check(t, check_channels=True):
    """HEAD check of the frames at t and t+60, then (optionally) the 13-channel header check of both, in-process."""
    ok, why, extra = head_check(t)
    if ok and check_channels:
        nread = 0
        for which, ts in (("t", t), ("t+60", t + pd.Timedelta(minutes=60))):
            try:
                shapes, nb, _ = remote_nc_channels(C.s3_key(ts))
            except Exception as e:
                ok, why = False, f"could not read the netCDF header of the frame at {which}: {e!r}"
                break
            nread += nb
            bad = [c for c in C.CHANNELS if shapes.get(c) != (4096, 4096)]
            if bad:
                ok, why = False, f"frame at {which} lacks channels {bad}"
                break
        extra["header_bytes_read"] = nread
    return ok, why, extra


def select_anchors(split, check_channels=True):
    """Rows for the notebook anchor time stamps (validation split only), checked like any other sample."""
    frames, allv = labels()
    lab = frames[split].set_index("timestamp")["V"]
    rows, skipped = [], []
    for i, (ts, pred_nb, gt_nb) in enumerate(NOTEBOOK_ANCHORS):
        t = pd.Timestamp(ts)
        T = t + LEAD
        rec = {"t": str(t), "T": str(T), "V_T": float(lab.get(T, np.nan)), "V_t": float(allv.get(t, np.nan)),
               "V_T27": float(allv.get(T - LAG27, np.nan))}
        if not np.isfinite(rec["V_T"]):
            skipped.append({**rec, "stratum": "anchor", "reason": f"no row at t + 4 d in {split}.csv"})
            continue
        ok, why, extra = frame_check(t, check_channels)
        if not ok:
            skipped.append({**rec, "stratum": "anchor", "reason": why})
            continue
        rows.append({"split": split, "role": "anchor", "stratum": t.year, "block": -(i + 1),
                     "block_first": str(t), "block_last": str(t), "block_candidates": 1, "timestamp": str(t),
                     "target_time": str(T), "V_target_km_s": rec["V_T"], "V_at_input_time_km_s": rec["V_t"],
                     "V_target_minus_27d_km_s": rec["V_T27"], "frame_key": C.s3_key(t),
                     "frame_bytes": extra["frame_bytes"], "frame_t60_key": C.s3_key(t + pd.Timedelta(minutes=60)),
                     "frame_t60_bytes": extra["frame_t60_bytes"], "channels_checked": bool(check_channels),
                     "header_bytes_read": extra.get("header_bytes_read", 0),
                     "notebook_pred_km_s": pred_nb, "notebook_ground_truth_km_s": gt_nb,
                     "notebook_ground_truth_equals_csv": bool(abs(rec["V_T"] - gt_nb) < 1e-9)})
    return pd.DataFrame(rows), skipped


def select_sample(split, n, seed, check_channels=True, want_per_block=2, max_tries=30, header_workers=4):
    """Seeded, year-stratified sample with one primary and one reserve per time block (see module docstring)."""
    cands, info = candidates(split)
    alloc, blocks = plan_blocks(cands, split, n, seed)
    state = [{"pos": 0, "found": [], "skipped": []} for _ in blocks]
    ctx = mp.get_context("spawn")
    rounds = 0
    with cf.ThreadPoolExecutor(8) as tp, cf.ProcessPoolExecutor(header_workers, mp_context=ctx) as pp:
        while True:
            batch = []  # (block index, candidate row)
            for bi, (blk, st) in enumerate(zip(blocks, state)):
                need = want_per_block - len(st["found"])
                while need > 0 and st["pos"] < min(len(blk["order"]), max_tries):
                    batch.append((bi, blk["order"][st["pos"]]))
                    st["pos"] += 1
                    need -= 1
            if not batch:
                break
            rounds += 1
            heads = list(tp.map(lambda x: head_check(pd.Timestamp(x[1]["t"])), batch))
            chan = {}
            if check_channels:  # both frames are read by WindSpeedDSDataset, so both headers are checked
                futs = {}
                for j, h in enumerate(heads):
                    if h[0]:
                        t = pd.Timestamp(batch[j][1]["t"])
                        futs[j] = (pp.submit(remote_nc_channels, C.s3_key(t)),
                                   pp.submit(remote_nc_channels, C.s3_key(t + pd.Timedelta(minutes=60))))
                for j, (f0, f1) in futs.items():
                    res = []
                    for f in (f0, f1):
                        try:
                            res.append(f.result())
                        except Exception as e:  # unreadable header
                            res.append(e)
                    chan[j] = res
            for j, (bi, row) in enumerate(batch):  # assignment in permutation order -> deterministic
                ok, why, extra = heads[j]
                rec = {"t": str(row["t"]), "T": str(row["T"]), "V_T": row["V_T"], "V_t": row["V_t"],
                       "V_T27": row["V_T27"]}
                if ok and check_channels:
                    nread = 0
                    for which, r in zip(("t", "t+60"), chan[j]):
                        if isinstance(r, Exception):
                            ok, why = False, f"could not read the netCDF header of the frame at {which}: {r!r}"
                            break
                        shapes, nb, _ = r
                        nread += nb
                        bad = [c for c in C.CHANNELS if shapes.get(c) != (4096, 4096)]
                        if bad:
                            ok, why = False, f"frame at {which} lacks channels {bad}"
                            break
                    extra["header_bytes_read"] = nread
                if ok and len(state[bi]["found"]) < want_per_block:
                    state[bi]["found"].append({**rec, **extra})
                elif not ok:
                    state[bi]["skipped"].append({**rec, "stratum": blocks[bi]["stratum"], "reason": why})
    rows, skipped = [], []
    for blk, st in zip(blocks, state):
        skipped += st["skipped"]
        for role, f in zip(("primary", "reserve"), st["found"]):
            t = pd.Timestamp(f["t"])
            rows.append({"split": split, "role": role, "stratum": blk["stratum"], "block": blk["block"],
                         "block_first": blk["block_first"], "block_last": blk["block_last"],
                         "block_candidates": blk["block_candidates"], "timestamp": str(t), "target_time": f["T"],
                         "V_target_km_s": f["V_T"], "V_at_input_time_km_s": f["V_t"],
                         "V_target_minus_27d_km_s": f["V_T27"],
                         "frame_key": C.s3_key(t), "frame_bytes": f["frame_bytes"],
                         "frame_t60_key": C.s3_key(t + pd.Timedelta(minutes=60)),
                         "frame_t60_bytes": f["frame_t60_bytes"], "channels_checked": bool(check_channels),
                         "header_bytes_read": f.get("header_bytes_read", 0)})
    df = pd.DataFrame(rows)
    meta = {"split": split, "n": n, "seed": seed, "check_channels": bool(check_channels),
            "allocation_per_year": {int(k): int(v) for k, v in alloc.items()}, "candidate_info": info,
            "n_blocks": len(blocks), "blocks_without_primary": [
                {"stratum": b["stratum"], "block": b["block"]} for b, s in zip(blocks, state) if not s["found"]],
            "blocks_without_reserve": int(sum(len(s["found"]) < 2 for s in state)),
            "skipped": skipped, "selection_rounds": rounds}
    return df, meta


def split_csv_path(cfg, split):
    """validation: config.yaml's solarwind_valid_index (relative to WORK); test: the HF test.csv in the same place."""
    if split == "validation":
        return cfg["data"]["solarwind_valid_index"]
    return os.path.join(os.path.dirname(cfg["data"]["solarwind_valid_index"]), SPLIT_CSV[split])


def build_wind_dataset(cfg, scalers, split, t, sdo_root, WindSpeedDSDataset, quiet=True):
    """The authors' WindSpeedDSDataset for a single input time stamp, with finetune.get_dataloaders' validation
    arguments (config.yaml); only the SDO index (t, t+60) and, for the test split, the label CSV differ."""
    t = pd.Timestamp(t)
    index_path = os.path.join(WORK, "index", f"{split}_{stamp(t)}.csv")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    C.write_index([t, t + pd.Timedelta(minutes=cfg["data"]["time_delta_target_minutes"])], index_path)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
        ds = WindSpeedDSDataset(
            sdo_data_root_path=sdo_root,
            index_path=index_path,
            time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
            n_input_timestamps=cfg["model"]["time_embedding"]["time_dim"],
            rollout_steps=cfg["rollout_steps"],
            channels=cfg["data"]["channels"],
            drop_hmi_probability=cfg["drop_hmi_probability"],
            num_mask_aia_channels=cfg["num_mask_aia_channels"],
            use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"],
            scalers=scalers,
            phase="valid",
            ds_solar_wind_path=split_csv_path(cfg, split),
            ds_time_column=cfg["data"]["ds_time_column"],
            ds_time_delta_in_out=cfg["data"]["ds_time_delta_in_out"],
            ds_time_tolerance=cfg["data"]["ds_time_tolerance"],
            ds_match_direction=cfg["data"]["ds_match_direction"],
            ds_normalize=cfg["data"]["ds_normalize"],
            ds_scaler=cfg["data"]["ds_scaler"],
        )
    ds._construction_stdout = buf.getvalue()
    return ds


def dataset_label(ds):
    """(target time, normalised V, V un-normalised with the dataset's own constants) of the one-sample dataset."""
    row = ds.df_valid_indices.iloc[0]
    return pd.Timestamp(row["ds_index"]), float(row["V"]), float(ds.un_norm_output(row["V"]))


def load_configs():
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))  # finetune.py's config (dataset arguments)
    cfg_inf = yaml.safe_load(open(os.path.join(TASK_DIR, "config_infer.yaml")))  # infer.py's default (model)
    diff = {}
    for k in sorted(set(cfg["model"]) | set(cfg_inf["model"])):
        if cfg["model"].get(k) != cfg_inf["model"].get(k):
            diff[f"model.{k}"] = (cfg["model"].get(k), cfg_inf["model"].get(k))
    for k in sorted(set(cfg["data"]) | set(cfg_inf["data"])):
        if cfg["data"].get(k) != cfg_inf["data"].get(k):
            diff[f"data.{k}"] = (cfg["data"].get(k), cfg_inf["data"].get(k))
    for k in ("optimizer",):
        if cfg[k] != cfg_inf[k]:
            diff[k] = (cfg[k], cfg_inf[k])
    return cfg, cfg_inf, diff


def check_joins(sample, cfg, scalers, split):
    """Build the authors' dataset for every sampled time (no frame is read) and confirm it yields exactly that input
    time, the target time t + 4 d, and the CSV's V after un-normalising with its own constants."""
    sys.path.insert(0, TASK_DIR)
    from dataset import WindSpeedDSDataset
    out = []
    for r in sample.itertuples():
        ds = build_wind_dataset(cfg, scalers, split, r.timestamp, os.path.join(WORK, "frames", "a"),
                                WindSpeedDSDataset)
        ok = len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == pd.Timestamp(r.timestamp)
        T, vn, v = dataset_label(ds) if len(ds) else (None, None, None)
        ok = ok and T == pd.Timestamp(r.target_time) and abs(v - r.V_target_km_s) < 1e-9
        out.append({"timestamp": r.timestamp, "role": r.role, "dataset_len": len(ds), "ok": bool(ok),
                    "target_time_from_dataset": str(T), "V_norm_from_dataset": vn, "V_unnorm_from_dataset": v,
                    "dataset_un_norm_min": float(ds.un_norm_min), "dataset_un_norm_ptp": float(ds.un_norm_ptp),
                    "sdo_paths_read_by_dataset": list(ds.index["path"])})
    return out


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


def load_model(cfg_inf, ckpt_path):
    """infer.load_model, step by step, with a safe unpickler and memory-mapped weights."""
    import torch
    sys.path.insert(0, TASK_DIR)
    import infer as I  # authors' get_model / apply_peft_lora (imported by infer.py from finetune.py)
    from surya.utils.distributed import set_global_seed
    set_global_seed(42)  # as infer.main
    model = I.get_model(cfg_inf, wandb_logger=None)
    if cfg_inf["model"]["use_lora"]:
        model = I.apply_peft_lora(model, cfg_inf)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
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
    return model, {"n_parameters": int(n_total), "n_trainable_after_peft": int(n_trainable),
                   "state_dict_keys": int(n_keys), "load_state_dict_strict": True}


def verify_frames(paths, channels):
    """Fail fast before the authors' __getitem__ (which retries a failing index 100 times, re-reading the frames):
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


def read_records(path):
    recs = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if line:
                r = json.loads(line)
                recs[r["timestamp"]] = r  # last record for a time stamp wins
    return recs


def csv_baselines(split, consts, seed=0):
    """My baselines on every candidate pair of the split (CSV only, no model, no S3 check)."""
    cands, info = candidates(split)
    clim = consts["train"]["mean"]
    out = {"pairs": info["candidates"], "definition": "all (t, T = t + 4 d) pairs of the split as in the sampling "
           "frame (test: t >= 2020-01-01; validation: t, t+60 in core-sdo valid index), S3 not checked",
           "V_target_mean": float(cands["V_T"].mean()), "V_target_std": float(cands["V_T"].std(ddof=0))}
    m = cands["V_t"].notna()
    out["persistence_4d_V_at_t"] = {**errors_metrics(cands.loc[m, "V_t"], cands.loc[m, "V_T"])}
    m = cands["V_T27"].notna()
    out["recurrence_27d_V_at_T_minus_27d"] = {**errors_metrics(cands.loc[m, "V_T27"], cands.loc[m, "V_T"])}
    out["climatology_train_mean"] = {**errors_metrics(np.full(len(cands), clim), cands["V_T"]),
                                     "value_km_s": clim}
    return out


def anchor_check(allrec, consts):
    """Our network output on the notebook's three samples, un-normalised the way infer.py does (validation.csv
    constants), against the predictions the notebook printed. Checks the whole pipeline (frames from S3, dataset,
    model build, LoRA wrap, strict load, float32 CPU forward), not the paper's numbers."""
    va, tr = consts["validation"], consts["train"]
    rows = []
    for ts, pred_nb, gt_nb in NOTEBOOK_ANCHORS:
        r = allrec.get(str(pd.Timestamp(ts)))
        if r is None or r.get("status") != "ok":
            rows.append({"timestamp": ts, "status": "not run" if r is None else r.get("status"),
                         "error": None if r is None else r.get("error", "")[:300]})
            continue
        y = float(r["model_output_normalised"])
        ours = y * va["ptp"] + va["min"]
        rows.append({"timestamp": ts, "status": "ok", "notebook_pred_km_s": pred_nb,
                     "ours_km_s_validation_csv_constants": ours, "diff_km_s": ours - pred_nb,
                     "within_tolerance": bool(abs(ours - pred_nb) <= ANCHOR_TOL_KM_S),
                     "ours_normalised": y, "notebook_normalised": (pred_nb - va["min"]) / va["ptp"],
                     "ours_km_s_train_constants": y * tr["ptp"] + tr["min"],
                     "ground_truth_notebook_km_s": gt_nb, "ground_truth_ours_km_s": r["V_target_km_s"],
                     "ground_truth_equal": bool(abs(float(r["V_target_km_s"]) - gt_nb) < 1e-9),
                     "forward_s": r["seconds"]["forward"]})
    run = [x for x in rows if x["status"] == "ok"]
    return {"source": "solar_wind_tutorial.ipynb cell 9 (infer.py with config_infer.yaml; CPU, autocast disabled, "
                      "float32; torch 2.8.0+cu126); printed to 2 decimals",
            "tolerance_km_s": ANCHOR_TOL_KM_S, "n_run": len(run),
            "all_within_tolerance": bool(all(x["within_tolerance"] for x in run)) if run else None,
            "max_abs_diff_km_s": float(max(abs(x["diff_km_s"]) for x in run)) if run else None,
            "samples": rows}


def summarize(out_dir, seed=0):
    consts = norm_constants()
    tr, va = consts["train"], consts["validation"]
    summary = {
        "label": "PARTIAL REPRODUCTION: released fine-tuned weights scored on a small seeded sample of the "
                 "SuryaBench solar-wind split named below, on CPU in float32. Not the paper's (unstated) "
                 "evaluation set.",
        "paper_reference": {"source": "arXiv 2508.14112 v2, Sec. 2.6.5, Table 6", "values": PAPER_TABLE6,
                            "table7_27day_persistence_rmse_km_s": PAPER_TABLE7_27DAY_PERSISTENCE,
                            "split_in_paper": "not stated", "metric_definition_in_paper": "not stated "
                            "(task equation y_{t+dt} = f(x_t), dt = 4 days, Sec. 2.6.5)",
                            "note": "sqrt(5698.62) = 75.49 != 75.92: RMSE and Validation MSE are not from the "
                                    "same set/aggregation"},
        "primary_prediction": "model output * ptp(train.csv V) + min(train.csv V) = y * 560 + 240 km/s",
        "normalisation_constants": consts,
        "metrics": "RMSE = sqrt(mean((pred-obs)^2)), MAE = mean|pred-obs|, MSE = mean((pred-obs)^2) in km/s over "
                   "samples (as infer.py; finetune.evaluate_model computes the same in normalised units)",
        "bootstrap": {"resamples": N_BOOT, "seed": seed, "interval": "percentile 95%, resampling samples"},
        "baselines_note": "persistence_4d, recurrence_27d and climatology_train_mean are MY baselines, not in "
                          "Table 6",
        "splits": {}}
    for split in ("test", "validation"):
        allrec = read_records(per_sample_path(out_dir, split))
        anchors = anchor_check(allrec, consts) if split == "validation" else None
        recs = [r for r in allrec.values() if r.get("status") == "ok" and r.get("role") in ("primary", "reserve")]
        failed = [r for r in allrec.values() if r.get("status") != "ok"]
        if anchors and anchors["n_run"]:
            summary["splits"].setdefault(split, {})["notebook_anchor_check"] = anchors
        if not recs:
            continue
        # every scored primary/reserve: a reserve is run only in place of a failed primary, or for all blocks with
        # --include-reserves (then two random pairs per block)
        recs = sorted(recs, key=lambda r: r["timestamp"])
        per_block = int(pd.Series([(int(r["stratum"]), int(r["block"])) for r in recs]).value_counts().max())
        obs = np.array([r["V_target_km_s"] for r in recs], float)
        y = np.array([r["model_output_normalised"] for r in recs], float)
        sc = consts[split]
        preds = {"train_constants_primary": y * tr["ptp"] + tr["min"],
                 "validation_csv_constants_infer_py_as_shipped": y * va["ptp"] + va["min"],
                 f"{split}_csv_constants_infer_py_on_this_split": y * sc["ptp"] + sc["min"]}
        res = {"model": {k: scored(v, obs, seed=seed) for k, v in preds.items()}}
        p_main = preds["train_constants_primary"]
        cmp = {}
        for key, ref_key in (("rmse", "rmse_km_s"), ("mae", "mae_km_s"), ("mse", "validation_mse_km2_s2")):
            ours = res["model"]["train_constants_primary"]
            lo, hi = ours["ci95"][key]
            ref = PAPER_TABLE6["Surya"][ref_key]
            cmp[key] = {"ours": ours[key], "ci95": [lo, hi], "paper_surya": ref, "paper_inside_ci": bool(lo <= ref <= hi),
                        "ours_minus_paper": ours[key] - ref, "paper_resnet50": PAPER_TABLE6["ResNet50"][ref_key],
                        "paper_alexnet": PAPER_TABLE6["AlexNet"][ref_key]}
        base = {}
        vt = np.array([np.nan if r.get("V_at_input_time_km_s") is None else r["V_at_input_time_km_s"] for r in recs])
        v27 = np.array([np.nan if r.get("V_target_minus_27d_km_s") is None else r["V_target_minus_27d_km_s"]
                        for r in recs])
        for name, bp in (("persistence_4d_V_at_t", vt), ("recurrence_27d_V_at_T_minus_27d", v27)):
            m = np.isfinite(bp)
            if m.sum() >= 2:
                base[name] = {**scored(bp[m], obs[m], seed=seed),
                              "model_minus_baseline_rmse_paired": paired_rmse_diff(p_main[m], bp[m], obs[m],
                                                                                   seed=seed)}
        clim = np.full(len(obs), tr["mean"])
        base["climatology_train_mean"] = {**scored(clim, obs, seed=seed), "value_km_s": tr["mean"],
                                          "model_minus_baseline_rmse_paired": paired_rmse_diff(p_main, clim, obs,
                                                                                               seed=seed)}
        res["baselines_on_same_samples"] = base
        # how typical the scored samples are (CSV only) and a control-variate estimate of the full-split MSE
        meta_json = os.path.join(out_dir, f"sampling_{split}.json")
        if os.path.exists(meta_json):
            sm = json.load(open(meta_json))
            res["sample_representativeness_csv_only"] = design_spread(
                split, sm["n"], sm["seed"], {k: v["rmse"] for k, v in base.items()}, per_block=per_block)
        cands, _ = candidates(split)
        full = cands[cands["V_t"].notna()]
        X_full = np.array([np.mean((full["V_t"] - full["V_T"]) ** 2), np.mean((tr["mean"] - full["V_T"]) ** 2)])
        m = np.isfinite(vt)
        if m.sum() >= 8:
            Xs = np.stack([(vt[m] - obs[m]) ** 2, (tr["mean"] - obs[m]) ** 2], 1)
            ra = regression_adjusted((p_main[m] - obs[m]) ** 2, Xs, X_full, seed=seed)
            ra.update({"label": "MY ADJUSTMENT (not the paper's): estimate of the model's MSE/RMSE over every pair "
                                "of the split that has V(t), using the squared errors of 4-day persistence and of "
                                "the training-mean climatology as control variates",
                       "control_variates": ["persistence_4d sq. error", "climatology_train_mean sq. error"],
                       "full_split_means_of_control_variates": [float(v) for v in X_full],
                       "sample_means_of_control_variates": [float(v) for v in Xs.mean(0)],
                       "unadjusted_on_same_rows": {"mse": float(np.mean((p_main[m] - obs[m]) ** 2)),
                                                   "rmse": float(np.sqrt(np.mean((p_main[m] - obs[m]) ** 2)))}})
            res["model_regression_adjusted_full_split"] = ra
        # finetune.evaluate_model's criterion: MSELoss between the raw output and the target scaled with the
        # constants of the CSV the dataset was built on (for the validation split that is exactly its valid loss)
        tn = np.array([r["target_normalised_by_dataset"] for r in recs], float)
        rl = float(np.mean((y - tn) ** 2))
        res["repo_style_loss_normalised"] = {
            "mean_MSELoss_output_vs_dataset_target": rl,
            "times_ptp_squared_km2_s2": rl * sc["ptp"] ** 2,
            "note": f"targets scaled with {split}.csv constants (min {sc['min']}, range {sc['ptp']}) while the "
                    f"network was trained on train.csv constants; for the validation split this is what "
                    f"finetune.evaluate_model reports as valid/loss on these samples"}
        years = np.array([int(r["stratum"]) for r in recs])
        per_year = {}
        for yv in np.unique(years):
            m = years == yv
            per_year[int(yv)] = {"n": int(m.sum()), **{k: v for k, v in errors_metrics(p_main[m], obs[m]).items()
                                                       if k in ("rmse", "mae", "bias")}}
        summary["splits"].setdefault(split, {}).update({
            "n_samples_scored": len(recs), "n_failed_records": len(failed), "max_samples_per_block": per_block,
            "failed": [{"timestamp": r["timestamp"], "error": r.get("error", "")[:300]} for r in failed],
            "roles": {k: int(v) for k, v in pd.Series([r["role"] for r in recs]).value_counts().items()},
            "time_range_input": [recs[0]["timestamp"], recs[-1]["timestamp"]],
            "V_target_sample_mean": float(obs.mean()), "V_target_sample_std": float(obs.std()),
            "metrics": res, "comparison_with_paper_table6_surya": cmp, "per_year": per_year,
            "baselines_full_split_csv_only": csv_baselines(split, consts, seed),
            "checks": {"all_inputs_finite": all(r["input"]["nonfinite"] == 0 for r in recs),
                       "all_outputs_finite": all(np.isfinite(r["model_output_normalised"]) for r in recs),
                       "dataset_target_equals_csv_V": all(r["dataset_target_unnorm_equals_csv"] for r in recs),
                       "finetune_permute_noop_all": all(r.get("finetune_permute_is_noop", False) for r in recs)},
            "timing": {"forward_s_mean": float(np.mean([r["seconds"]["forward"] for r in recs])),
                       "forward_s_max": float(np.max([r["seconds"]["forward"] for r in recs])),
                       "dataset_and_load_s_mean": float(np.mean([r["seconds"]["dataset_and_load"] for r in recs])),
                       "sample_s_mean": float(np.mean([r["seconds"]["total"] for r in recs])),
                       "peak_rss_gb_max": float(np.max([r["peak_rss_gb"] for r in recs]))},
            "per_sample": [{"timestamp": r["timestamp"], "target_time": r["target_time"], "stratum": r["stratum"],
                            "V_obs": float(obs[i]), "V_pred_train_constants": float(p_main[i]),
                            "V_pred_validation_csv_constants": float(
                                preds["validation_csv_constants_infer_py_as_shipped"][i]),
                            "V_at_input_time": jnum(vt[i]), "V_T_minus_27d": jnum(v27[i]),
                            "model_output_normalised": float(y[i])} for i, r in enumerate(recs)],
        })
    env = os.path.join(out_dir, "environment.json")
    if os.path.exists(env):
        summary["environment"] = json.load(open(env))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    return summary


# ---------------------------------------------------------------------------------------------------- main
def get_sample(args, out_dir, cfg, scalers, force=False):
    sample_csv = os.path.join(out_dir, f"sample_{args.split}.csv")
    meta_json = os.path.join(out_dir, f"sampling_{args.split}.json")
    want_anchors = args.split == "validation" and not args.no_anchors
    if not force and os.path.exists(sample_csv) and os.path.exists(meta_json):
        meta = json.load(open(meta_json))
        if (meta["n"], meta["seed"], meta["check_channels"], meta.get("anchors", False)) == \
                (args.n, args.seed, not args.no_channel_check, want_anchors):
            log(f"reusing {sample_csv}")
            return pd.read_csv(sample_csv), meta
    t0 = time.time()
    sample, meta = select_sample(args.split, args.n, args.seed, check_channels=not args.no_channel_check,
                                 header_workers=args.header_workers)
    meta["anchors"] = want_anchors
    if want_anchors:
        anc, anc_skipped = select_anchors(args.split, check_channels=not args.no_channel_check)
        meta["anchor_info"] = {
            "source": "solar_wind_tutorial.ipynb cell 9 saved output (infer.py, CPU, float32, torch 2.8.0+cu126)",
            "tolerance_km_s": ANCHOR_TOL_KM_S, "requested": [a[0] for a in NOTEBOOK_ANCHORS],
            "kept": list(anc["timestamp"]) if len(anc) else [], "skipped": anc_skipped,
            "coincide_with_stratified_sample": sorted(set(anc["timestamp"]) & set(sample["timestamp"]))
            if len(anc) else [],
            "notebook_ground_truth_equals_csv_all": bool(anc["notebook_ground_truth_equals_csv"].all())
            if len(anc) else None}
        sample = pd.concat([sample, anc], ignore_index=True)
    meta["selection_seconds"] = round(time.time() - t0, 1)
    meta["dataset_join_check"] = check_joins(sample, cfg, scalers, args.split)
    meta["dataset_join_all_ok"] = all(x["ok"] for x in meta["dataset_join_check"])
    sample.to_csv(sample_csv, index=False)
    json.dump(meta, open(meta_json, "w"), indent=1, default=str)
    return sample, meta


def bandwidth_probe(sample):
    out = {}
    for name, url in (("s3_frame", f"{C.S3}/{sample['frame_key'].iloc[0]}"),
                      ("hf_checkpoint", hf_url(CKPT_REPO, CKPT_REV, CKPT_NAME, kind="models"))):
        t0 = time.time()
        r = requests.get(url, headers={"Range": "bytes=0-16777215"}, timeout=120)
        r.raise_for_status()
        out[name] = {"bytes": len(r.content), "seconds": round(time.time() - t0, 2),
                     "MB_per_s": round(len(r.content) / 1e6 / max(1e-3, time.time() - t0), 1)}
    return out


def sample_baselines(sample, consts, seed, roles=("primary",)):
    """My baselines on the sampled primaries (CSV only): lets the dry run show how typical the sample is."""
    prim = sample[sample.role.isin(roles)]
    obs = prim["V_target_km_s"].to_numpy(float)
    out = {"n": int(len(prim)), "V_target_mean": float(obs.mean()), "V_target_std": float(obs.std())}
    for name, col in (("persistence_4d_V_at_t", "V_at_input_time_km_s"),
                      ("recurrence_27d_V_at_T_minus_27d", "V_target_minus_27d_km_s")):
        v = prim[col].to_numpy(float)
        m = np.isfinite(v)
        out[name] = scored(v[m], obs[m], seed=seed) if m.sum() >= 2 else None
    out["climatology_train_mean"] = scored(np.full(len(obs), consts["train"]["mean"]), obs, seed=seed)
    return out


def dry_run(args, out_dir, label_info):
    from surya.utils.data import build_scalers
    cfg, cfg_inf, cfg_diff = load_configs()
    scalers = build_scalers(info=yaml.safe_load(open(cfg["data"]["scalers_path"])))
    check = metric_self_test(out_dir)
    log(f"metric self-test passed (evaluate_model rel diff {check['finetune_evaluate_model_max_rel_diff']:.1e}, "
        f"infer.py printout abs diff {check['infer_single_sample_max_abs_diff']:.1e})")
    sample, meta = get_sample(args, out_dir, cfg, scalers, force=True)
    # what a run executes: anchors first (validation split), then one primary per block
    roles = ["anchor", "primary"] + (["reserve"] if args.include_reserves else [])
    prim = pd.concat([sample[sample.role == r] for r in roles], ignore_index=True).drop_duplicates("timestamp")
    consts = norm_constants()
    bw = bandwidth_probe(sample)
    ck = requests.head(hf_url(CKPT_REPO, CKPT_REV, CKPT_NAME, kind="models"), allow_redirects=False, timeout=60)
    s3_bw, hf_bw = bw["s3_frame"]["MB_per_s"] * 1e6, bw["hf_checkpoint"]["MB_per_s"] * 1e6
    fb = (prim["frame_bytes"] + prim["frame_t60_bytes"]).to_numpy(float)  # both frames of each sample
    ckpt_present = os.path.exists(os.path.join(WORK, CKPT_NAME)) and \
        os.path.getsize(os.path.join(WORK, CKPT_NAME)) == CKPT_SIZE
    one_off = (0 if ckpt_present else CKPT_SIZE / hf_bw) + SEC_MODEL
    per_sample = 2 * SEC_LOAD_FRAME + SEC_FORWARD + SEC_SCORE
    dl = fb / s3_bw  # seconds to download the two frames of each sample; sample j+1 downloads while j runs
    hidden = not args.no_prefetch
    if hidden:
        total_s = one_off + (dl[0] if len(dl) else 0.0) + len(dl) * per_sample + \
            float(np.maximum(0, dl[1:] - per_sample).sum())
        frames_max = int(max([fb[i] + fb[i + 1] for i in range(len(fb) - 1)] or [fb.max()]))
    else:
        total_s = one_off + float(dl.sum()) + len(dl) * per_sample
        frames_max = int(fb.max())
    est = {"assumptions_s": {"forward": SEC_FORWARD, "frame_read_and_normalise_each": SEC_LOAD_FRAME,
                             "frames_per_sample": 2, "scoring": SEC_SCORE, "model_build_and_load": SEC_MODEL},
           "measured_bandwidth": bw, "checkpoint_already_on_disk": bool(ckpt_present),
           "one_off_minutes_checkpoint_and_model": round(one_off / 60, 1),
           "per_sample_minutes": round(per_sample / 60, 2),
           "frame_download_minutes_per_sample_mean": round(float(dl.mean()) / 60, 2) if len(dl) else 0,
           "prefetch": hidden, "download_hidden_by_prefetch": bool(hidden and dl.max() < per_sample),
           "total_minutes": round(total_s / 60, 1),
           # forward passes that fit in one ~90-minute run; a rerun of the same command resumes from the jsonl
           "max_forward_per_90_min_run": int(max(1, (90 * 60 - one_off - (dl[0] if len(dl) else 0)) // per_sample))}
    disk = {"frames_total_bytes_downloaded_over_run": int(fb.sum()), "frames_max_on_disk_bytes": frames_max,
            "checkpoint_bytes": CKPT_SIZE, "checkpoint_head_x_linked_size": ck.headers.get("x-linked-size"),
            "foundation_weights_already_on_disk_linked_not_copied": os.path.join(FM_DIR, "surya.366m.v1.pt")}
    disk["peak_disk_gb_new"] = round((frames_max + (0 if ckpt_present else CKPT_SIZE)) / 1e9, 2)
    if frames_max > 3e9:
        log("WARNING: more than 3 GB of frames would be on disk at once; use --no-prefetch")
    meta.update({"dry_run": True, "estimate": est, "disk": disk, "config_diff_config_vs_config_infer": cfg_diff,
                 "forward_passes_planned": int(len(prim)), "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}",
                 "checkpoint_sha256": CKPT_SHA256, "labels": f"{LABEL_REPO}@{LABEL_REV}", "label_files": label_info,
                 "normalisation_constants": consts, "upstream_dir": UPSTREAM,
                 "upstream_head": os.popen(f"git -C {UPSTREAM} rev-parse --short HEAD").read().strip(),
                 "baselines_on_sample_csv_only": sample_baselines(sample, consts, args.seed),
                 "baselines_on_sample_with_reserves_csv_only": sample_baselines(sample, consts, args.seed,
                                                                                ("primary", "reserve")),
                 "baselines_full_split_csv_only": csv_baselines(args.split, consts, args.seed)})
    rep = {}
    for key, pb in (("primaries", 1), ("primaries_and_reserves", 2)):
        sb_ = meta["baselines_on_sample_csv_only" if pb == 1 else "baselines_on_sample_with_reserves_csv_only"]
        rep[key] = design_spread(args.split, args.n, args.seed,
                                 {k: v["rmse"] for k, v in sb_.items() if isinstance(v, dict) and "rmse" in v},
                                 per_block=pb)
    meta["sample_representativeness_csv_only"] = rep
    json.dump(meta, open(os.path.join(out_dir, f"sampling_{args.split}.json"), "w"), indent=1, default=str)
    sb, fs = meta["baselines_on_sample_csv_only"], meta["baselines_full_split_csv_only"]
    log(f"split {args.split}: {meta['candidate_info']['candidates']} candidate pairs, "
        f"{int((sample.role == 'primary').sum())} primaries + {int((sample.role == 'reserve').sum())} reserves + "
        f"{int((sample.role == 'anchor').sum())} anchors, {len(meta['skipped'])} skipped, "
        f"join check ok: {meta['dataset_join_all_ok']}; {len(prim)} forward passes planned")
    p4s, p4f = sb["persistence_4d_V_at_t"], fs["persistence_4d_V_at_t"]
    log(f"my 4-day persistence RMSE: sample {p4s['rmse']:.1f} km/s {[round(v, 1) for v in p4s['ci95']['rmse']]} "
        f"(n={p4s['n']}), full split {p4f['rmse']:.1f} km/s (n={p4f['n']}); paper Surya 75.92")
    for key, r in rep.items():
        for name in ("persistence_4d_V_at_t", "climatology_train_mean"):
            x = r[name]
            if x["drawn_sample_rmse"] is not None:
                log(f"  {key}: {name} RMSE {x['drawn_sample_rmse']:.1f} is at the {x['drawn_sample_percentile']:.1f} "
                    f"percentile of same-design samples (median {x['design_rmse_median']:.1f}, full split "
                    f"{x['full_split_rmse']:.1f}); bootstrap 95% coverage {x['bootstrap95_coverage_of_full_split_rmse']:.2f}")
    log(f"frames to download over the run: {fb.sum() / 1e9:.2f} GB (at most {frames_max / 1e9:.2f} GB on disk at "
        f"once); checkpoint {CKPT_SIZE / 1e9:.2f} GB")
    log(f"estimated {est['total_minutes']} min ({len(prim)} forward passes); new disk at peak "
        f"~{disk['peak_disk_gb_new']} GB")
    if est["total_minutes"] > 90:
        log(f"more than 90 min: run with --max-forward {est['max_forward_per_90_min_run']} and repeat the same "
            f"command until it reports 0 samples to run (each rerun resumes)")
    if meta.get("anchors"):
        ai = meta["anchor_info"]
        log(f"notebook anchors kept: {ai['kept']} (skipped {len(ai['skipped'])}); notebook ground truth equals "
            f"{args.split}.csv: {ai['notebook_ground_truth_equals_csv_all']}")
    print(sample[["role", "stratum", "block", "timestamp", "target_time", "V_target_km_s",
                  "V_at_input_time_km_s"]].to_string(index=False))


def run(args, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from surya.utils.data import build_scalers
    sys.path.insert(0, TASK_DIR)
    from dataset import WindSpeedDSDataset
    from finetune import custom_collate_fn

    cfg, cfg_inf, _ = load_configs()
    cfg_inf["data"]["scalers"] = yaml.safe_load(open(cfg_inf["data"]["scalers_path"]))  # as infer.main
    cfg_inf["dtype"] = torch.float32  # as infer.main for dtype "float32"; HelioSpectFormer ignores it
    scalers = build_scalers(info=yaml.safe_load(open(cfg["data"]["scalers_path"])))
    metric_self_test(out_dir)
    sample, meta = get_sample(args, out_dir, cfg, scalers)
    if not meta.get("dataset_join_all_ok", False):
        raise RuntimeError("dataset join check failed in the sample file; rerun --dry-run")
    consts = norm_constants()

    recs_path = per_sample_path(out_dir, args.split)
    previous = read_records(recs_path)
    done = {t for t, r in previous.items() if r.get("status") == "ok"}
    blocks, anchors = {}, []
    for r in sample.to_dict("records"):
        if r["role"] == "anchor":
            anchors.append(r)
        else:
            blocks.setdefault((int(r["stratum"]), int(r["block"])), {})[r["role"]] = r
    ok_blocks = {(int(r["stratum"]), int(r["block"])) for r in previous.values()
                 if r.get("status") == "ok" and r.get("role") in ("primary", "reserve")}
    work, complete = [], 0
    for key, b in sorted(blocks.items()):
        if key in ok_blocks:  # one sample per block: a block is finished once its primary or reserve is scored
            complete += 1
            continue
        prim, res = b.get("primary"), b.get("reserve")
        failed_before = prim is not None and previous.get(prim["timestamp"], {}).get("status") == "failed"
        if prim is not None and not failed_before:
            work.append(prim)
        elif res is not None:
            work.append(res)
    if args.include_reserves:  # second random pair of every block, after all primaries, interleaved across years
        in_work = {r["timestamp"] for r in work}
        for key in sorted(blocks, key=lambda k: (k[1], k[0])):
            res = blocks[key].get("reserve")
            if res is not None and res["timestamp"] not in in_work:
                work.append(res)
    # notebook anchors run first (an early end-to-end check); one that is also a stratified sample is run once
    in_work = {r["timestamp"] for r in work}
    work = [a for a in anchors if a["timestamp"] not in in_work] + work
    pending = [r for r in work if r["timestamp"] not in done][: args.max_forward]
    log(f"{len(blocks)} blocks in the sample, {complete} already scored, {len(anchors)} anchors, "
        f"{len(pending)} samples to run now")
    if not pending:
        summarize(out_dir, args.seed)
        return

    # disk pre-flight: checkpoint (if absent) + frames of one sample (--no-prefetch) or two, plus 0.5 GB margin
    import shutil
    ck_local = os.path.join(WORK, CKPT_NAME)
    need_ck = 0 if (os.path.exists(ck_local) and os.path.getsize(ck_local) == CKPT_SIZE) else CKPT_SIZE
    fr = sorted((float(r["frame_bytes"]) + float(r["frame_t60_bytes"]) for r in pending), reverse=True)
    need = need_ck + sum(fr[: 1 if args.no_prefetch else 2]) + 0.5e9
    free = shutil.disk_usage(WORK).free
    if free < need:
        raise RuntimeError(f"{free / 1e9:.1f} GB free in {WORK}, need ~{need / 1e9:.1f} GB"
                           + ("" if args.no_prefetch else "; --no-prefetch keeps only one sample's frames"))
    ckpt = ensure_checkpoint()
    ensure_pretrained_link(cfg_inf)
    t0 = time.time()
    model, minfo = load_model(cfg_inf, ckpt)
    minfo["seconds_build_and_load"] = round(time.time() - t0, 1)
    env = {"torch": torch.__version__, "threads": torch.get_num_threads(), "precision": "float32, no autocast",
           "upstream_head": os.popen(f"git -C {UPSTREAM} rev-parse --short HEAD").read().strip(),
           "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}", "checkpoint_sha256": CKPT_SHA256,
           "labels": f"{LABEL_REPO}@{LABEL_REV}", "model": minfo, "normalisation_constants": consts,
           "peak_rss_gb_after_model_load": C.peak_rss_gb()}
    json.dump(env, open(os.path.join(out_dir, "environment.json"), "w"), indent=1, default=str)
    log(f"model ready in {minfo['seconds_build_and_load']} s, peak RSS {C.peak_rss_gb()} GB")

    tr, va, sc = consts["train"], consts["validation"], consts[args.split]
    caches = [C.FrameCache(os.path.join(WORK, "frames", "a")), C.FrameCache(os.path.join(WORK, "frames", "b"))]
    pool = cf.ThreadPoolExecutor(1)

    def frames_of(row):
        t = pd.Timestamp(row["timestamp"])
        return [t, t + pd.Timedelta(minutes=cfg["data"]["time_delta_target_minutes"])]

    fut = pool.submit(caches[0].ensure, frames_of(pending[0]))
    j = 0
    while j < len(pending):
        row = pending[j]
        t = pd.Timestamp(row["timestamp"])
        cache = caches[j % 2]
        rec = {k: row.get(k) for k in ("split", "role", "stratum", "block", "timestamp", "target_time",
                                       "V_target_km_s", "frame_key", "frame_bytes", "frame_t60_key",
                                       "frame_t60_bytes")}
        for k in ("V_at_input_time_km_s", "V_target_minus_27d_km_s"):
            rec[k] = jnum(row.get(k))
        t_start = time.time()
        ds = loader = batch = x = out = None
        try:
            dl_s = fut.result()
        except Exception as e:
            dl_s, err = None, e
        else:
            err = None
        if args.no_prefetch:
            cache_next = None
        elif j + 1 < len(pending):  # prefetch the next sample's two frames into the other cache
            fut = pool.submit(caches[(j + 1) % 2].ensure, frames_of(pending[j + 1]))
        try:
            if err is not None:
                raise err
            t1 = time.time()
            verify_frames([os.path.join(cache.root, C.s3_key(ts)) for ts in frames_of(row)], cfg["data"]["channels"])
            ds = build_wind_dataset(cfg, scalers, args.split, t, cache.root, WindSpeedDSDataset)
            assert len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == t, "dataset did not select the time stamp"
            T, v_norm, v_un = dataset_label(ds)
            assert T == pd.Timestamp(row["target_time"]), f"dataset target time {T} != {row['target_time']}"
            rec["dataset_un_norm_min"], rec["dataset_un_norm_ptp"] = float(ds.un_norm_min), float(ds.un_norm_ptp)
            loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
            batch, metadata = next(iter(loader))
            t2 = time.time()
            fc = batch.pop("forecast")  # the t+60 frame; HelioSpectformer1D never reads it
            rec["forecast_frame"] = {"shape": list(fc.shape), "nonfinite": int((~torch.isfinite(fc)).sum())}
            del fc
            x = batch["ts"]
            sl = x[..., :256, :256]
            rec["finetune_permute_is_noop"] = bool(torch.equal(sl.flatten(1, 2),
                                                               sl.permute(0, 2, 1, 3, 4).flatten(1, 2)))
            del sl
            rec["input"] = {"shape": list(x.shape), "dtype": str(x.dtype), "nonfinite": int((~torch.isfinite(x)).sum()),
                            "time_delta_input": batch["time_delta_input"].tolist(), "per_channel_normalised": {}}
            for ci, ch in enumerate(cfg["data"]["channels"]):  # mean / std / non-finite count after the scalers
                v = x[0, ci, 0].double()
                m = float(torch.nanmean(v))
                rec["input"]["per_channel_normalised"][ch] = [
                    round(m, 4), round(float(torch.nanmean((v - m) ** 2)) ** 0.5, 4), int((~torch.isfinite(v)).sum())]
                del v
            target = batch["target"]
            rec["target_normalised_by_dataset"] = float(target.reshape(-1)[0])
            rec["target_dtype"] = str(target.dtype)
            tv = float(ds.un_norm_output(rec["target_normalised_by_dataset"]))
            rec["dataset_target_unnorm_km_s"] = tv
            rec["dataset_target_unnorm_equals_csv"] = bool(abs(tv - float(row["V_target_km_s"])) < 1e-6)
            with torch.no_grad():
                out = model(batch)
            t3 = time.time()
            yhat = float(out.reshape(-1)[0])
            rec["output"] = {"shape": list(out.shape), "dtype": str(out.dtype)}
            rec["model_output_normalised"] = yhat
            rec["V_pred_km_s"] = {"train_constants_primary": yhat * tr["ptp"] + tr["min"],
                                  "validation_csv_constants_infer_py_as_shipped": yhat * va["ptp"] + va["min"],
                                  f"{args.split}_csv_constants": float(ds.un_norm_output(yhat))}
            rec["sq_err_norm_vs_dataset_target"] = (yhat - rec["target_normalised_by_dataset"]) ** 2
            try:
                rec["metadata"] = {k: [str(v) for v in np.ravel(metadata[k])] for k in metadata}
            except Exception:
                rec["metadata"] = str(metadata)[:500]
            rec["seconds"] = {"frame_download": dl_s, "dataset_and_load": round(t2 - t1, 1),
                              "forward": round(t3 - t2, 1), "score": round(time.time() - t3, 1),
                              "total": round(time.time() - t_start, 1)}
            rec["status"] = "ok"
            p = rec["V_pred_km_s"]["train_constants_primary"]
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} -> {row['target_time']}: V pred {p:.1f} obs "
                f"{row['V_target_km_s']:.1f} km/s, forward {rec['seconds']['forward']} s")
        except Exception as e:  # record, substitute the block's reserve once, carry on
            rec["status"] = "failed"
            rec["error"] = repr(e)[:2000]
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} FAILED: {e!r}")
            res = blocks.get((int(row["stratum"]), int(row["block"])), {}).get("reserve")
            if (row["role"] == "primary" and res is not None and res["timestamp"] not in done
                    and all(p["timestamp"] != res["timestamp"] for p in pending)):
                pending.append(res)
                if j + 1 == len(pending) - 1 and not args.no_prefetch:  # nothing was prefetched for it yet
                    fut = pool.submit(caches[(j + 1) % 2].ensure, frames_of(res))
        finally:
            ds = loader = batch = x = out = None  # release the input tensors before the next sample
        rec["peak_rss_gb"] = C.peak_rss_gb()
        with open(recs_path, "a") as fp:
            fp.write(json.dumps(rec, default=str) + "\n")
        gc.collect()
        j += 1
        if args.no_prefetch and j < len(pending):
            cache.clear()
            fut = pool.submit(caches[j % 2].ensure, frames_of(pending[j]))
    pool.shutdown(wait=True)
    for c in caches:
        c.clear()
    if args.delete_checkpoint:
        os.remove(ckpt)
    s = summarize(out_dir, args.seed)
    sp = s["splits"].get(args.split, {})
    if "notebook_anchor_check" in sp:
        a = sp["notebook_anchor_check"]
        log(f"notebook anchors: {a['n_run']} run, max |ours - notebook| {a['max_abs_diff_km_s']:.3f} km/s, "
            f"all within {a['tolerance_km_s']} km/s: {a['all_within_tolerance']}")
    if "metrics" in sp:
        m = sp["metrics"]["model"]["train_constants_primary"]
        log(f"{args.split}: n={sp['n_samples_scored']} RMSE {m['rmse']:.2f} {m['ci95']['rmse']} MAE {m['mae']:.2f} "
            f"{m['ci95']['mae']} MSE {m['mse']:.0f} km/s (paper 75.92 / 58.06 / 5698.62)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="select and check the sample; no weights, frames or model")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--n", type=int, default=None, help="samples (default 30 test, 10 validation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "Surya", "solar_wind"))
    ap.add_argument("--max-forward", type=int, default=40, help="forward passes allowed in this run")
    ap.add_argument("--no-channel-check", action="store_true", help="skip the netCDF header check of each frame")
    ap.add_argument("--header-workers", type=int, default=4,
                    help="processes reading netCDF headers during sample selection (each ~0.2 GB of RAM)")
    ap.add_argument("--include-reserves", action="store_true",
                    help="also score every block's reserve (two random pairs per block, about twice the forward "
                         "passes; spread over several runs with --max-forward)")
    ap.add_argument("--no-anchors", action="store_true",
                    help="validation split: leave out the three notebook anchor samples (3 extra forward passes)")
    ap.add_argument("--no-prefetch", action="store_true", help="keep only one sample's frames on disk (~1.2 GB)")
    ap.add_argument("--delete-checkpoint", action="store_true", help="remove the 1.8 GB checkpoint at the end")
    ap.add_argument("--summarize-only", action="store_true", help="recompute summary.json from the jsonl files")
    args = ap.parse_args()
    args.n = args.n or DEFAULT_N[args.split]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)  # config paths are ./assets/...; HelioNetCDFDataset writes logs/data/ relative to the cwd
    label_info = ensure_small_inputs()
    if args.summarize_only:
        summarize(out_dir, args.seed)
        return
    if args.dry_run:
        dry_run(args, out_dir, label_info)
    else:
        run(args, out_dir)


if __name__ == "__main__":
    main()
