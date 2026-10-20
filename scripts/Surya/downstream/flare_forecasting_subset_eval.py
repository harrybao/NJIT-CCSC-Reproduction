"""Solar-flare forecasting with the released fine-tuned Surya model, scored on a seeded, class-stratified sample.

PARTIAL REPRODUCTION. Table 4 of arXiv 2508.14112 v2 (Sec. 2.6.3) reports TSS 0.436, HSS 0.522 and F1 0.561 for Surya
(AlexNet 0.358 / 0.398 / 0.454, ResNet50 0.018 / 0.028 / 0.055) for the binary task "a flare of class M1.0 or above
in [t, t+24 h)" (label_max of nasa-ibm-ai4science/surya-bench-flare-forecasting), with the metrics of Eq. 21-23. The
paper does not say which split, how many samples or which threshold. This script scores the released checkpoint
nasa-ibm-ai4science/solar_flares_surya on a small seeded sample of the SuryaBench test split (2020-2024) or validation
split (Jan 15-31 of 2011-2019) on a CPU. It can say whether the paper's numbers are plausible for this model on that
split; it cannot reproduce the full-split numbers.

What is the authors' code (imported unchanged from the clone, nothing copied or edited):
  * SolarFlareDataset (downstream_examples/solar_flare_forcasting/dataset.py) with the arguments infer.get_dataloader
    uses: inputs at t-60 and t (time_delta_input_minutes [-60, 0]), a target frame at t+60 (time_delta_target_minutes
    +60, rollout_steps 0) that the class requires and loads but the classifier never reads, label = label_max of the
    split CSV, scalers from scalers.yaml, phase "valid", no channel masking, no flips.
  * custom_collate_fn, get_model and apply_peft_lora (finetune.py, as imported by infer.py) with config_infer.yaml,
    then the checkpoint handling of infer.load_model ending in load_state_dict(strict=True).
  * DistributedClassificationMetrics (metrics.py) for the cross-checks of every TSS/HSS/F1 value.
Differences from infer.py (none changes the network, its weights or its inputs): torch.load uses weights_only=True and
mmap=True (the checkpoint's pickle references only OrderedDict, _rebuild_tensor_v2 and FloatStorage; checked in the
dry run from a byte-range read of data.pkl); the forward pass runs in float32 without autocast (on CPU, infer.py's
autocast with float32 is disabled anyway, and CPU bfloat16 breaks Surya: results/Surya/bf16_diagnosis); the
DataLoader uses num_workers=0 and a one-sample dataset per time stamp; the unused target tensor batch["forecast"] is
dropped before the forward pass to save 0.87 GB (HelioSpectformer1D reads only batch["ts"] and
batch["time_delta_input"] with this config).

Output and decision rule. The model has one output (num_outputs=1): a logit for the positive class "flare"
(label_max = 1). infer.py predicts int(sigmoid(logit) > 0.5), i.e. logit > 0. finetune.evaluate_model, the only place
where the repo computes TSS/HSS/F1, passes the raw logits to DistributedClassificationMetrics(threshold=0.5), i.e.
predicts logit > 0.5 (probability > 0.622). Both rules are scored from the same logits; the infer.py rule is primary.

Metrics. TSS = TPR - FPR (Eq. 21) does not depend on class prevalence, so it is estimated directly from a sample with
equal numbers of positives and negatives. HSS (Eq. 22) and F1 (Eq. 23) do depend on prevalence; they are reported
(a) re-weighted to a target prevalence pi from the sample's TPR and FPR, with expected counts per unit
    TP = pi TPR, FN = pi (1 - TPR), FP = (1 - pi) FPR, TN = (1 - pi)(1 - FPR), which gives
    F1  = 2 pi TPR / (pi (1 + TPR) + (1 - pi) FPR)
    HSS = 2 pi (1 - pi)(TPR - FPR) / (pi [pi (1 - TPR) + (1 - pi)(1 - FPR)] + (1 - pi)[pi TPR + (1 - pi) FPR]),
    for pi = the prevalence of the candidate set of the split (primary), of the whole split CSV, and the prevalence
    implied by the paper's own Table 4 triple (Eq. 21-23 solved for prevalence, TPR, FPR);
(b) raw, on the sample as drawn (prevalence about 0.5); these are not comparable with the paper.
Bootstrap: 10,000 resamples, positives and negatives resampled separately (the design fixes their numbers), percentile
95% intervals. Wilson intervals for TPR and FPR and a ROC AUC (threshold-free, not in the paper) are added as context.

Sampling. Candidates are the hourly rows of <split>.csv whose t-60 and t+60 rows are also in that CSV (the authors'
class inner-joins the frame index with the CSV and then cannot load a frame whose row was dropped), whose frames at
t-60, t and t+60 are all in the S3 bucket listing, and, for the validation split, whose three frames have present == 1
in core-sdo's valid_index_surya_1_0.csv (the index finetune.py validates on). Half of the sample is drawn from the
positives and half from the negatives. Inside each class the sample is allocated to years in proportion to that
class's candidates per year (largest remainder, ties broken with the seed), so the sample is self-weighting within
each class; inside a (class, year) stratum the candidates are cut into equal consecutive time blocks, one sample per
block, drawn in a seeded random order. A draw is skipped (and the next one in the block tried) if a HEAD request does
not find one of its three frames on S3, or if the netCDF header of one of them lacks a 4096x4096 channel (read with
HTTP range requests); every skip is recorded with its reason. Each block also gets a reserve, used only if the primary
fails at run time.

Data. Frames come from the public bucket s3://nasa-surya-bench (bitwise-identical to the HF copies, see
results/Surya/s3_vs_hf_frames.json) through surya_common.FrameCache. At most one sample's three frames (about 1.8 GB)
are on disk: once a sample has been read into memory its frames are deleted and the next sample's are downloaded while
the model runs. --alias-target-frame points the unused t+60 index row at the t file (after the HEAD check of t+60) to
save a third of the traffic; it is off by default.

Usage (from the repository root):
  python scripts/Surya/downstream/flare_forecasting_subset_eval.py --dry-run --split test --n 40 --seed 0
  python scripts/Surya/downstream/flare_forecasting_subset_eval.py --split test --n 40 --seed 0
  python scripts/Surya/downstream/flare_forecasting_subset_eval.py --split validation --n 20 --seed 0
  python scripts/Surya/downstream/flare_forecasting_subset_eval.py --summarize-only
A rerun skips time stamps already in per_sample_<split>.jsonl, so an interrupted run resumes; --max-forward caps the
forward passes of one run.
"""
import argparse
import collections
import concurrent.futures as cf
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import pickletools
import re
import struct
import sys
import time
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import surya_common as C  # noqa: E402  (sets HF_HUB_DISABLE_XET=1 before huggingface_hub is imported)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
UPSTREAM = "/home/user/upstream/surya/Surya"
TASK_DIR = os.path.join(UPSTREAM, "downstream_examples", "solar_flare_forcasting")
FM_DIR = "/home/user/work/surya_run/data/Surya-1.0"
WORK = "/home/user/work/surya_ft/flare_forecasting"
LABEL_DIR = os.path.join(WORK, "labels")
META_DIR = os.path.join(WORK, "meta")

HF = "https://huggingface.co"
LABEL_REPO, LABEL_REV = "nasa-ibm-ai4science/surya-bench-flare-forecasting", "bf474bc1e7752bd529e9707650f9310db8693763"
LABEL_BYTES = {"test.csv": 1531173, "validation.csv": 122615}
CORE_SDO_REPO, CORE_SDO_REV = "nasa-ibm-ai4science/core-sdo", "9c10abe52c1f206d1b66715745229e29ef543cc6"
CKPT_REPO, CKPT_REV = "nasa-ibm-ai4science/solar_flares_surya", "ec7c42ab22fbf126c4bfffadb420f6e45a669832"
CKPT_NAME, CKPT_SIZE = "solar_flare_weights.pth", 1794022584
CKPT_SHA256 = "a907da5f994f20bdf76914cccb8bafb4d12e56d04ff400c407e0de98d13d2f69"  # HF LFS oid
FM_CKPT = os.path.join(FM_DIR, "surya.366m.v1.pt")
SCALERS = os.path.join(FM_DIR, "scalers.yaml")

PAPER_TABLE4 = {  # arXiv 2508.14112 v2, Sec. 2.6.3, Table 4 (copied exactly)
    "AlexNet": {"tss": 0.358, "hss": 0.398, "f1": 0.454},
    "ResNet50": {"tss": 0.018, "hss": 0.028, "f1": 0.055},
    "Surya": {"tss": 0.436, "hss": 0.522, "f1": 0.561},
}
SPLIT_CSV = {"test": "test.csv", "validation": "validation.csv"}
STRATA_YEARS = {"test": list(range(2020, 2025)), "validation": list(range(2011, 2020))}
DEFAULT_N = {"test": 40, "validation": 20}
THRESHOLD = 0.5  # metrics.py / finetune.py / infer.py
def sigmoid(z):
    """Logistic function without overflow warnings for large negative logits."""
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(-np.abs(z))
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


RULES = {  # name -> (description, function of the logit)
    "infer_py_sigmoid_gt_0.5": ("infer.py: int(sigmoid(logit) > 0.5), i.e. logit > 0 (primary)",
                                lambda z: (sigmoid(z) > THRESHOLD).astype(int)),
    "finetune_eval_logit_gt_0.5": ("finetune.evaluate_model: DistributedClassificationMetrics(threshold=0.5) fed raw "
                                   "logits, i.e. logit > 0.5 (probability > 0.6225)",
                                   lambda z: (z > THRESHOLD).astype(int)),
}
PRIMARY_RULE = "infer_py_sigmoid_gt_0.5"
N_BOOT = 10000
HOUR = pd.Timedelta(minutes=60)
# Timing assumptions for the estimate (the dry run cannot time the model): one fp32 forward pass of the 4096x4096
# backbone took about 136 s on this machine (results/Surya/validation_test, 271.9 s for two passes); reading and
# normalising one 13-channel frame is taken as 25 s, input statistics as 5 s, building and loading the model as 150 s.
SEC_FORWARD, SEC_LOAD_FRAME, SEC_STATS, SEC_MODEL = 136, 25, 5, 150


# ----------------------------------------------------------------------------------------------- small utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hf_url(repo, rev, path, kind="datasets"):
    return f"{HF}/datasets/{repo}/resolve/{rev}/{path}" if kind == "datasets" else f"{HF}/{repo}/resolve/{rev}/{path}"


def fetch_small(url, dest, max_bytes=50 * 2 ** 20, expect_bytes=None):
    """Download a small file (labels, index) once; refuse anything above max_bytes."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and (expect_bytes is None or
                                                               os.path.getsize(dest) == expect_bytes):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        n = int(r.headers.get("Content-Length", 0))
        if n > max_bytes:
            raise IOError(f"{url} is {n} bytes, above the {max_bytes} limit for small files")
        tmp = dest + ".part"
        with open(tmp, "wb") as fp:
            for chunk in r.iter_content(1 << 20):
                fp.write(chunk)
    if expect_bytes is not None and os.path.getsize(tmp) != expect_bytes:
        raise IOError(f"{url}: {os.path.getsize(tmp)} bytes, expected {expect_bytes}")
    os.replace(tmp, dest)
    return dest


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def stamp(t):
    return pd.Timestamp(t).strftime("%Y%m%d_%H%M")


def retry(fn, tries=4):
    for i in range(tries):
        try:
            return fn()
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def frame_times(t):
    """The three frames the authors' dataset loads for reference time t: inputs t-60, t and the target t+60."""
    t = pd.Timestamp(t)
    return [t - HOUR, t, t + HOUR]


# ------------------------------------------------------------------------------------------- remote file checks
class RangeFile(io.RawIOBase):
    """Read-only file object over HTTP range requests, with a block cache; enough for h5py to read a header."""

    def __init__(self, url, block=1 << 16):
        self.url, self.block, self.pos, self.cache, self.nbytes = url, block, 0, {}, 0
        r = retry(lambda: requests.head(url, timeout=60))
        r.raise_for_status()
        self.size = int(r.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _blk(self, i):
        if i not in self.cache:
            a = i * self.block
            b = min(self.size, a + self.block) - 1

            def get():
                r = requests.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
                r.raise_for_status()
                return r.content
            self.cache[i] = retry(get)
            self.nbytes += len(self.cache[i])
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        if n <= 0:
            return 0
        out = bytearray()
        p = self.pos
        while len(out) < n:
            i = p // self.block
            d = self._blk(i)
            take = d[p - i * self.block: p - i * self.block + n - len(out)]
            out += take
            p += len(take)
        buf[:n] = out
        self.pos += n
        return n


def remote_nc_shapes(key):
    """(process worker) Shapes of the data variables of one S3 netCDF frame, from its HDF5 header only."""
    import h5py
    rf = RangeFile(f"{C.S3}/{key}")
    with h5py.File(io.BufferedReader(rf, 1 << 16), "r") as f:
        shapes = {k: tuple(f[k].shape) for k in f.keys() if isinstance(f[k], h5py.Dataset)}
    return shapes, rf.nbytes


def checkpoint_pickle_check(max_bytes=4 * 2 ** 20):
    """Read only the data.pkl entry at the start of the checkpoint zip (byte range, no tensor data needed) and list
    the globals its pickle references, to confirm that torch.load(weights_only=True) can load it."""
    url = hf_url(CKPT_REPO, CKPT_REV, CKPT_NAME, kind="models")
    n = 1 << 17
    while True:
        r = retry(lambda: requests.get(url, headers={"Range": f"bytes=0-{n - 1}"}, timeout=120))
        r.raise_for_status()
        b = r.content
        sig, = struct.unpack("<I", b[:4])
        nl, el = struct.unpack("<HH", b[26:30])
        name = b[30:30 + nl].decode()
        bio = io.BytesIO(b[30 + nl + el:])
        try:
            g, strings, n_reduce = collections.Counter(), [], 0
            for op, arg, _ in pickletools.genops(bio):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"):
                    strings.append(arg)
                if op.name == "GLOBAL":
                    g[str(arg)] += 1
                if op.name == "STACK_GLOBAL":
                    g[" ".join(strings[-2:])] += 1
                if op.name == "REDUCE":
                    n_reduce += 1
            break
        except ValueError:  # pickle continues past the bytes read
            if n >= max_bytes:
                raise
            n *= 4
    allowed = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage"}
    tensor_keys = sorted({s for s in strings if re.match(r"^base_model\.model\..+\.(weight|bias)$", s)
                          or s in ("base_model.model.cls_token",)})
    return {"zip_signature_ok": sig == 0x04034B50, "first_entry": name, "pickle_bytes": bio.tell(),
            "bytes_read": len(b), "globals": dict(g), "only_tensor_globals": set(g) <= allowed,
            "n_reduce_ops": n_reduce, "n_weight_bias_keys_seen": len(tensor_keys),
            "has_cls_token": "base_model.model.cls_token" in strings,
            "has_unembed": "base_model.model.unembed.weight" in strings,
            "has_lora": any("lora_A" in s for s in strings)}


def bandwidth_probe(s3_key):
    """16 MB range reads: one S3 frame (frame download speed) and the core-sdo train index CSV on HF (CDN speed for
    the checkpoint; no weight bytes are read)."""
    out = {}
    for name, url in (("s3_frame", f"{C.S3}/{s3_key}"),
                      ("hf_cdn_csv", hf_url(CORE_SDO_REPO, CORE_SDO_REV, "train_index_surya_1_0.csv"))):
        t0 = time.time()
        r = retry(lambda: requests.get(url, headers={"Range": "bytes=0-16777215"}, timeout=120))
        r.raise_for_status()
        dt = max(1e-3, time.time() - t0)
        out[name] = {"bytes": len(r.content), "seconds": round(dt, 2), "MB_per_s": round(len(r.content) / 1e6 / dt, 1)}
    return out


# --------------------------------------------------------------------------------------------------- metrics
def add_task_dir():
    if sys.path[0] != TASK_DIR:
        sys.path.insert(0, TASK_DIR)


def repo_metric_class():
    add_task_dir()
    import metrics as M  # the authors' metrics.py
    assert os.path.dirname(os.path.abspath(M.__file__)) == TASK_DIR, M.__file__
    return M.DistributedClassificationMetrics


def counts(pred, y):
    """(tp, tn, fp, fn) of 0/1 arrays; leading axes are kept (pred and y broadcast)."""
    pred, y = np.asarray(pred).astype(bool), np.asarray(y).astype(bool)
    return ((pred & y).sum(-1), (~pred & ~y).sum(-1), (pred & ~y).sum(-1), (~pred & y).sum(-1))


def metrics_from_counts(tp, tn, fp, fn):
    """Eq. 21-23 of arXiv 2508.14112 v2 (Sec. 2.6.3), plus precision, recall and accuracy; float64, vectorised."""
    tp, tn, fp, fn = (np.asarray(x, dtype=np.float64) for x in (tp, tn, fp, fn))
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "tss": tp / (tp + fn) - fp / (fp + tn),                                                 # Eq. 21
            "hss": 2 * (tp * tn - fp * fn) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)),       # Eq. 22
            "f1": 2 * tp / (2 * tp + fp + fn),                                                      # Eq. 23
            "precision": tp / (tp + fp), "recall": tp / (tp + fn),
            "accuracy": (tp + tn) / (tp + tn + fp + fn)}


def expected_counts(tpr, fpr, prevalence, total=1.0):
    """Expected confusion matrix (tp, tn, fp, fn) of a classifier with these TPR/FPR on a set of `total` instances
    with this prevalence."""
    tpr, fpr = np.asarray(tpr, dtype=np.float64), np.asarray(fpr, dtype=np.float64)
    p, n = total * prevalence, total * (1 - prevalence)
    return p * tpr, n * (1 - fpr), n * fpr, p * (1 - tpr)


def reweighted(tpr, fpr, prevalence):
    return metrics_from_counts(*expected_counts(tpr, fpr, prevalence))


def reweighted_closed_form(tpr, fpr, pi):
    """The closed forms quoted in the docstring (used only to check expected_counts + Eq. 22-23)."""
    f1 = 2 * pi * tpr / (pi * (1 + tpr) + (1 - pi) * fpr)
    hss = 2 * pi * (1 - pi) * (tpr - fpr) / (pi * (pi * (1 - tpr) + (1 - pi) * (1 - fpr))
                                             + (1 - pi) * (pi * tpr + (1 - pi) * fpr))
    return {"tss": tpr - fpr, "hss": hss, "f1": f1}


def repo_metrics_on_values(values, y, threshold=THRESHOLD):
    """Exactly what finetune.evaluate_model does: DistributedClassificationMetrics(threshold).update(values, target)
    then compute_and_reset(). values are whatever is passed as `prediction` (logits there, probabilities here when
    the infer.py rule is checked)."""
    import torch
    m = repo_metric_class()(threshold=threshold)
    m.update(torch.as_tensor(np.asarray(values, dtype=np.float32)), torch.as_tensor(np.asarray(y, dtype=np.float32)))
    return {k: float(v) for k, v in m.compute_and_reset().items()}


def repo_metrics_on_counts(tp, tn, fp, fn):
    """The repo's formulas (compute_and_reset) applied to a given (possibly fractional) confusion matrix."""
    import torch
    m = repo_metric_class()(threshold=THRESHOLD)
    m._counts_tp_tn_fp_fn = torch.tensor([tp, tn, fp, fn], dtype=torch.float64)
    return {k: float(v) for k, v in m.compute_and_reset().items()}


def auc_mann_whitney(score_pos, score_neg):
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie); last axis is the sample axis, leading axes are kept."""
    sp, sn = np.asarray(score_pos, float)[..., :, None], np.asarray(score_neg, float)[..., None, :]
    return (sp > sn).mean((-2, -1)) + 0.5 * (sp == sn).mean((-2, -1))


def wilson(k, n, z=1.959964):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def implied_confusion(tss, hss, f1):
    """Solve Eq. 21-23 for (prevalence pi, TPR, FPR). From Eq. 21 and 23, TPR(pi) = F1 [(1 - pi) TSS - pi] / (F1 - 2 pi)
    and FPR = TPR - TSS; Eq. 22 is then a function of pi alone, solved on a fine grid and refined by bisection."""
    def at(pi):
        r = f1 * ((1 - pi) * tss - pi) / (f1 - 2 * pi)
        return r, r - tss

    def h(pi):
        r, f = at(pi)
        return reweighted(r, f, pi)["hss"] - hss

    grid = np.linspace(1e-5, f1 / 2 - 1e-5, 200001)
    r, f = at(grid)
    ok = (r >= 0) & (r <= 1) & (f >= 0) & (f <= 1)
    vals = np.where(ok, reweighted(r, f, grid)["hss"] - hss, np.nan)
    sol = []
    for i in np.flatnonzero(np.isfinite(vals[:-1]) & np.isfinite(vals[1:]) & (np.sign(vals[:-1]) != np.sign(vals[1:]))):
        a, b = grid[i], grid[i + 1]
        for _ in range(80):
            m = 0.5 * (a + b)
            if np.sign(h(m)) == np.sign(h(a)):
                a = m
            else:
                b = m
        pi = 0.5 * (a + b)
        rr, ff = at(pi)
        sol.append({"prevalence": float(pi), "tpr": float(rr), "fpr": float(ff)})
    return sol


def paper_implied():
    """Prevalence/TPR/FPR implied by Table 4 (Surya), and their range over the rounding box (each value +-0.0005)."""
    p = PAPER_TABLE4["Surya"]
    sol = implied_confusion(p["tss"], p["hss"], p["f1"])
    corners = []
    for a in (-5e-4, 5e-4):
        for b in (-5e-4, 5e-4):
            for c in (-5e-4, 5e-4):
                corners += implied_confusion(p["tss"] + a, p["hss"] + b, p["f1"] + c)
    rng = {k: [min(s[k] for s in corners), max(s[k] for s in corners)] for k in ("prevalence", "tpr", "fpr")}
    return {"solutions": sol, "unique": len(sol) == 1, "range_over_rounding_box": rng,
            "note": "Eq. 21-23 fix the confusion matrix up to scale; the Table 4 triple has a unique solution."}


def bootstrap_eval(y, logit, prevalences, n_boot=N_BOOT, seed=0):
    """All metrics with class-stratified bootstrap intervals (positives and negatives resampled separately)."""
    y, logit = np.asarray(y, int), np.asarray(logit, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    ip = pos[rng.integers(0, len(pos), size=(n_boot, len(pos)))]
    ineg = neg[rng.integers(0, len(neg), size=(n_boot, len(neg)))]

    def ci(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) else [None, None]

    def entry(est, boot, ref=None):
        e = {"estimate": float(est), "ci95": ci(boot), "bootstrap_se": float(np.nanstd(boot))}
        if ref is not None:
            e["paper_surya"] = ref
            e["paper_inside_ci"] = None if e["ci95"][0] is None else bool(e["ci95"][0] <= ref <= e["ci95"][1])
            e["estimate_minus_paper"] = float(est) - ref
        return e

    ref = PAPER_TABLE4["Surya"]
    out = {"n_pos": int(len(pos)), "n_neg": int(len(neg)), "rules": {}}
    for name, (desc, rule) in RULES.items():
        pred = rule(logit)
        tp, fn = int(pred[pos].sum()), int(len(pos) - pred[pos].sum())
        fp, tn = int(pred[neg].sum()), int(len(neg) - pred[neg].sum())
        tpr, fpr = tp / len(pos), fp / len(neg)
        tpr_b, fpr_b = pred[ip].mean(1), pred[ineg].mean(1)
        r = {"description": desc, "sample_counts": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
             "tpr": {**entry(tpr, tpr_b), "ci95_wilson": wilson(tp, len(pos))},
             "fpr": {**entry(fpr, fpr_b), "ci95_wilson": wilson(fp, len(neg))},
             "tss": entry(tpr - fpr, tpr_b - fpr_b, ref["tss"])}
        r["reweighted"] = {}
        for pname, pi in prevalences.items():
            est, boot = reweighted(tpr, fpr, pi), reweighted(tpr_b, fpr_b, pi)
            ec = expected_counts(tpr, fpr, pi)
            r["reweighted"][pname] = {
                "prevalence": float(pi),
                "hss": entry(est["hss"], boot["hss"], ref["hss"]), "f1": entry(est["f1"], boot["f1"], ref["f1"]),
                "precision": entry(est["precision"], boot["precision"]),
                "expected_counts_per_unit": dict(zip(("tp", "tn", "fp", "fn"), (float(x) for x in ec))),
                "repo_formulas_on_expected_counts": {k: v for k, v in repo_metrics_on_counts(*ec).items()
                                                     if k in ("tss", "hss", "f1", "precision", "recall")}}
        raw, raw_b = metrics_from_counts(tp, tn, fp, fn), metrics_from_counts(
            pred[ip].sum(1), (1 - pred[ineg]).sum(1), pred[ineg].sum(1), (1 - pred[ip]).sum(1))
        r["raw_sample"] = {"note": f"as drawn: prevalence {len(pos) / len(y):.3f}; HSS and F1 here are NOT comparable "
                                   "with the paper", **{k: entry(raw[k], raw_b[k]) for k in raw}}
        # the authors' metric class on the actual per-sample values, exactly as finetune.evaluate_model calls it
        values = logit if name == "finetune_eval_logit_gt_0.5" else sigmoid(logit)
        repo = repo_metrics_on_values(values, y)
        r["raw_sample"]["repo_DistributedClassificationMetrics"] = {
            "fed": "raw logits" if name == "finetune_eval_logit_gt_0.5" else "sigmoid probabilities",
            **{k: repo[k] for k in ("tss", "hss", "f1", "tp", "tn", "fp", "fn")},
            "max_abs_diff_vs_ours": float(max(abs(repo[k] - raw[k]) for k in ("tss", "hss", "f1")))}
        out["rules"][name] = r
    out["roc_auc_context_not_in_paper"] = entry(auc_mann_whitney(logit[pos], logit[neg]),
                                                auc_mann_whitney(logit[ip], logit[ineg]))
    return out


def metric_self_test(out_dir):
    """Checks every metric path against the authors' DistributedClassificationMetrics on synthetic predictions, a
    hand-worked example, the re-weighting identity, the implied-prevalence solver and the interval coverage."""
    import torch
    rep, worst = {"cases": {}}, 0.0
    rng = np.random.default_rng(12345)
    keys = ("tss", "hss", "f1", "precision", "recall", "accuracy")
    # 1. synthetic predictions: our Eq. 21-23 vs metrics.py, fed probabilities and fed raw logits
    for n, prev, sep in [(40, 0.5, 1.0), (40, 0.5, 0.0), (1000, 0.1, 1.5), (5000, 0.2943, 0.8), (37, 0.2, 3.0),
                         (3243, 0.108, 1.2)]:
        y = (rng.random(n) < prev).astype(int)
        y[:2] = [0, 1]
        z = rng.normal(0, 1, n) + sep * (2 * y - 1)
        for fed, vals in (("sigmoid", sigmoid(z)), ("logit", z)):
            repo = repo_metrics_on_values(vals, y)
            mine = metrics_from_counts(*counts((vals > THRESHOLD).astype(int), y))
            loop = dict(tp=0, tn=0, fp=0, fn=0)  # independent element-by-element count
            for v, t in zip(vals, y):
                loop[("t" if (v > THRESHOLD) == bool(t) else "f") + ("p" if v > THRESHOLD else "n")] += 1
            d = max(abs(repo[k] - float(mine[k])) for k in keys)
            d = max(d, *(abs(repo[k] - loop[k]) for k in ("tp", "tn", "fp", "fn")))
            worst = max(worst, d)
            rep["cases"][f"n{n}_prev{prev}_sep{sep}_fed_{fed}"] = {
                "ours": {k: float(mine[k]) for k in ("tss", "hss", "f1")},
                "repo_metrics_py": {k: repo[k] for k in ("tss", "hss", "f1")}, "max_abs_diff": d}
        # decision rules: infer.py's int(F.sigmoid(x).item() > 0.5) equals logit > 0 and our rule function
        zz = torch.as_tensor(z, dtype=torch.float32)
        infer_pred = np.array([int(torch.sigmoid(v).item() > 0.5) for v in zz])
        assert (infer_pred == RULES["infer_py_sigmoid_gt_0.5"][1](z)).all() and (infer_pred == (z > 0)).all()
        assert (RULES["finetune_eval_logit_gt_0.5"][1](z) == (sigmoid(z) > sigmoid(0.5))).all()
    rep["logit_0.5_as_probability"] = float(sigmoid(0.5))
    # 2. hand-worked example: TP 3, FN 1, FP 2, TN 4 -> TSS 3/4 - 2/6, HSS 2(12-2)/(4*5 + 5*6) = 0.4, F1 6/9
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.7, 0.1, 0.2, 0.3, 0.4])
    expect = {"tss": 0.75 - 2 / 6, "hss": 0.4, "f1": 6 / 9}
    repo, mine = repo_metrics_on_values(p, y), metrics_from_counts(*counts((p > 0.5).astype(int), y))
    d = max(max(abs(repo[k] - v), abs(float(mine[k]) - v)) for k, v in expect.items())
    worst = max(worst, d)
    rep["hand_example"] = {"expected": expect, "repo": {k: repo[k] for k in expect},
                           "ours": {k: float(mine[k]) for k in expect}, "max_abs_diff": d}
    # 3. re-weighting identity: TPR/FPR of a population + its prevalence give back its own HSS/F1 exactly
    N, pi = 43848, 0.2943
    y = (rng.random(N) < pi).astype(int)
    z = rng.normal(0, 1, N) + 1.0 * (2 * y - 1) - 0.7
    pred = (z > 0).astype(int)
    pop = repo_metrics_on_values(sigmoid(z), y)
    tpr, fpr, pi_pop = pred[y == 1].mean(), pred[y == 0].mean(), y.mean()
    rw, cf_ = reweighted(tpr, fpr, pi_pop), reweighted_closed_form(tpr, fpr, pi_pop)
    rc = repo_metrics_on_counts(*expected_counts(tpr, fpr, pi_pop, total=N))
    d = max(max(abs(pop[k] - float(rw[k])), abs(float(cf_[k]) - float(rw[k])), abs(rc[k] - float(rw[k])))
            for k in ("tss", "hss", "f1"))
    worst = max(worst, d)
    rep["reweighting_identity"] = {"population_repo": {k: pop[k] for k in ("tss", "hss", "f1")},
                                   "reweighted_from_tpr_fpr": {k: float(rw[k]) for k in ("tss", "hss", "f1")},
                                   "closed_form": {k: float(cf_[k]) for k in ("tss", "hss", "f1")},
                                   "repo_formulas_on_expected_counts": {k: rc[k] for k in ("tss", "hss", "f1")},
                                   "max_abs_diff": d}
    # 4. implied-prevalence solver recovers a known confusion matrix
    known = (0.2, 0.6, 0.1)
    m = reweighted(known[1], known[2], known[0])
    sol = implied_confusion(float(m["tss"]), float(m["hss"]), float(m["f1"]))
    d = max(abs(sol[0]["prevalence"] - known[0]), abs(sol[0]["tpr"] - known[1]), abs(sol[0]["fpr"] - known[2])) \
        if len(sol) == 1 else 1.0
    worst = max(worst, d)
    rep["implied_solver"] = {"known_pi_tpr_fpr": known, "solved": sol, "max_abs_diff": d}
    # 5. AUC against a brute-force pair count
    sp, sn = rng.normal(1, 1, 20), rng.normal(0, 1, 20)
    brute = np.mean([1.0 if a > b else (0.5 if a == b else 0.0) for a in sp for b in sn])
    d = abs(float(auc_mann_whitney(sp, sn)) - brute)
    worst = max(worst, d)
    rep["auc_check_max_abs_diff"] = d
    # 6. coverage of the class-stratified percentile bootstrap for 20 + 20 samples, by simulation (binomial
    #    resampling of the counts is equivalent to resampling individuals for these count-based metrics)
    cov = {}
    for label, (tpr0, fpr0, pi0) in {"paper_implied_like": (0.456, 0.020, 0.106), "moderate": (0.6, 0.15, 0.2943),
                                     "weak": (0.3, 0.1, 0.2943)}.items():
        reps, B, k = 1000, 2000, 20
        truth = reweighted(tpr0, fpr0, pi0)
        tp = rng.binomial(k, tpr0, reps)
        fp = rng.binomial(k, fpr0, reps)
        tpb = rng.binomial(k, (tp / k)[:, None], (reps, B)) / k
        fpb = rng.binomial(k, (fp / k)[:, None], (reps, B)) / k
        est, boot = reweighted(tp / k, fp / k, pi0), reweighted(tpb, fpb, pi0)
        c = {}
        for key in ("tss", "hss", "f1"):
            lo, hi = np.nanpercentile(boot[key], 2.5, axis=1), np.nanpercentile(boot[key], 97.5, axis=1)
            c[key] = {"coverage": float(np.mean((lo <= truth[key]) & (truth[key] <= hi))),
                      "mean_estimate": float(np.nanmean(est[key])), "truth": float(truth[key]),
                      "mean_ci_width": float(np.nanmean(hi - lo))}
        cov[label] = {"tpr_fpr_pi": [tpr0, fpr0, pi0], **c}
    rep["bootstrap_coverage_simulation_20_plus_20"] = cov
    rep["max_abs_diff_overall"] = worst
    rep["passed"] = bool(worst < 1e-6)
    json.dump(rep, open(os.path.join(out_dir, "metric_check.json"), "w"), indent=1)
    if not rep["passed"]:
        raise AssertionError(f"metric self-test failed, max abs diff {worst}")
    return rep


# -------------------------------------------------------------------------------------------- data and labels
def ensure_small_inputs(split):
    fetch_small(hf_url(LABEL_REPO, LABEL_REV, SPLIT_CSV[split]), os.path.join(LABEL_DIR, SPLIT_CSV[split]),
                expect_bytes=LABEL_BYTES[SPLIT_CSV[split]])
    if split == "validation":
        fetch_small(hf_url(CORE_SDO_REPO, CORE_SDO_REV, "valid_index_surya_1_0.csv"),
                    os.path.join(LABEL_DIR, "valid_index_surya_1_0.csv"), expect_bytes=961993)


def list_month(year, month):
    """On-the-hour frames of one month in the S3 bucket (ListObjectsV2, anonymous), cached as a small CSV."""
    dest = os.path.join(META_DIR, "s3_listing", f"{year:04d}_{month:02d}.csv")
    if os.path.exists(dest):
        return pd.read_csv(dest), 0
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    rows, token, nbytes = [], None, 0
    while True:
        params = {"list-type": "2", "prefix": f"{year:04d}/{month:02d}/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token

        def get():
            r = requests.get(C.S3 + "/", params=params, timeout=60)
            r.raise_for_status()
            return r.content
        body = retry(get)
        nbytes += len(body)
        root = ET.fromstring(body)
        for c in root.findall("s:Contents", ns):
            k = c.findtext("s:Key", namespaces=ns)
            if re.fullmatch(r"\d{4}/\d{2}/\d{8}_\d{2}00\.nc", k):
                rows.append((k, int(c.findtext("s:Size", namespaces=ns))))
        if root.findtext("s:IsTruncated", namespaces=ns) == "true":
            token = root.findtext("s:NextContinuationToken", namespaces=ns)
        else:
            break
    df = pd.DataFrame(rows, columns=["key", "bytes"])
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.to_csv(dest + ".part", index=False)
    os.replace(dest + ".part", dest)
    return df, nbytes


def s3_listing(months):
    t0 = time.time()
    with cf.ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda ym: list_month(*ym), sorted(months)))
    df = pd.concat([r[0] for r in res], ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["key"].str.slice(8, 21), format="%Y%m%d_%H%M")
    return df, {"months": len(months), "on_the_hour_frames": int(len(df)),
                "listing_bytes_transferred_this_call": int(sum(r[1] for r in res)),
                "seconds": round(time.time() - t0, 1)}


def candidates(split, use_listing=True):
    lab = pd.read_csv(os.path.join(LABEL_DIR, SPLIT_CSV[split]))
    lab["timestamp"] = pd.to_datetime(lab["timestamp"])
    lab = lab.sort_values("timestamp").reset_index(drop=True)
    t = lab["timestamp"].values
    info = {"rows_in_csv": int(len(lab)), "positives_in_csv": int(lab["label_max"].sum()),
            "prevalence_csv": float(lab["label_max"].mean()), "filters": []}
    keep = np.isin(t - np.timedelta64(60, "m"), t) & np.isin(t + np.timedelta64(60, "m"), t)
    info["filters"].append({"filter": "t-60 and t+60 rows also in the split CSV", "rows_left": int(keep.sum())})
    if split == "validation":
        idx = pd.read_csv(os.path.join(LABEL_DIR, "valid_index_surya_1_0.csv"))
        have = pd.to_datetime(idx.loc[idx["present"] == 1, "timestep"]).values
        k2 = np.isin(t - np.timedelta64(60, "m"), have) & np.isin(t, have) & np.isin(t + np.timedelta64(60, "m"), have)
        keep &= k2
        info["filters"].append({"filter": "t-60, t, t+60 present == 1 in core-sdo valid_index_surya_1_0.csv",
                                "rows_left": int(keep.sum())})
    listing = None
    if use_listing:
        need = pd.DatetimeIndex(np.concatenate([t[keep] + np.timedelta64(d, "m") for d in (-60, 0, 60)]))
        months = {(int(y), int(m)) for y, m in zip(need.year, need.month)}
        listing, linfo = s3_listing(months)
        on_s3 = listing["timestamp"].values
        k3 = np.isin(t - np.timedelta64(60, "m"), on_s3) & np.isin(t, on_s3) & np.isin(t + np.timedelta64(60, "m"), on_s3)
        keep &= k3
        info["s3_listing"] = linfo
        info["filters"].append({"filter": "t-60, t, t+60 frames in the S3 bucket listing", "rows_left": int(keep.sum())})
    cands = lab[keep].reset_index(drop=True)
    info["candidates"] = int(len(cands))
    info["candidate_positives"] = int(cands["label_max"].sum())
    info["prevalence_candidates"] = float(cands["label_max"].mean())
    g = cands.groupby([cands["timestamp"].dt.year, "label_max"]).size().unstack(fill_value=0)
    info["candidates_per_year"] = {int(y): {"neg": int(r.get(0, 0)), "pos": int(r.get(1, 0))} for y, r in g.iterrows()}
    sizes = None
    if listing is not None:
        sizes = dict(zip(listing["timestamp"], listing["bytes"]))
    return cands, info, sizes


def largest_remainder(counts_by_year, k, rng):
    years = sorted(counts_by_year)
    c = np.array([counts_by_year[y] for y in years], float)
    if k == 0 or c.sum() == 0:
        return {y: 0 for y in years}
    q = k * c / c.sum()
    base = np.floor(q).astype(int)
    frac = q - base
    tie = rng.random(len(years))  # seeded tie-break
    order = sorted(range(len(years)), key=lambda i: (-frac[i], tie[i]))
    for i in order[: k - base.sum()]:
        base[i] += 1
    return {y: int(min(b, counts_by_year[y])) for y, b in zip(years, base)}


def plan_blocks(cands, split, n, seed):
    """Class-stratified (half positives), year-proportional allocation, one draw order per consecutive time block."""
    rng = np.random.default_rng(seed)
    n_pos = n // 2
    alloc, blocks = {}, []
    for cls, k in ((1, n_pos), (0, n - n_pos)):
        sub_c = cands[cands["label_max"] == cls]
        cnt = {y: int((sub_c["timestamp"].dt.year == y).sum()) for y in STRATA_YEARS[split]}
        cnt = {y: v for y, v in cnt.items() if v > 0}
        a = largest_remainder(cnt, k, rng)
        alloc["pos" if cls else "neg"] = a
        for y in sorted(a):
            if a[y] == 0:
                continue
            sub = sub_c[sub_c["timestamp"].dt.year == y].reset_index(drop=True)
            for b, part in enumerate(np.array_split(np.arange(len(sub)), a[y])):
                order = rng.permutation(part)
                blocks.append({"cls": cls, "stratum": f"{'pos' if cls else 'neg'}_{y}", "year": y, "block": b,
                               "block_first": str(sub["timestamp"][part[0]]),
                               "block_last": str(sub["timestamp"][part[-1]]), "block_candidates": int(len(part)),
                               "order": [sub.iloc[i] for i in order]})
    return alloc, blocks


def head_check(t):
    sizes = []
    for name, ft in zip(("t-60", "t", "t+60"), frame_times(t)):
        ok, nbytes = retry(lambda: C.s3_exists(ft))
        if not ok:
            return False, f"frame at {name} ({C.s3_key(ft)}) not on S3 (HEAD)", sizes
        sizes.append(nbytes)
    return True, "", sizes


def select_sample(split, n, seed, use_listing=True, header_check=True, want_per_block=2, max_tries=30):
    cands, info, listed_sizes = candidates(split, use_listing)
    alloc, blocks = plan_blocks(cands, split, n, seed)
    state = [{"pos": 0, "found": [], "skipped": []} for _ in blocks]
    ctx = mp.get_context("spawn")
    rounds, header_bytes = 0, 0
    with cf.ThreadPoolExecutor(8) as tp, cf.ProcessPoolExecutor(4, mp_context=ctx) as pp:
        while True:
            batch = []
            for bi, (blk, st) in enumerate(zip(blocks, state)):
                need = want_per_block - len(st["found"])
                while need > 0 and st["pos"] < min(len(blk["order"]), max_tries):
                    batch.append((bi, blk["order"][st["pos"]]))
                    st["pos"] += 1
                    need -= 1
            if not batch:
                break
            rounds += 1
            heads = list(tp.map(lambda x: head_check(x[1]["timestamp"]), batch))
            shapes = {}
            if header_check:
                futs = {(j, q): pp.submit(remote_nc_shapes, C.s3_key(ft))
                        for j, h in enumerate(heads) if h[0] for q, ft in enumerate(frame_times(batch[j][1]["timestamp"]))}
                for key, f in futs.items():
                    try:
                        shapes[key] = f.result()
                    except Exception as e:
                        shapes[key] = e
            for j, (bi, row) in enumerate(batch):  # assignment in permutation order -> deterministic
                ok, why, sizes = heads[j]
                t = pd.Timestamp(row["timestamp"])
                rec = {"timestamp": str(t), "label_max": int(row["label_max"]), "max_goes_class": row["max_goes_class"],
                       "label_cum": int(row["label_cum"]), "cumulative_index": float(row["cumulative_index"])}
                if ok and listed_sizes is not None:
                    lst = [listed_sizes.get(ft) for ft in frame_times(t)]
                    if lst != sizes:
                        ok, why = False, f"HEAD sizes {sizes} differ from the listing {lst}"
                if ok and header_check:
                    for q, name in enumerate(("t-60", "t", "t+60")):
                        r = shapes[(j, q)]
                        if isinstance(r, Exception):
                            ok, why = False, f"could not read the netCDF header of the {name} frame: {r!r}"
                            break
                        header_bytes += r[1]
                        bad = [c for c in C.CHANNELS if r[0].get(c) != (4096, 4096)]
                        if bad:
                            ok, why = False, f"{name} frame lacks 4096x4096 channels {bad}"
                            break
                if ok and len(state[bi]["found"]) < want_per_block:
                    state[bi]["found"].append({**rec, "frame_bytes": sizes})
                elif not ok:
                    state[bi]["skipped"].append({**rec, "stratum": blocks[bi]["stratum"], "reason": why})
    rows, skipped = [], []
    for blk, st in zip(blocks, state):
        skipped += st["skipped"]
        for role, f in zip(("primary", "reserve"), st["found"]):
            ft = frame_times(f["timestamp"])
            rows.append({"split": split, "role": role, "class": "pos" if blk["cls"] else "neg", "year": blk["year"],
                         "stratum": blk["stratum"], "block": blk["block"], "block_first": blk["block_first"],
                         "block_last": blk["block_last"], "block_candidates": blk["block_candidates"],
                         "timestamp": f["timestamp"], "label_max": f["label_max"],
                         "max_goes_class": f["max_goes_class"], "label_cum": f["label_cum"],
                         "cumulative_index": f["cumulative_index"],
                         "frame_tm60_key": C.s3_key(ft[0]), "frame_t_key": C.s3_key(ft[1]),
                         "frame_tp60_key": C.s3_key(ft[2]), "frame_tm60_bytes": f["frame_bytes"][0],
                         "frame_t_bytes": f["frame_bytes"][1], "frame_tp60_bytes": f["frame_bytes"][2],
                         "headers_checked": bool(header_check)})
    df = pd.DataFrame(rows)
    meta = {"split": split, "n": n, "seed": seed, "use_listing": bool(use_listing), "header_check": bool(header_check),
            "design": "class-stratified (n//2 positives), year-proportional within class, one draw per time block",
            "allocation_per_class_and_year": alloc, "candidate_info": info, "n_blocks": len(blocks),
            "blocks_without_primary": [b["stratum"] + f"#{b['block']}" for b, s in zip(blocks, state) if not s["found"]],
            "blocks_without_reserve": int(sum(len(s["found"]) < 2 for s in state)),
            "skipped": skipped, "n_skipped": len(skipped), "selection_rounds": rounds,
            "header_bytes_read": int(header_bytes)}
    return df, meta


# ------------------------------------------------------------------------------------- authors' dataset/model
def load_configs():
    """config_infer.yaml (infer.py's default; dataset and model arguments) and config.yaml (finetune.py) for a diff.
    Paths that config_infer.yaml gives relative to the task folder are pointed at the files already on disk."""
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config_infer.yaml")))
    cfg_ft = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))
    diff = {}
    for k in sorted(set(cfg["model"]) | set(cfg_ft["model"])):
        if cfg["model"].get(k) != cfg_ft["model"].get(k):
            diff[f"model.{k}"] = (cfg["model"].get(k), cfg_ft["model"].get(k))
    for k in ("time_delta_input_minutes", "time_delta_target_minutes", "channels", "pooling", "n_input_timestamps"):
        if cfg["data"].get(k) != cfg_ft["data"].get(k):
            diff[f"data.{k}"] = (cfg["data"].get(k), cfg_ft["data"].get(k))
    for k in ("rollout_steps", "num_mask_aia_channels", "drop_hmi_probability", "use_latitude_in_learned_flow", "dtype"):
        if cfg.get(k) != cfg_ft.get(k):
            diff[k] = (cfg.get(k), cfg_ft.get(k))
    overrides = {"pretrained_path": (cfg["pretrained_path"], FM_CKPT),
                 "data.scalers_path": (cfg["data"]["scalers_path"], SCALERS)}
    cfg["pretrained_path"] = FM_CKPT
    cfg["data"]["scalers_path"] = SCALERS
    return cfg, {"config_infer_vs_config_yaml": diff, "path_overrides": overrides}


def write_sample_index(t, path, alias_target=False):
    """Per-sample frame index (core-sdo format) with the three frames the dataset needs. With alias_target the t+60
    row points at the t file (the classifier never reads the target frame)."""
    C.write_index(frame_times(t), path)
    if alias_target:
        df = pd.read_csv(path, index_col=0)
        df.loc[df["timestep"] == (pd.Timestamp(t) + HOUR).strftime("%Y-%m-%d %H:%M:%S"), "path"] = C.s3_key(t)
        df.to_csv(path)
    return path


def build_flare_dataset(cfg, scalers, split, t, sdo_root, SolarFlareDataset, alias_target=False):
    """The authors' SolarFlareDataset for one reference time, with exactly the arguments of infer.get_dataloader."""
    index_path = os.path.join(WORK, "index", f"{split}_{stamp(t)}.csv")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    write_sample_index(t, index_path, alias_target)
    return SolarFlareDataset(
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
        flare_index_path=os.path.join(LABEL_DIR, SPLIT_CSV[split]),
        pooling=cfg["data"]["pooling"],
        random_vert_flip=False,
    )


def check_joins(sample, cfg, scalers, split, alias_target=False):
    """Build the authors' dataset for every sampled time (no frame is read) and confirm that it yields exactly that
    time, keeps all three frame rows after its inner join with the labels, and attaches the CSV's label_max."""
    add_task_dir()
    from dataset import SolarFlareDataset
    out = []
    for r in sample.itertuples():
        t = pd.Timestamp(r.timestamp)
        ds = build_flare_dataset(cfg, scalers, split, t, os.path.join(WORK, "frames"), SolarFlareDataset, alias_target)
        rows_ok = all(np.datetime64(ft) in ds.index.index for ft in frame_times(t))
        ok = (len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == t and rows_ok
              and int(ds.index.loc[t, "label_max"]) == int(r.label_max))
        out.append({"timestamp": str(t), "role": r.role, "dataset_len": len(ds), "frame_rows_after_join": int(rows_ok),
                    "label_from_dataset_index": int(ds.index.loc[t, "label_max"]) if t in ds.index.index else None,
                    "label_csv": int(r.label_max), "paths": list(ds.index["path"]), "ok": bool(ok)})
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


def load_model(cfg, ckpt_path):
    """infer.load_model step by step (get_model -> apply_peft_lora -> load_state_dict(strict=True) -> eval), with a
    weights-only unpickler and memory-mapped weights."""
    import torch
    add_task_dir()
    import infer as I  # authors' module; get_model / apply_peft_lora come from finetune.py
    from surya.utils.distributed import set_global_seed
    set_global_seed(42)  # as infer.main
    model = I.get_model(cfg, wandb_logger=None)  # also loads the foundation weights from cfg["pretrained_path"]
    if cfg["model"]["use_lora"]:
        model = I.apply_peft_lora(model, cfg)
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
                   "state_dict_keys": int(n_keys), "load_state_dict_strict": True,
                   "model_class": type(model).__name__, "training_flag": bool(model.training)}


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
                recs[r["timestamp"]] = r  # the last record of a time stamp wins
    return recs


def split_prevalences(split, out_dir):
    meta_json = os.path.join(out_dir, f"sampling_{split}.json")
    meta = json.load(open(meta_json)) if os.path.exists(meta_json) else {}
    ci = meta.get("candidate_info", {})
    pi_paper = paper_implied()["solutions"][0]["prevalence"]
    prev = {}
    if "prevalence_candidates" in ci:
        prev["candidate_set_of_split (primary)"] = ci["prevalence_candidates"]
    if "prevalence_csv" in ci:
        prev["whole_split_csv"] = ci["prevalence_csv"]
    prev["paper_table4_implied"] = pi_paper
    return prev, meta


def summarize(out_dir, seed=0, n_boot=N_BOOT):
    summary = {
        "label": "PARTIAL REPRODUCTION: released fine-tuned weights (solar_flares_surya) scored on a small, seeded, "
                 "class-stratified sample of the SuryaBench flare split named below, on CPU in float32. Not the "
                 "paper's (unstated) evaluation set.",
        "paper_reference": {"source": "arXiv 2508.14112 v2, Sec. 2.6.3, Table 4; metrics Eq. 21-23",
                            "values": PAPER_TABLE4, "split_in_paper": "not stated",
                            "implied_by_table4": paper_implied()},
        "primary_rule": PRIMARY_RULE, "rules": {k: v[0] for k, v in RULES.items()},
        "reweighting_formula": "TP = pi*TPR, FN = pi*(1-TPR), FP = (1-pi)*FPR, TN = (1-pi)*(1-FPR); HSS and F1 = "
                               "Eq. 22 and 23 of these counts. Closed forms: F1 = 2*pi*TPR / (pi*(1+TPR) + "
                               "(1-pi)*FPR); HSS = 2*pi*(1-pi)*(TPR-FPR) / (pi*(pi*(1-TPR) + (1-pi)*(1-FPR)) + "
                               "(1-pi)*(pi*TPR + (1-pi)*FPR)). TSS = TPR - FPR needs no re-weighting.",
        "bootstrap": {"resamples": n_boot, "seed": seed, "scheme": "positives and negatives resampled separately",
                      "interval": "percentile 95%"},
        "splits": {}}
    for split in ("test", "validation"):
        allrec = read_records(per_sample_path(out_dir, split))
        recs = [r for r in allrec.values() if r.get("status") == "ok"]
        failed = [r for r in allrec.values() if r.get("status") != "ok"]
        if not recs:
            continue
        by_block = {}  # one sample per time block: the primary if it was scored, else the reserve
        for r in recs:
            k = (r["stratum"], int(r["block"]))
            if k not in by_block or r["role"] == "primary":
                by_block[k] = r
        recs = sorted(by_block.values(), key=lambda r: r["timestamp"])
        y = np.array([int(r["label_max"]) for r in recs])
        z = np.array([float(r["logit"]) for r in recs])
        assert all(int(r["label_from_dataset"]) == int(r["label_max"]) for r in recs)
        prev, meta = split_prevalences(split, out_dir)
        sp = {"n_scored": len(recs), "n_failed_records": len(failed),
              "roles": dict(collections.Counter(r["role"] for r in recs)),
              "time_range": [recs[0]["timestamp"], recs[-1]["timestamp"]],
              "prevalences_used_for_reweighting": prev,
              "sample_design": {k: meta.get(k) for k in ("n", "seed", "design", "allocation_per_class_and_year",
                                                         "use_listing", "header_check")}}
        if y.min() == y.max():
            sp["note"] = "only one class scored so far; metrics need both"
        else:
            sp["metrics"] = bootstrap_eval(y, z, prev, n_boot=n_boot, seed=seed)
        years = np.array([int(r["year"]) for r in recs])
        sp["per_year"] = {}
        for yr in np.unique(years):
            m = years == yr
            pr = RULES[PRIMARY_RULE][1](z[m])
            sp["per_year"][int(yr)] = {"n_pos": int(y[m].sum()), "n_neg": int((1 - y[m]).sum()),
                                       "hits_pos": int(pr[y[m] == 1].sum()), "false_alarms_neg": int(pr[y[m] == 0].sum())}
        sp["timing"] = {"forward_s_mean": float(np.mean([r["seconds"]["forward"] for r in recs])),
                        "forward_s_max": float(np.max([r["seconds"]["forward"] for r in recs])),
                        "load_s_mean": float(np.mean([r["seconds"]["dataset_and_load"] for r in recs])),
                        "sample_s_mean": float(np.mean([r["seconds"]["total"] for r in recs])),
                        "peak_rss_gb_max": float(np.max([r["peak_rss_gb"] for r in recs]))}
        sp["per_sample"] = [{"timestamp": r["timestamp"], "year": r["year"], "class": r["class"], "role": r["role"],
                             "label_max": r["label_max"], "max_goes_class": r["max_goes_class"], "logit": r["logit"],
                             "prob": r["prob"], **{f"pred_{k}": int(f(np.array([r["logit"]]))[0])
                                                   for k, (_, f) in RULES.items()}} for r in recs]
        summary["splits"][split] = sp
    env = os.path.join(out_dir, "environment.json")
    if os.path.exists(env):
        summary["environment"] = json.load(open(env))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=str)
    return summary


# ---------------------------------------------------------------------------------------------------- main
def get_sample(args, out_dir, cfg, scalers, force=False):
    sample_csv = os.path.join(out_dir, f"sample_{args.split}.csv")
    meta_json = os.path.join(out_dir, f"sampling_{args.split}.json")
    want = (args.n, args.seed, not args.no_listing, not args.no_header_check)
    if not force and os.path.exists(sample_csv) and os.path.exists(meta_json):
        meta = json.load(open(meta_json))
        if (meta["n"], meta["seed"], meta["use_listing"], meta["header_check"]) == want:
            log(f"reusing {sample_csv}")
            return pd.read_csv(sample_csv), meta
        raise RuntimeError(f"{sample_csv} was drawn with (n, seed, listing, header check) = "
                           f"{(meta['n'], meta['seed'], meta['use_listing'], meta['header_check'])}, not {want}; "
                           "rerun --dry-run with the new settings (and move the old per-sample file away)")
    t0 = time.time()
    sample, meta = select_sample(args.split, args.n, args.seed, use_listing=not args.no_listing,
                                 header_check=not args.no_header_check)
    meta["selection_seconds"] = round(time.time() - t0, 1)
    meta["dataset_join_check"] = check_joins(sample, cfg, scalers, args.split, args.alias_target_frame)
    meta["dataset_join_all_ok"] = all(x["ok"] for x in meta["dataset_join_check"])
    sample.to_csv(sample_csv, index=False)
    json.dump(meta, open(meta_json, "w"), indent=1, default=str)
    return sample, meta


def estimate(sample, bw, alias_target):
    prim = sample[sample.role == "primary"]
    cols = ["frame_tm60_bytes", "frame_t_bytes"] + ([] if alias_target else ["frame_tp60_bytes"])
    fb = prim[cols].to_numpy(float).sum(1)
    s3, hf = bw["s3_frame"]["MB_per_s"] * 1e6, bw["hf_cdn_csv"]["MB_per_s"] * 1e6
    d = fb / s3
    load = len(cols) * SEC_LOAD_FRAME
    compute = SEC_STATS + SEC_FORWARD
    # the first sample's frames download while the checkpoint is fetched and the model is built; afterwards sample
    # j+1's frames download while sample j runs through the model (they start once sample j has been read)
    total = CKPT_SIZE / hf + max(SEC_MODEL, d[0] - CKPT_SIZE / hf if len(d) else 0)
    total += sum(load + max(compute, d[j + 1] if j + 1 < len(d) else 0) for j in range(len(d)))
    return {"frames_per_sample": len(cols), "frames_total_bytes": int(fb.sum()),
            "frames_max_on_disk_bytes": int(fb.max()) if len(fb) else 0,
            "per_sample_download_minutes_mean": round(float(d.mean()) / 60, 2) if len(d) else 0,
            "per_sample_compute_minutes": round((load + compute) / 60, 2),
            "download_hidden_behind_forward": bool((d[1:] <= compute).all()) if len(d) > 1 else True,
            "total_minutes": round(total / 60, 1), "forward_passes": int(len(prim))}


def dry_run(args, out_dir):
    from surya.utils.data import build_scalers
    cfg, cfg_info = load_configs()
    scalers = build_scalers(info=yaml.safe_load(open(SCALERS)))
    check = metric_self_test(out_dir)
    log(f"metric self-test passed (max abs diff {check['max_abs_diff_overall']:.2e} vs metrics.py)")
    imp = paper_implied()
    log(f"Table 4 implies prevalence {imp['solutions'][0]['prevalence']:.4f}, TPR {imp['solutions'][0]['tpr']:.4f}, "
        f"FPR {imp['solutions'][0]['fpr']:.4f} (prevalence range over rounding "
        f"{imp['range_over_rounding_box']['prevalence'][0]:.4f}-{imp['range_over_rounding_box']['prevalence'][1]:.4f})")
    sample, meta = get_sample(args, out_dir, cfg, scalers, force=True)
    prim = sample[sample.role == "primary"]
    bw = bandwidth_probe(prim["frame_t_key"].iloc[0])
    ck = retry(lambda: requests.head(hf_url(CKPT_REPO, CKPT_REV, CKPT_NAME, kind="models"), allow_redirects=False,
                                     timeout=60))
    pk = checkpoint_pickle_check()
    est = {"assumptions_s": {"forward": SEC_FORWARD, "read_and_normalise_per_frame": SEC_LOAD_FRAME,
                             "input_statistics": SEC_STATS, "model_build_and_load": SEC_MODEL},
           "measured_bandwidth": bw,
           "download_target_frame (default)": estimate(sample, bw, False),
           "alias_target_frame": estimate(sample, bw, True)}
    chosen = est["alias_target_frame" if args.alias_target_frame else "download_target_frame (default)"]
    disk = {"checkpoint_bytes": CKPT_SIZE, "checkpoint_head_x_linked_size": ck.headers.get("x-linked-size"),
            "frames_max_on_disk_bytes": chosen["frames_max_on_disk_bytes"],
            "frames_total_bytes_over_run": chosen["frames_total_bytes"],
            "peak_disk_gb_new_files": round((CKPT_SIZE + chosen["frames_max_on_disk_bytes"]) / 1e9, 2),
            "foundation_model_file_already_on_disk_bytes": os.path.getsize(FM_CKPT)}
    meta.update({"dry_run": True, "estimate": est, "disk": disk, "checkpoint_pickle": pk, "config": cfg_info,
                 "paper_implied": imp, "alias_target_frame": bool(args.alias_target_frame),
                 "forward_passes_planned": int(len(prim)), "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}",
                 "labels": f"{LABEL_REPO}@{LABEL_REV}", "upstream_dir": UPSTREAM,
                 "upstream_head": os.popen(f"git -C {UPSTREAM} rev-parse --short HEAD").read().strip(),
                 "scalers_sha256": sha256_file(SCALERS)})
    json.dump(meta, open(os.path.join(out_dir, f"sampling_{args.split}.json"), "w"), indent=1, default=str)
    ci = meta["candidate_info"]
    log(f"split {args.split}: {ci['rows_in_csv']} rows, prevalence {ci['prevalence_csv']:.4f}; {ci['candidates']} "
        f"candidates, prevalence {ci['prevalence_candidates']:.4f}")
    log(f"sample: {len(prim)} primaries ({int((prim['class'] == 'pos').sum())} pos / "
        f"{int((prim['class'] == 'neg').sum())} neg) + {int((sample.role == 'reserve').sum())} reserves; "
        f"{meta['n_skipped']} draws skipped; dataset join check ok: {meta['dataset_join_all_ok']}")
    log(f"checkpoint pickle: only tensor globals {pk['only_tensor_globals']} {pk['globals']}")
    log(f"frames over the run {chosen['frames_total_bytes'] / 1e9:.1f} GB (at most "
        f"{chosen['frames_max_on_disk_bytes'] / 1e9:.2f} GB on disk); checkpoint {CKPT_SIZE / 1e9:.2f} GB; "
        f"peak new disk ~{disk['peak_disk_gb_new_files']} GB")
    log(f"estimate: {chosen['total_minutes']} min for {chosen['forward_passes']} forward passes "
        f"(alias-target-frame: {est['alias_target_frame']['total_minutes']} min); S3 {bw['s3_frame']['MB_per_s']} MB/s")
    print(sample[["role", "stratum", "block", "timestamp", "label_max", "max_goes_class"]].to_string(index=False))


def input_stats(x, channels):
    """Per channel and input time: mean and std after the scalers, and the count of non-finite values."""
    import torch
    with torch.no_grad():
        s, m = torch.std_mean(x[0], dim=(-2, -1))  # (C, T)
        nonfinite = int(x.numel() - torch.isfinite(x).sum())
    return {"nonfinite": nonfinite,
            "per_channel_mean_std": {ch: [[round(float(m[i, k]), 4), round(float(s[i, k]), 4)] for k in range(m.shape[1])]
                                     for i, ch in enumerate(channels)}}


def run(args, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from surya.utils.data import build_scalers
    add_task_dir()
    from dataset import SolarFlareDataset
    from finetune import custom_collate_fn

    cfg, cfg_info = load_configs()
    cfg["data"]["scalers"] = yaml.safe_load(open(SCALERS))  # as infer.main
    cfg["dtype"] = torch.float32  # as infer.main for dtype "float32"
    scalers = build_scalers(info=cfg["data"]["scalers"])
    metric_self_test(out_dir)
    sample, meta = get_sample(args, out_dir, cfg, scalers)
    if not meta.get("dataset_join_all_ok", False):
        raise RuntimeError("dataset join check failed in the sample file; rerun --dry-run")

    recs_path = per_sample_path(out_dir, args.split)
    previous = read_records(recs_path)
    done = {t for t, r in previous.items() if r.get("status") == "ok"}
    ok_blocks = {(r["stratum"], int(r["block"])) for r in previous.values() if r.get("status") == "ok"}
    blocks = {}
    for r in sample.to_dict("records"):
        blocks.setdefault((r["stratum"], int(r["block"])), {})[r["role"]] = r
    work, complete = [], 0
    for key, b in sorted(blocks.items()):
        if key in ok_blocks:  # one sample per block
            complete += 1
            continue
        failed_before = b.get("primary") and previous.get(b["primary"]["timestamp"], {}).get("status") == "failed"
        if b.get("primary") and not failed_before:
            work.append(b["primary"])
        elif b.get("reserve"):
            work.append(b["reserve"])
    pending = [r for r in work if r["timestamp"] not in done][: args.max_forward]
    log(f"{len(blocks)} blocks in the sample, {complete} already scored, {len(pending)} samples to run now "
        f"(cap {args.max_forward})")
    if not pending:
        summarize(out_dir, args.seed)
        return

    cache = C.FrameCache(os.path.join(WORK, "frames"))
    pool = cf.ThreadPoolExecutor(1)

    def frames_for(row):
        ft = frame_times(row["timestamp"])
        return ft[:2] if args.alias_target_frame else ft

    fut, fut_for = pool.submit(cache.ensure, frames_for(pending[0])), 0  # downloads while the model is prepared
    ckpt = ensure_checkpoint()
    t0 = time.time()
    model, minfo = load_model(cfg, ckpt)
    minfo["seconds_build_and_load"] = round(time.time() - t0, 1)
    env = {"torch": torch.__version__, "threads": torch.get_num_threads(), "precision": "float32, no autocast",
           "grad": "torch.no_grad()", "upstream_head": os.popen(f"git -C {UPSTREAM} rev-parse --short HEAD").read().strip(),
           "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}", "checkpoint_sha256": CKPT_SHA256,
           "foundation_model": FM_CKPT, "labels": f"{LABEL_REPO}@{LABEL_REV}", "scalers": SCALERS,
           "scalers_sha256": sha256_file(SCALERS), "config": cfg_info, "alias_target_frame": bool(args.alias_target_frame),
           "model": minfo, "peak_rss_gb_after_model_load": C.peak_rss_gb()}
    json.dump(env, open(os.path.join(out_dir, "environment.json"), "w"), indent=1, default=str)
    log(f"model ready in {minfo['seconds_build_and_load']} s, peak RSS {C.peak_rss_gb()} GB")

    j = 0
    while j < len(pending):
        row = pending[j]
        t = pd.Timestamp(row["timestamp"])
        rec = {k: row.get(k) for k in ("split", "role", "class", "year", "stratum", "block", "timestamp", "label_max",
                                       "max_goes_class", "label_cum", "frame_tm60_key", "frame_t_key",
                                       "frame_tp60_key")}
        rec["alias_target_frame"] = bool(args.alias_target_frame)
        t_start = time.time()
        ds = loader = batch = metadata = x = logit = None
        retry_with_reserve = False
        try:
            if fut_for != j:
                fut, fut_for = pool.submit(cache.ensure, frames_for(row)), j
            dl_s = fut.result()
            t1 = time.time()
            ds = build_flare_dataset(cfg, scalers, args.split, t, cache.root, SolarFlareDataset, args.alias_target_frame)
            assert len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == t, "dataset did not select the time stamp"
            loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
            batch, metadata = next(iter(loader))
            t2 = time.time()
            if j + 1 < len(pending):  # the frames are in memory: replace them on disk by the next sample's
                fut, fut_for = pool.submit(cache.ensure, frames_for(pending[j + 1])), j + 1
            x = batch["ts"]
            rec["label_from_dataset"] = int(batch["label"].item())
            assert rec["label_from_dataset"] == int(row["label_max"]), "dataset label differs from the CSV"
            rec["input"] = {"ts_shape": list(x.shape), "ts_dtype": str(x.dtype),
                            "time_delta_input": [float(v) for v in batch["time_delta_input"].ravel()],
                            "forecast_shape_dropped": list(batch["forecast"].shape),
                            **input_stats(x, cfg["data"]["channels"])}
            rec["metadata"] = {k: [str(v) for v in np.ravel(metadata[k])] for k in metadata}
            batch.pop("forecast")  # the t+60 target is not used by HelioSpectformer1D; frees 0.87 GB
            t3 = time.time()
            with torch.no_grad():
                out = model(batch)
            t4 = time.time()
            rec["output"] = {"shape": list(out.shape), "dtype": str(out.dtype)}
            logit = float(out.reshape(-1)[0])
            rec["logit"] = logit
            rec["prob"] = float(torch.sigmoid(out.reshape(-1)[0]))
            rec["pred_infer_py"] = int(torch.sigmoid(out).item() > 0.5)  # infer.py's expression
            rec["pred_finetune_eval"] = int(logit > THRESHOLD)
            rec["seconds"] = {"frame_download_wait": round(t1 - t_start, 1), "frame_download_call": round(dl_s, 1),
                              "dataset_and_load": round(t2 - t1, 1), "input_stats": round(t3 - t2, 1),
                              "forward": round(t4 - t3, 1), "total": round(time.time() - t_start, 1)}
            rec["status"] = "ok"
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} label {row['label_max']} ({row['max_goes_class']}) "
                f"logit {logit:+.3f} p {rec['prob']:.3f} forward {rec['seconds']['forward']} s "
                f"load {rec['seconds']['dataset_and_load']} s")
        except Exception as e:  # record, substitute the block's reserve once, carry on
            rec["status"] = "failed"
            rec["error"] = repr(e)[:2000]
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} FAILED: {e!r}")
            retry_with_reserve = row["role"] == "primary"
        finally:
            ds = loader = batch = metadata = x = out = None  # release the 1.7 GB input
        if retry_with_reserve:
            res = blocks[(row["stratum"], int(row["block"]))].get("reserve")
            if res is not None and res["timestamp"] not in done and all(p["timestamp"] != res["timestamp"] for p in pending):
                pending.append(res)
        rec["peak_rss_gb"] = C.peak_rss_gb()
        with open(recs_path, "a") as fp:
            fp.write(json.dumps(rec, default=str) + "\n")
        gc.collect()
        j += 1
    pool.shutdown(wait=True)
    cache.clear()
    if args.delete_checkpoint:
        os.remove(ckpt)
    s = summarize(out_dir, args.seed)
    sp = s["splits"].get(args.split, {})
    if "metrics" in sp:
        r = sp["metrics"]["rules"][PRIMARY_RULE]
        rw = r["reweighted"]["candidate_set_of_split (primary)"]
        log(f"{args.split}: {sp['metrics']['n_pos']} pos + {sp['metrics']['n_neg']} neg; TSS {r['tss']['estimate']:.3f} "
            f"{r['tss']['ci95']}; HSS(rw) {rw['hss']['estimate']:.3f} {rw['hss']['ci95']}; F1(rw) "
            f"{rw['f1']['estimate']:.3f} {rw['f1']['ci95']} (paper 0.436 / 0.522 / 0.561)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="select and check the sample; no weights, frames or model")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--n", type=int, default=None, help="samples, half positive (default 40 test, 20 validation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "Surya", "flare_forecasting"))
    ap.add_argument("--max-forward", type=int, default=40, help="forward passes allowed in this run")
    ap.add_argument("--no-listing", action="store_true", help="do not list the S3 bucket (prevalence from the CSV)")
    ap.add_argument("--no-header-check", action="store_true", help="skip the netCDF header check of each frame")
    ap.add_argument("--alias-target-frame", action="store_true",
                    help="do not download the t+60 frame; its index row points at the t file (never read by the model)")
    ap.add_argument("--delete-checkpoint", action="store_true", help="remove the 1.8 GB checkpoint at the end")
    ap.add_argument("--summarize-only", action="store_true", help="recompute summary.json from the jsonl files")
    args = ap.parse_args()
    args.n = args.n or DEFAULT_N[args.split]
    if args.n < 2:
        ap.error("--n must be at least 2 (one positive, one negative)")
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)  # the authors' dataset writes logs/data/ relative to the working directory
    if args.summarize_only:
        summarize(out_dir, args.seed)
        return
    ensure_small_inputs(args.split)
    if args.dry_run:
        dry_run(args, out_dir)
    else:
        run(args, out_dir)


if __name__ == "__main__":
    main()
