"""Active-region (PIL) segmentation with the released fine-tuned Surya model, scored on a seeded sample of ARPIL.

PARTIAL REPRODUCTION. Table 3 of arXiv 2508.14112 v2 (Sec. 2.6.2) reports IoU 0.768 and Dice 0.853 for Surya
(UNet: 0.688 / 0.801) without saying which split, how many images, which threshold, or how the scores were averaged.
The repo has no script that computes either number (infer.py only plots the sigmoid output; finetune.py defines
DiceLoss and IoULoss but trains and validates with BCEWithLogitsLoss). This script scores the released checkpoint
nasa-ibm-ai4science/ar_segmentation_surya on a small, seeded, year-stratified sample of the SuryaBench ARPIL test
split (2020-2024) or validation split (Jan 15-31 of 2011-2019) on a CPU, and reports IoU and Dice with bootstrap
95% intervals over images. It cannot say whether the full split would give the paper's number; it can say whether
the paper's number is plausible for these images.

What is the authors' code (imported unchanged from the clone, nothing copied or edited):
  * ArDSDataset (downstream_examples/ar_segmentation/dataset.py) with the arguments finetune.get_dataloaders uses
    for its validation set (config.yaml): input frame at t only (time_delta_input_minutes [0]), the mask
    union_with_intersect / 255 as the target, scalers from scalers.yaml. As in the authors' class, a time t is only
    accepted when t+60 min is also in the SDO index (time_delta_target_minutes +60, rollout_steps 0), although the
    t+60 frame is never read; so t+60 is HEAD-checked on S3 and listed in the per-sample index but not downloaded.
  * custom_collate_fn, IoULoss and DiceLoss from finetune.py.
  * get_model and apply_peft_lora from infer.py with config_infer.yaml (infer.py's default config), then the
    checkpoint loading of infer.load_model with load_state_dict(strict=True).
Differences from infer.py (none changes the network or its weights): torch.load uses weights_only=True and
mmap=True (the file holds only an OrderedDict of float tensors, checked from its pickle opcodes); the forward pass
runs in float32 without autocast, because CPU bfloat16 autocast breaks Surya (results/Surya/bf16_diagnosis); the
DataLoader uses num_workers=0 and a one-sample dataset per time stamp.

Metrics (Sec. 2.6.2 gives no equation; standard definitions are used and both aggregations are reported):
  hard, threshold 0.5 on sigmoid(logit): IoU = TP/(TP+FP+FN), Dice = 2TP/(2TP+FP+FN); an image whose mask and
    prediction are both empty scores 1 (the eps convention of the repo's loss classes in the limit).
  soft, exactly the repo's loss classes: IoU = 1 - IoULoss(eps=1e-6)(logits, mask), Dice = 1 - DiceLoss(smooth=1e-5);
    eps/smooth from config.yaml (model.iou.eps, model.dice.smooth; YAML reads them as strings, so they are cast).
  per-image mean (mean over images of the per-image score) and pooled (sums of TP/FP/FN, or of the soft sums, over all
    images first). Bootstrap: 10,000 resamples of images, percentile 95% interval.

Data. Input frames come from the public bucket s3://nasa-surya-bench (bitwise-identical to the HF copies, see
results/Surya/s3_vs_hf_frames.json) through surya_common.FrameCache, at most two frames on disk at once (the next
sample's frame is prefetched while the current one is scored). Masks come from data.tar.gz of
nasa-ibm-ai4science/surya-bench-ar-segmentation (1.31 GB): it is streamed once over HTTP (never stored), its SHA-256
is checked, and only the sampled members are written to disk.

Sampling. Candidates are the rows of <split>.csv with present == 1 (for the validation split also with t and t+60 in
the core-sdo valid_index_surya_1_0.csv that finetune.py uses). The sample is split evenly over years (remainder to
years drawn with the seed); inside a year the candidates are cut into equal consecutive time blocks, one image per
block, drawn in a seeded random order. A candidate is skipped (and the next one in the block tried) if its frame at
t or t+60 is not on S3 or the frame at t lacks one of the 13 channels (read from the netCDF header with HTTP range
requests). Every skip is recorded with its reason. Each block also gets a reserve, used only if the primary's mask is
missing from data.tar.gz or the sample fails.

Usage (from the repository root):
  python scripts/Surya/downstream/ar_segmentation_subset_eval.py --dry-run --split test --n 30 --seed 0
  python scripts/Surya/downstream/ar_segmentation_subset_eval.py --split test --n 30 --seed 0
  python scripts/Surya/downstream/ar_segmentation_subset_eval.py --split validation --n 10 --seed 0
  python scripts/Surya/downstream/ar_segmentation_subset_eval.py --summarize-only
A rerun skips time stamps already in per_sample_<split>.jsonl, so an interrupted run resumes.
"""
import argparse
import concurrent.futures as cf
import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import surya_common as C  # noqa: E402  (sets HF_HUB_DISABLE_XET=1 before huggingface_hub is imported)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
UPSTREAM = "/home/user/upstream/surya/Surya"
TASK_DIR = os.path.join(UPSTREAM, "downstream_examples", "ar_segmentation")
FM_DIR = "/home/user/work/surya_run/data/Surya-1.0"
WORK = "/home/user/work/surya_ft/ar_segmentation"
# dataset.py ArDSDataset.__getitem__ hard-codes os.path.join("./assets/surya-bench-ar-segmentation", file_path) and
# config.yaml's ar_index_* / scalers_path are relative too, so the script runs with WORK as the working directory and
# lays out WORK/assets/ the way download_data.sh + the README's tar step would.
ASSETS = os.path.join(WORK, "assets")
LABEL_DIR = os.path.join(ASSETS, "surya-bench-ar-segmentation")

HF = "https://huggingface.co"
LABEL_REPO, LABEL_REV = "nasa-ibm-ai4science/surya-bench-ar-segmentation", "615a7df4ac3498c97d0d039e7b0654774ad86602"
TAR_NAME, TAR_SIZE = "data.tar.gz", 1306605651
TAR_SHA256 = "d49e88e102f200d27037d76eb429e57b5fa461c703a7288d871d1b0c9afb3034"  # HF LFS oid
CKPT_REPO, CKPT_REV = "nasa-ibm-ai4science/ar_segmentation_surya", "73a6de53c4bea812f7bc2ac29b9351ae2f2fe5fe"
CKPT_NAME, CKPT_SIZE = "ar_segmentation_weights.pth", 1784013352
CKPT_SHA256 = "ad391831f340c4f894a23bd269e7c4b75dea58090c825278513d36f0f85456ad"  # HF LFS oid
CORE_SDO_REPO, CORE_SDO_REV = "nasa-ibm-ai4science/core-sdo", "9c10abe52c1f206d1b66715745229e29ef543cc6"

PAPER_TABLE3 = {  # arXiv 2508.14112 v2, Sec. 2.6.2, Table 3 (copied exactly)
    "Surya": {"params": "4.1 M", "iou": 0.768, "dice": 0.853},
    "Unet": {"params": "9.2 M", "iou": 0.688, "dice": 0.801},
}
SPLIT_CSV = {"test": "test.csv", "validation": "validation.csv"}
CONFIG_AR_INDEX = {"test": "ar_index_test", "validation": "ar_index_valid"}  # keys in config.yaml
STRATA_YEARS = {"test": list(range(2020, 2025)), "validation": list(range(2011, 2020))}
DEFAULT_N = {"test": 30, "validation": 10}
THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]
HARD_THRESHOLD = 0.5
N_BOOT = 10000
# Timing assumptions for the estimate (the dry run cannot time the model): one fp32 forward pass of the 4096x4096
# Surya backbone took 136 s on this machine (results/Surya/validation_test, 271.9 s for two passes); reading and
# normalising one 13-channel frame is taken as 30 s; scoring as 5 s; building and loading the model as 120 s.
SEC_FORWARD, SEC_LOAD, SEC_SCORE, SEC_MODEL = 136, 30, 5, 120


# ----------------------------------------------------------------------------------------------- small utilities
def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def hf_url(repo, rev, path, kind="datasets"):
    return f"{HF}/{kind}/{repo}/resolve/{rev}/{path}" if kind == "datasets" else f"{HF}/{repo}/resolve/{rev}/{path}"


def fetch_small(url, dest, max_bytes=50 * 2 ** 20):
    """Download a small file (labels, index, config) once; refuse anything above max_bytes."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
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


def s3_exists_retry(t, tries=4):
    for i in range(tries):
        try:
            return C.s3_exists(t)
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


class RangeFile(io.RawIOBase):
    """Read-only file object over HTTP range requests, with a block cache; enough for h5py to read a header."""

    def __init__(self, url, block=1 << 16):
        self.url, self.block, self.pos, self.cache, self.nbytes = url, block, 0, {}, 0
        r = requests.head(url, timeout=60)
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
            for k in range(4):
                try:
                    r = requests.get(self.url, headers={"Range": f"bytes={a}-{b}"}, timeout=60)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if k == 3:
                        raise
                    time.sleep(2 ** (k + 1))
            self.cache[i] = r.content
            self.nbytes += len(r.content)
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


def remote_nc_channels(key):
    """(worker) Shapes and stored (compressed) chunk sizes of the data variables of one S3 netCDF frame, read from
    its HDF5 header only (about 0.8 MB of range requests per frame)."""
    import h5py
    rf = RangeFile(f"{C.S3}/{key}")
    with h5py.File(io.BufferedReader(rf, 1 << 16), "r") as f:
        shapes = {k: tuple(f[k].shape) for k in f.keys() if isinstance(f[k], h5py.Dataset)}
        stored = {}
        for k in C.CHANNELS:
            try:
                stored[k] = int(sum(f[k].id.get_chunk_info(i).size for i in range(f[k].id.get_num_chunks())))
            except Exception:
                stored[k] = None
    return shapes, rf.nbytes, stored


class ResumableHTTP(io.RawIOBase):
    """Sequential reader over one HTTP object that reconnects with a Range header after a dropped connection and
    hashes every byte it hands out (so a streamed archive can be verified without being stored)."""

    def __init__(self, url, size, retries=10):
        self.url, self.size, self.retries = url, size, retries
        self.pos, self.buf, self.it, self.resp = 0, b"", None, None
        self.sha = hashlib.sha256()
        self.reconnects = 0

    def readable(self):
        return True

    def _open(self):
        if self.resp is not None:
            self.resp.close()
        headers = {"Range": f"bytes={self.pos}-"} if self.pos else {}
        self.resp = requests.get(self.url, headers=headers, stream=True, timeout=120)
        self.resp.raise_for_status()
        if self.pos and self.resp.status_code != 206:
            raise IOError("server ignored the Range header on reconnect")
        self.it = self.resp.raw.stream(1 << 20, decode_content=False)

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        fails = 0
        while not self.buf:
            try:
                if self.it is None:
                    self._open()
                self.buf = next(self.it)
            except StopIteration:  # body ended early: reconnect from self.pos, but not forever
                self.it = None
                if self.pos >= self.size:
                    return 0
                fails += 1
                self.reconnects += 1
                if fails > self.retries:
                    raise IOError(f"stream ended at byte {self.pos} of {self.size} after {fails} reconnects")
                time.sleep(min(60, 2 ** fails))
            except Exception:  # dropped connection, timeout, proxy reset
                self.it = None
                fails += 1
                self.reconnects += 1
                if fails > self.retries:
                    raise
                time.sleep(min(60, 2 ** fails))
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]
        self.sha.update(self.buf[:n])
        self.buf = self.buf[n:]
        self.pos += n
        return n


# --------------------------------------------------------------------------------------------------- metrics
def iou_dice_counts(tp, fp, fn):
    """Hard IoU and Dice from pixel counts; both empty (TP=FP=FN=0) scores 1."""
    tp, fp, fn = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn))
    di, dd = tp + fp + fn, 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        iou = np.where(di > 0, tp / di, 1.0)
        dice = np.where(dd > 0, 2 * tp / dd, 1.0)
    return iou, dice


def soft_iou(s_pt, s_p, s_t, eps):
    """finetune.IoULoss: (sum(p*t)+eps) / (sum(p)+sum(t)-sum(p*t)+eps) per image."""
    s_pt, s_p, s_t = (np.asarray(x, dtype=np.float64) for x in (s_pt, s_p, s_t))
    return (s_pt + eps) / (s_p + s_t - s_pt + eps)


def soft_dice(s_pt, s_p, s_t, smooth):
    """finetune.DiceLoss: (2 sum(p*t)+smooth) / (sum(p)+sum(t)+smooth) over everything it is given."""
    s_pt, s_p, s_t = (np.asarray(x, dtype=np.float64) for x in (s_pt, s_p, s_t))
    return (2 * s_pt + smooth) / (s_p + s_t + smooth)


def image_counts(logits, target):
    """Per-image sums from one (1,1,H,W) logit map and binary target, in float64."""
    import torch
    p = torch.sigmoid(logits.double())
    t = target.double()
    tb = t > 0.5
    out = {"n_pixels": int(t.numel()), "target_px": int(tb.sum()),
           "soft": {"sum_p": float(p.sum()), "sum_t": float(t.sum()), "sum_pt": float((p * t).sum())},
           "hard": {}}
    for thr in THRESHOLDS:
        pb = p > thr
        out["hard"][f"{thr:.1f}"] = {"tp": int((pb & tb).sum()), "fp": int((pb & ~tb).sum()),
                                     "fn": int((~pb & tb).sum())}
    del p, t, tb
    return out


def bootstrap(stat, n, n_boot=N_BOOT, seed=0):
    """Percentile 95% interval of stat(idx) over resamples of n images; stat takes an (B, n) index array."""
    rng = np.random.default_rng(seed)
    vals = stat(rng.integers(0, n, size=(n_boot, n)))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def aggregate(tp, fp, fn, s_pt, s_p, s_t, eps, smooth, n_boot=N_BOOT, seed=0):
    tp, fp, fn, s_pt, s_p, s_t = (np.asarray(x, dtype=np.float64) for x in (tp, fp, fn, s_pt, s_p, s_t))
    n = len(tp)
    iou, dice = iou_dice_counts(tp, fp, fn)
    siou, sdice = soft_iou(s_pt, s_p, s_t, eps), soft_dice(s_pt, s_p, s_t, smooth)
    nonempty = (tp + fn) > 0
    res = {}

    def put(name, est, stat):
        res[name] = {"estimate": float(est), "ci95": bootstrap(stat, n, n_boot, seed), "n_images": int(n)}

    put("hard_iou_mean_over_images", iou.mean(), lambda i: iou[i].mean(1))
    put("hard_dice_mean_over_images", dice.mean(), lambda i: dice[i].mean(1))
    put("hard_iou_pooled_pixels", tp.sum() / max(1.0, (tp + fp + fn).sum()),
        lambda i: tp[i].sum(1) / np.maximum(1.0, (tp + fp + fn)[i].sum(1)))
    put("hard_dice_pooled_pixels", 2 * tp.sum() / max(1.0, (2 * tp + fp + fn).sum()),
        lambda i: 2 * tp[i].sum(1) / np.maximum(1.0, (2 * tp + fp + fn)[i].sum(1)))
    put("soft_iou_mean_over_images_repo_IoULoss", siou.mean(), lambda i: siou[i].mean(1))
    put("soft_dice_mean_over_images_repo_DiceLoss", sdice.mean(), lambda i: sdice[i].mean(1))
    put("soft_iou_pooled_pixels", soft_iou(s_pt.sum(), s_p.sum(), s_t.sum(), eps),
        lambda i: soft_iou(s_pt[i].sum(1), s_p[i].sum(1), s_t[i].sum(1), eps))
    put("soft_dice_pooled_pixels", soft_dice(s_pt.sum(), s_p.sum(), s_t.sum(), smooth),
        lambda i: soft_dice(s_pt[i].sum(1), s_p[i].sum(1), s_t[i].sum(1), smooth))
    if nonempty.sum() >= 2:
        k = np.flatnonzero(nonempty)
        ik, dk = iou[k], dice[k]
        res["hard_iou_mean_over_images_with_nonempty_mask"] = {
            "estimate": float(ik.mean()), "ci95": bootstrap(lambda i: ik[i].mean(1), len(k), n_boot, seed),
            "n_images": int(len(k))}
        res["hard_dice_mean_over_images_with_nonempty_mask"] = {
            "estimate": float(dk.mean()), "ci95": bootstrap(lambda i: dk[i].mean(1), len(k), n_boot, seed),
            "n_images": int(len(k))}
    return res, iou, dice, siou, sdice


def metric_self_test(out_dir):
    """Checks the metric code against the repo's loss classes and an independent numpy computation."""
    import torch
    sys.path.insert(0, TASK_DIR)
    from finetune import DiceLoss, IoULoss  # authors' classes

    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))
    eps, smooth = float(cfg["model"]["iou"]["eps"]), float(cfg["model"]["dice"]["smooth"])
    g = torch.Generator().manual_seed(0)
    H = 96
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(H), indexing="ij")
    blob = (((yy - 40) ** 2 + (xx - 50) ** 2) < 15 ** 2).double()
    cases = {
        "random_logits_blob_mask": (torch.randn(1, 1, H, H, generator=g, dtype=torch.float64) * 3,
                                    blob[None, None]),
        "good_prediction_blob_mask": ((blob[None, None] * 2 - 1) * 4
                                      + torch.randn(1, 1, H, H, generator=g, dtype=torch.float64), blob[None, None]),
        "empty_mask_confident_empty_prediction": (torch.full((1, 1, H, H), -12.0, dtype=torch.float64),
                                                  torch.zeros(1, 1, H, H, dtype=torch.float64)),
        "empty_mask_some_false_positives": (torch.where(blob[None, None] > 0, 3.0, -12.0).double(),
                                            torch.zeros(1, 1, H, H, dtype=torch.float64)),
        "nonempty_mask_empty_prediction": (torch.full((1, 1, H, H), -12.0, dtype=torch.float64), blob[None, None]),
    }
    report, worst = {"eps": eps, "smooth": smooth, "cases": {}}, 0.0
    for name, (lg, tg) in cases.items():
        c = image_counts(lg, tg)
        s, h = c["soft"], c["hard"]["0.5"]
        mine_si = float(soft_iou(s["sum_pt"], s["sum_p"], s["sum_t"], eps))
        mine_sd = float(soft_dice(s["sum_pt"], s["sum_p"], s["sum_t"], smooth))
        repo_si = 1.0 - float(IoULoss(eps=eps)(lg, tg))
        repo_sd = 1.0 - float(DiceLoss(smooth=smooth)(lg, tg))
        a = torch.sigmoid(lg).numpy().ravel() > 0.5
        b = tg.numpy().ravel() > 0.5
        inter, union = np.logical_and(a, b).sum(), np.logical_or(a, b).sum()
        np_iou = inter / union if union else 1.0
        np_dice = 2 * inter / (a.sum() + b.sum()) if (a.sum() + b.sum()) else 1.0
        hi, hd = (float(x) for x in iou_dice_counts(h["tp"], h["fp"], h["fn"]))
        d = max(abs(mine_si - repo_si), abs(mine_sd - repo_sd), abs(hi - np_iou), abs(hd - np_dice),
                abs(hd - 2 * hi / (1 + hi)))
        worst = max(worst, d)
        report["cases"][name] = {"soft_iou_mine": mine_si, "soft_iou_repo_1_minus_IoULoss": repo_si,
                                 "soft_dice_mine": mine_sd, "soft_dice_repo_1_minus_DiceLoss": repo_sd,
                                 "hard_iou_mine": hi, "hard_iou_numpy_sets": float(np_iou),
                                 "hard_dice_mine": hd, "hard_dice_numpy_sets": float(np_dice), "max_abs_diff": d}
    # batch behaviour of the repo classes: IoULoss averages per-image IoU, DiceLoss pools the whole batch
    lg2 = torch.cat([cases["random_logits_blob_mask"][0], cases["good_prediction_blob_mask"][0]])
    tg2 = torch.cat([cases["random_logits_blob_mask"][1], cases["good_prediction_blob_mask"][1]])
    c1, c2 = image_counts(lg2[:1], tg2[:1])["soft"], image_counts(lg2[1:], tg2[1:])["soft"]
    s_pt, s_p, s_t = (np.array([c1[k], c2[k]]) for k in ("sum_pt", "sum_p", "sum_t"))
    b_iou_repo, b_dice_repo = 1 - float(IoULoss(eps=eps)(lg2, tg2)), 1 - float(DiceLoss(smooth=smooth)(lg2, tg2))
    b_iou_mean = float(soft_iou(s_pt, s_p, s_t, eps).mean())
    b_dice_pool = float(soft_dice(s_pt.sum(), s_p.sum(), s_t.sum(), smooth))
    report["batch_of_2"] = {"repo_IoULoss": b_iou_repo, "mean_of_per_image_soft_iou": b_iou_mean,
                            "repo_DiceLoss": b_dice_repo, "pooled_soft_dice": b_dice_pool,
                            "note": "IoULoss averages per-image IoU over the batch; DiceLoss pools all pixels of "
                                    "the batch. With batch_size 1 (config.yaml) both are per-image."}
    worst = max(worst, abs(b_iou_repo - b_iou_mean), abs(b_dice_repo - b_dice_pool))
    # aggregation on a hand-worked example: image 1 TP1 FP1 FN0 (IoU 1/2, Dice 2/3), image 2 TP3 FP0 FN1
    # (IoU 3/4, Dice 6/7): mean IoU 0.625, pooled IoU 4/6, mean Dice 0.7619048, pooled Dice 8/10
    agg, *_ = aggregate([1, 3], [1, 0], [0, 1], [1, 3], [2, 3], [1, 4], eps, smooth, n_boot=200)
    expect = {"hard_iou_mean_over_images": 0.625, "hard_iou_pooled_pixels": 4 / 6,
              "hard_dice_mean_over_images": (2 / 3 + 6 / 7) / 2, "hard_dice_pooled_pixels": 0.8}
    report["hand_example"] = {k: {"got": agg[k]["estimate"], "expected": v} for k, v in expect.items()}
    worst = max(worst, max(abs(agg[k]["estimate"] - v) for k, v in expect.items()))
    report["max_abs_diff_overall"] = worst
    report["passed"] = bool(worst < 1e-9)
    json.dump(report, open(os.path.join(out_dir, "metric_check.json"), "w"), indent=1)
    if not report["passed"]:
        raise AssertionError(f"metric self-test failed, max abs diff {worst}")
    return report


# -------------------------------------------------------------------------------------------- data and labels
def ensure_small_inputs(split):
    """Labels CSV (pinned revision), core-sdo validation index, scalers.yaml copied where config.yaml expects it."""
    os.makedirs(LABEL_DIR, exist_ok=True)
    fetch_small(hf_url(LABEL_REPO, LABEL_REV, SPLIT_CSV[split]), os.path.join(LABEL_DIR, SPLIT_CSV[split]))
    sc_dst = os.path.join(ASSETS, "scalers.yaml")
    if not os.path.exists(sc_dst):
        # core-sdo's scalers.yaml (config.yaml scalers_path) and Surya-1.0's are byte-identical (sha256 a944e8f1...)
        with open(os.path.join(FM_DIR, "scalers.yaml"), "rb") as a, open(sc_dst, "wb") as b:
            b.write(a.read())
    if split == "validation":
        fetch_small(hf_url(CORE_SDO_REPO, CORE_SDO_REV, "valid_index_surya_1_0.csv"),
                    os.path.join(ASSETS, "valid_index_surya_1_0.csv"))


def candidates(split):
    lab = pd.read_csv(os.path.join(LABEL_DIR, SPLIT_CSV[split]))
    info = {"rows_in_csv": int(len(lab)), "rows_present_1": int((lab["present"] == 1).sum())}
    lab = lab[lab["present"] == 1].copy()
    lab["timestamp"] = pd.to_datetime(lab["timestamp"])
    lab = lab[lab["timestamp"].dt.minute == 0]
    if split == "validation":
        idx = pd.read_csv(os.path.join(ASSETS, "valid_index_surya_1_0.csv"))
        have = set(pd.to_datetime(idx.loc[idx["present"] == 1, "timestep"]))
        keep = lab["timestamp"].map(lambda t: t in have and t + pd.Timedelta(minutes=60) in have)
        info["rows_with_t_and_t60_in_core_sdo_valid_index"] = int(keep.sum())
        lab = lab[keep]
    lab = lab.sort_values("timestamp").reset_index(drop=True)
    info["candidates"] = int(len(lab))
    info["candidates_per_year"] = {int(k): int(v) for k, v in lab.groupby(lab["timestamp"].dt.year).size().items()}
    return lab, info


def plan_blocks(cands, split, n, seed):
    rng = np.random.default_rng(seed)
    years = [y for y in STRATA_YEARS[split] if (cands["timestamp"].dt.year == y).any()]
    base, extra = divmod(n, len(years))
    alloc = {y: base for y in years}
    for y in sorted(int(v) for v in rng.choice(years, size=extra, replace=False)):
        alloc[y] += 1
    blocks = []
    for y in years:
        sub = cands[cands["timestamp"].dt.year == y].reset_index(drop=True)
        if alloc[y] == 0:
            continue
        for b, part in enumerate(np.array_split(np.arange(len(sub)), alloc[y])):
            order = rng.permutation(part)
            blocks.append({"stratum": y, "block": b, "block_first": str(sub["timestamp"][part[0]]),
                           "block_last": str(sub["timestamp"][part[-1]]), "block_candidates": int(len(part)),
                           "order": [(sub["timestamp"][i], sub["file_path"][i]) for i in order]})
    return alloc, blocks


def head_check(t):
    ok0, b0 = s3_exists_retry(t)
    if not ok0:
        return False, "input frame at t not on S3", {}
    t60 = t + pd.Timedelta(minutes=60)
    ok1, b1 = s3_exists_retry(t60)
    if not ok1:
        return False, "frame at t+60 min not on S3 (the authors' dataset only accepts t when t+60 is indexed)", {}
    return True, "", {"frame_bytes": b0, "frame_t60_bytes": b1}


def select_sample(split, n, seed, check_channels=True, want_per_block=2, max_tries=30):
    """Seeded, year-stratified sample with one primary and one reserve per time block (see module docstring)."""
    cands, info = candidates(split)
    alloc, blocks = plan_blocks(cands, split, n, seed)
    state = [{"pos": 0, "found": [], "skipped": []} for _ in blocks]
    ctx = mp.get_context("spawn")
    rounds = 0
    with cf.ThreadPoolExecutor(8) as tp, cf.ProcessPoolExecutor(4, mp_context=ctx) as pp:
        while True:
            batch = []  # (block index, timestamp, file_path)
            for bi, (blk, st) in enumerate(zip(blocks, state)):
                need = want_per_block - len(st["found"])
                while need > 0 and st["pos"] < min(len(blk["order"]), max_tries):
                    t, fpth = blk["order"][st["pos"]]
                    batch.append((bi, t, fpth))
                    st["pos"] += 1
                    need -= 1
            if not batch:
                break
            rounds += 1
            heads = list(tp.map(lambda x: head_check(x[1]), batch))
            chan = {}
            if check_channels:
                todo = [j for j, h in enumerate(heads) if h[0]]
                futs = {j: pp.submit(remote_nc_channels, C.s3_key(batch[j][1])) for j in todo}
                for j, f in futs.items():
                    try:
                        chan[j] = f.result()
                    except Exception as e:  # unreadable header
                        chan[j] = e
            for j, (bi, t, fpth) in enumerate(batch):  # assignment in permutation order -> deterministic
                ok, why, extra = heads[j]
                rec = {"timestamp": str(t), "mask_file_path": fpth}
                if ok and check_channels:
                    r = chan[j]
                    if isinstance(r, Exception):
                        ok, why = False, f"could not read the netCDF header: {r!r}"
                    else:
                        shapes, nread, stored = r
                        bad = [c for c in C.CHANNELS if shapes.get(c) != (4096, 4096)]
                        extra["header_bytes_read"] = nread
                        extra["stored_mb_per_channel"] = json.dumps(
                            {c: (round(v / 1e6, 1) if v is not None else None) for c, v in stored.items()})
                        if bad:
                            ok, why = False, f"input frame lacks channels {bad}"
                if ok and len(state[bi]["found"]) < want_per_block:
                    state[bi]["found"].append({**rec, **extra})
                elif not ok:
                    state[bi]["skipped"].append({**rec, "stratum": blocks[bi]["stratum"], "reason": why})
    rows, skipped = [], []
    for blk, st in zip(blocks, state):
        skipped += st["skipped"]
        for role, f in zip(("primary", "reserve"), st["found"]):
            rows.append({"split": split, "role": role, "stratum": blk["stratum"], "block": blk["block"],
                         "block_first": blk["block_first"], "block_last": blk["block_last"],
                         "block_candidates": blk["block_candidates"], "timestamp": f["timestamp"],
                         "mask_file_path": f["mask_file_path"], "frame_key": C.s3_key(f["timestamp"]),
                         "frame_bytes": f["frame_bytes"],
                         "frame_t60_key": C.s3_key(pd.Timestamp(f["timestamp"]) + pd.Timedelta(minutes=60)),
                         "frame_t60_bytes": f["frame_t60_bytes"],
                         "channels_checked": bool(check_channels),
                         "stored_mb_per_channel": f.get("stored_mb_per_channel", "")})
    df = pd.DataFrame(rows)
    meta = {"split": split, "n": n, "seed": seed, "check_channels": bool(check_channels),
            "allocation_per_year": {int(k): int(v) for k, v in alloc.items()}, "candidate_info": info,
            "n_blocks": len(blocks), "blocks_without_primary": [
                {"stratum": b["stratum"], "block": b["block"]} for b, s in zip(blocks, state) if not s["found"]],
            "blocks_without_reserve": int(sum(len(s["found"]) < 2 for s in state)),
            "skipped": skipped, "selection_rounds": rounds}
    return df, meta


def build_ar_dataset(cfg, scalers, split, t, sdo_root, ArDSDataset):
    """The authors' ArDSDataset for a single time stamp, with finetune.get_dataloaders' validation arguments."""
    t = pd.Timestamp(t)
    index_path = os.path.join(WORK, "index", f"{split}_{stamp(t)}.csv")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    C.write_index([t, t + pd.Timedelta(minutes=cfg["data"]["time_delta_target_minutes"])], index_path)
    channels = cfg["adapter"]["channels"] if cfg["adapter"]["use_channel_adapter"] else cfg["data"]["channels"]
    return ArDSDataset(
        sdo_data_root_path=sdo_root,
        index_path=index_path,
        time_delta_input_minutes=cfg["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=cfg["data"]["time_delta_target_minutes"],
        n_input_timestamps=cfg["model"]["time_embedding"]["time_dim"],
        rollout_steps=cfg["rollout_steps"],
        scalers=scalers,
        num_mask_aia_channels=cfg["num_mask_aia_channels"],
        drop_hmi_probablity=cfg["drop_hmi_probablity"],
        use_latitude_in_learned_flow=cfg["use_latitude_in_learned_flow"],
        channels=channels,
        phase="valid",
        ds_ar_index_paths=cfg["data"][CONFIG_AR_INDEX[split]],
    )


def load_configs():
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))  # finetune.py's config (dataset arguments)
    cfg_inf = yaml.safe_load(open(os.path.join(TASK_DIR, "config_infer.yaml")))  # infer.py's default (model)
    keys = ["model_type", "img_size", "patch_size", "in_channels", "time_embedding", "embed_dim", "depth",
            "spectral_blocks", "num_heads", "mlp_ratio", "rpe", "drop_rate", "window_size", "dp_rank",
            "learned_flow", "init_weights", "checkpoint_layers", "ensemble", "finetune", "ft_unembedding_type",
            "ft_out_chans", "use_lora", "lora_config"]
    diff = {k: (cfg["model"].get(k), cfg_inf["model"].get(k)) for k in keys if cfg["model"].get(k) != cfg_inf["model"].get(k)}
    for k in ("time_delta_input_minutes", "time_delta_target_minutes", "channels"):
        if cfg["data"][k] != cfg_inf["data"][k]:
            diff[f"data.{k}"] = (cfg["data"][k], cfg_inf["data"][k])
    return cfg, cfg_inf, diff


def check_joins(sample, cfg, scalers, split):
    """Build the authors' dataset for every sampled time (no frame is read) and confirm it yields exactly that time
    and the expected mask path."""
    sys.path.insert(0, TASK_DIR)
    from dataset import ArDSDataset
    out = []
    for r in sample.itertuples():
        ds = build_ar_dataset(cfg, scalers, split, r.timestamp, os.path.join(WORK, "frames", "a"), ArDSDataset)
        ok = (len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == pd.Timestamp(r.timestamp)
              and ds.ar_valid_indices.iloc[0]["file_path"] == r.mask_file_path)
        out.append({"timestamp": r.timestamp, "role": r.role, "dataset_len": len(ds), "ok": bool(ok),
                    "mask_path_read_by_ArDSDataset": os.path.abspath(
                        os.path.join("./assets/surya-bench-ar-segmentation", r.mask_file_path)),
                    "sdo_path_read_by_ArDSDataset": ds.index.loc[ds.valid_indices[0], "path"] if len(ds) else None})
    return out


def ensure_masks(file_paths, manifest_path):
    """Stream data.tar.gz once (not stored), write only the wanted members, verify the archive's SHA-256."""
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {"members": {}}
    want = {fp[len("data/"):] if fp.startswith("data/") else fp: fp for fp in file_paths}
    verified = manifest.get("sha256_ok") is True  # masks on disk came from a stream whose SHA-256 matched
    if verified:
        missing = [m for m, fp in want.items() if not os.path.exists(os.path.join(LABEL_DIR, fp))]
        if not missing:
            return manifest
        if manifest.get("complete_listing") and all(m in manifest["not_in_tar"] for m in missing):
            return manifest
    else:
        # no verified stream yet (first run, or an earlier stream crashed or failed its SHA-256): any mask already
        # on disk is unverified, so every wanted member is (re)written from a stream that is checked to the end
        missing = list(want)
        manifest = {"members": {}}
    log(f"streaming {TAR_NAME} ({TAR_SIZE / 1e9:.2f} GB) to extract {len(missing)} masks")
    t0 = time.time()
    reader = ResumableHTTP(hf_url(LABEL_REPO, LABEL_REV, TAR_NAME), TAR_SIZE)
    found, per_year, listing = {}, {}, []
    with tarfile.open(fileobj=io.BufferedReader(reader, 1 << 22), mode="r|gz", bufsize=1 << 20) as tf:
        for m in tf:
            name = m.name
            while name.startswith("./"):
                name = name[2:]
            if name.startswith("data/"):
                name = name[5:]
            if not m.isfile():
                continue
            listing.append((name, m.size))
            per_year[name[:4]] = per_year.get(name[:4], 0) + 1
            if name in want and m.size < 16 * 2 ** 20:
                data = tf.extractfile(m).read()
                dest = os.path.join(LABEL_DIR, want[name])  # our own path, never the member's
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(dest + ".part", "wb") as fp:
                    fp.write(data)
                os.replace(dest + ".part", dest)
                found[name] = m.size
    while reader.read(1 << 20):  # drain the gzip trailer so the hash covers the whole object
        pass
    sha = reader.sha.hexdigest()
    os.makedirs(os.path.join(WORK, "meta"), exist_ok=True)
    pd.DataFrame(listing, columns=["member", "bytes"]).to_csv(os.path.join(WORK, "meta", "tar_members.csv.gz"),
                                                             index=False)
    manifest = {"archive": f"{LABEL_REPO}@{LABEL_REV}/{TAR_NAME}", "bytes_streamed": reader.pos,
                "sha256": sha, "sha256_expected": TAR_SHA256, "sha256_ok": sha == TAR_SHA256,
                "reconnects": reader.reconnects, "seconds": round(time.time() - t0, 1),
                "n_file_members": len(listing), "file_members_per_year": dict(sorted(per_year.items())),
                "complete_listing": True, "members": {**manifest.get("members", {}), **found},
                "not_in_tar": sorted(set(missing) - set(found))}
    json.dump(manifest, open(manifest_path, "w"), indent=1)
    if not manifest["sha256_ok"]:
        raise IOError(f"{TAR_NAME} SHA-256 mismatch: {sha}")
    log(f"tar done in {manifest['seconds']} s: {len(found)} masks written, {len(manifest['not_in_tar'])} not in tar")
    return manifest


def mask_info(fp):
    import h5py
    with h5py.File(os.path.join(LABEL_DIR, fp), "r") as f:
        u = f["union_with_intersect"][...]
    vals = sorted(int(v) for v in np.unique(u))
    return {"shape": list(u.shape), "dtype": str(u.dtype), "values": vals, "positive_px": int((u > 0).sum())}


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


def load_model(cfg_inf, ckpt_path):
    """infer.load_model, step by step, with a safe unpickler and memory-mapped weights."""
    import torch
    sys.path.insert(0, TASK_DIR)
    import infer as I  # authors' get_model / apply_peft_lora
    from surya.utils.distributed import set_global_seed
    set_global_seed(42)  # as infer.main
    model = I.get_model(cfg_inf)
    if cfg_inf["model"]["use_lora"]:
        model = I.apply_peft_lora(model, cfg_inf)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True, mmap=True)
        mmap_used = True
    except RuntimeError as e:  # mmap needs the new zip format with aligned records; the values are the same without
        log(f"mmap load failed ({e!r}); loading into memory instead")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mmap_used = False
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
    n_lora = sum(v.numel() for k, v in model_state.items() if ".lora_" in k)
    lora_ranks = sorted({int(v.shape[0]) for k, v in model_state.items() if ".lora_A." in k})
    del checkpoint, model_state
    gc.collect()
    model.to(torch.device("cpu"))
    model.eval()
    model.requires_grad_(False)  # no autograd state under no_grad anyway; avoids any parameter caching
    if lora_ranks != [int(cfg_inf["model"]["lora_config"]["r"])]:
        raise RuntimeError(f"checkpoint LoRA ranks {lora_ranks} differ from config r={cfg_inf['model']['lora_config']['r']}")
    return model, {"n_parameters": int(n_total), "n_trainable_after_peft": int(n_trainable),
                   "state_dict_keys": int(n_keys), "load_state_dict_strict": True, "torch_load_mmap": mmap_used,
                   "checkpoint_lora_params": int(n_lora), "checkpoint_lora_ranks": lora_ranks,
                   "lora_alpha_from_config_not_checkpoint": cfg_inf["model"]["lora_config"]["lora_alpha"]}


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


def sample_key(r):
    return (str(r["timestamp"]), int(r["stratum"]), int(r["block"]), str(r["role"]))


def records_in_sample(recs, sample):
    """Only the records whose (time, year, block, role) is a row of the current sample file. A jsonl left over from a
    sample drawn with another --n/--seed (or redrawn after S3 changed) must neither mark blocks as done nor enter the
    summary."""
    keys = {sample_key(r) for r in sample.to_dict("records")}
    kept = {t: r for t, r in recs.items() if sample_key(r) in keys}
    return kept, len(recs) - len(kept)


COMPACT_CHANNELS = ["aia131", "aia171", "aia193", "aia211", "aia304", "aia335"]


def is_compact_frame(stored_mb_json):
    """Frames whose AIA 131-335 channels are stored in well under half the usual ~60 MB each. Their values are
    quantised (e.g. 2020-01-17 02:00 aia131 takes only ~10 distinct values in its top rows, the normal frames are
    continuous) and they were produced later (production_date 2025-05-18 vs 2025-04-03); disk geometry and intensity
    scale agree with normal frames. Reported as a sensitivity check, not excluded from the primary metric."""
    try:
        d = json.loads(stored_mb_json) if isinstance(stored_mb_json, str) else dict(stored_mb_json)
        v = [d[c] for c in COMPACT_CHANNELS if d.get(c) is not None]
        return bool(v) and float(np.median(v)) < 30.0
    except Exception:
        return None


def summarize(out_dir, seed=0):
    cfg = yaml.safe_load(open(os.path.join(TASK_DIR, "config.yaml")))
    eps, smooth = float(cfg["model"]["iou"]["eps"]), float(cfg["model"]["dice"]["smooth"])
    summary = {
        "label": "PARTIAL REPRODUCTION: released fine-tuned weights scored on a small seeded sample of the ARPIL "
                 "split named below, on CPU in float32. Not the paper's (unstated) evaluation set.",
        "paper_reference": {"source": "arXiv 2508.14112 v2, Sec. 2.6.2, Table 3", "values": PAPER_TABLE3,
                            "split_in_paper": "not stated", "metric_definition_in_paper": "not stated"},
        "primary_metric": "hard (sigmoid > 0.5) IoU and Dice, mean over images (both-empty image = 1)",
        "eps_iou": eps, "smooth_dice": smooth, "bootstrap": {"resamples": N_BOOT, "seed": seed,
                                                            "interval": "percentile 95%"},
        "splits": {}}
    for split in ("test", "validation"):
        sample_csv = os.path.join(out_dir, f"sample_{split}.csv")
        all_recs = read_records(per_sample_path(out_dir, split))
        if not all_recs:
            continue
        if not os.path.exists(sample_csv):
            raise FileNotFoundError(f"{sample_csv} is missing; cannot tell which records belong to the sample")
        sample = pd.read_csv(sample_csv)
        all_recs, n_foreign = records_in_sample(all_recs, sample)
        recs = [r for r in all_recs.values() if r.get("status") == "ok"]
        failed = [r for r in all_recs.values() if r.get("status") != "ok"]
        if not recs:
            continue
        planned = sorted({(int(r["stratum"]), int(r["block"])) for r in sample.to_dict("records")})
        by_block = {}  # one image per time block: the primary if it was scored, else the reserve
        for r in recs:
            k = (int(r["stratum"]), int(r["block"]))
            if k not in by_block or r["role"] == "primary":
                by_block[k] = r
        recs = sorted(by_block.values(), key=lambda r: r["timestamp"])
        h = [r["counts"]["hard"]["0.5"] for r in recs]
        s = [r["counts"]["soft"] for r in recs]
        tp, fp, fn = (np.array([x[k] for x in h], float) for k in ("tp", "fp", "fn"))
        s_pt, s_p, s_t = (np.array([x[k] for x in s], float) for k in ("sum_pt", "sum_p", "sum_t"))
        res, iou, dice, siou, sdice = aggregate(tp, fp, fn, s_pt, s_p, s_t, eps, smooth, seed=seed)
        cmp = {}
        for name, key in (("iou", "hard_iou_mean_over_images"), ("dice", "hard_dice_mean_over_images"),
                          ("iou_pooled", "hard_iou_pooled_pixels"), ("dice_pooled", "hard_dice_pooled_pixels"),
                          ("iou_soft_repo", "soft_iou_mean_over_images_repo_IoULoss"),
                          ("dice_soft_repo", "soft_dice_mean_over_images_repo_DiceLoss")):
            ref = PAPER_TABLE3["Surya"]["iou" if name.startswith("iou") else "dice"]
            lo, hi = res[key]["ci95"]
            cmp[name] = {"ours": res[key]["estimate"], "ci95": [lo, hi], "paper_surya": ref,
                         "paper_inside_ci": bool(lo <= ref <= hi), "ours_minus_paper": res[key]["estimate"] - ref,
                         "paper_unet": PAPER_TABLE3["Unet"]["iou" if name.startswith("iou") else "dice"]}
        years = np.array([int(r["stratum"]) for r in recs])
        per_year = {int(y): {"n": int((years == y).sum()), "hard_iou_mean": float(iou[years == y].mean()),
                             "hard_dice_mean": float(dice[years == y].mean()),
                             "empty_masks": int(((tp + fn)[years == y] == 0).sum())} for y in np.unique(years)}
        sweep = {}
        for thr in THRESHOLDS:
            k = f"{thr:.1f}"
            a, b, c = (np.array([r["counts"]["hard"][k][q] for r in recs], float) for q in ("tp", "fp", "fn"))
            ii, dd = iou_dice_counts(a, b, c)
            sweep[k] = {"iou_mean_over_images": float(ii.mean()), "dice_mean_over_images": float(dd.mean()),
                        "iou_pooled": float(a.sum() / max(1, (a + b + c).sum())),
                        "dice_pooled": float(2 * a.sum() / max(1, (2 * a + b + c).sum()))}
        repo_diff = max(max(abs(r["repo_loss_check"]["soft_iou_from_IoULoss"] - si),
                            abs(r["repo_loss_check"]["soft_dice_from_DiceLoss"] - sd))
                        for r, si, sd in zip(recs, siou, sdice))
        compact = np.array([bool(is_compact_frame(r.get("stored_mb_per_channel") or "{}")) for r in recs])
        sens = {"n_compact_frames_scored": int(compact.sum()),
                "compact_frames": [r["timestamp"] for r, c in zip(recs, compact) if c]}
        if compact.any() and (~compact).sum() >= 2:
            k = np.flatnonzero(~compact)
            sens["hard_iou_mean_over_images_without_compact"] = {
                "estimate": float(iou[k].mean()), "ci95": bootstrap(lambda i: iou[k][i].mean(1), len(k), N_BOOT, seed),
                "n_images": int(len(k))}
            sens["hard_dice_mean_over_images_without_compact"] = {
                "estimate": float(dice[k].mean()), "ci95": bootstrap(lambda i: dice[k][i].mean(1), len(k), N_BOOT, seed),
                "n_images": int(len(k))}
        scored_blocks = {(int(r["stratum"]), int(r["block"])) for r in recs}
        summary["splits"][split] = {
            "n_images_scored": len(recs), "n_failed_records": len(failed),
            "n_blocks_planned": len(planned), "n_blocks_scored": len(scored_blocks),
            "blocks_not_scored": [list(b) for b in planned if b not in scored_blocks],
            "records_ignored_not_in_current_sample": int(n_foreign),
            "failed_records": [{"timestamp": r["timestamp"], "role": r["role"], "error": r.get("error", "")[:300]}
                               for r in failed],
            "compact_frame_sensitivity": sens,
            "roles": {k: int(v) for k, v in pd.Series([r["role"] for r in recs]).value_counts().items()},
            "time_range": [recs[0]["timestamp"], recs[-1]["timestamp"]],
            "empty_masks": int(((tp + fn) == 0).sum()), "empty_predictions": int(((tp + fp) == 0).sum()),
            "metrics": res, "comparison_with_paper": cmp, "per_year": per_year, "threshold_sweep": sweep,
            "max_abs_diff_soft_metric_vs_repo_loss_classes_float32": float(repo_diff),
            "mean_bce_with_logits": float(np.mean([r["bce_with_logits"] for r in recs])),
            "timing": {"forward_s_mean": float(np.mean([r["seconds"]["forward"] for r in recs])),
                       "forward_s_max": float(np.max([r["seconds"]["forward"] for r in recs])),
                       "sample_s_mean": float(np.mean([r["seconds"]["total"] for r in recs])),
                       "peak_rss_gb_max": float(np.max([r["peak_rss_gb"] for r in recs]))},
            "per_image": [{"timestamp": r["timestamp"], "stratum": r["stratum"], "target_px": r["counts"]["target_px"],
                           "pred_px": int(tp[i] + fp[i]), "hard_iou": float(iou[i]), "hard_dice": float(dice[i]),
                           "soft_iou": float(siou[i]), "soft_dice": float(sdice[i])} for i, r in enumerate(recs)],
        }
    env = os.path.join(out_dir, "environment.json")
    if os.path.exists(env):
        summary["environment"] = json.load(open(env))
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
    return summary


# ---------------------------------------------------------------------------------------------------- main
def get_sample(args, out_dir, cfg, scalers, force=False):
    sample_csv = os.path.join(out_dir, f"sample_{args.split}.csv")
    meta_json = os.path.join(out_dir, f"sampling_{args.split}.json")
    if not force and os.path.exists(sample_csv) and os.path.exists(meta_json):
        meta = json.load(open(meta_json))
        if (meta["n"], meta["seed"], meta["check_channels"]) == (args.n, args.seed, not args.no_channel_check):
            log(f"reusing {sample_csv}")
            return pd.read_csv(sample_csv), meta
    t0 = time.time()
    sample, meta = select_sample(args.split, args.n, args.seed, check_channels=not args.no_channel_check)
    meta["selection_seconds"] = round(time.time() - t0, 1)
    meta["dataset_join_check"] = check_joins(sample, cfg, scalers, args.split)
    meta["dataset_join_all_ok"] = all(x["ok"] for x in meta["dataset_join_check"])
    sample.to_csv(sample_csv, index=False)
    json.dump(meta, open(meta_json, "w"), indent=1, default=str)
    return sample, meta


def bandwidth_probe(sample):
    out = {}
    for name, url in (("s3_frame", f"{C.S3}/{sample['frame_key'].iloc[0]}"),
                      ("hf_tar", hf_url(LABEL_REPO, LABEL_REV, TAR_NAME))):
        t0 = time.time()
        r = requests.get(url, headers={"Range": "bytes=0-16777215"}, timeout=120)
        r.raise_for_status()
        out[name] = {"bytes": len(r.content), "seconds": round(time.time() - t0, 2),
                     "MB_per_s": round(len(r.content) / 1e6 / max(1e-3, time.time() - t0), 1)}
    return out


def dry_run(args, out_dir):
    from surya.utils.data import build_scalers
    cfg, cfg_inf, cfg_diff = load_configs()
    scalers = build_scalers(info=yaml.safe_load(open(os.path.join(ASSETS, "scalers.yaml"))))
    check = metric_self_test(out_dir)
    log(f"metric self-test passed (max abs diff {check['max_abs_diff_overall']:.2e})")
    sample, meta = get_sample(args, out_dir, cfg, scalers, force=True)
    prim = sample[sample.role == "primary"]
    bw = bandwidth_probe(sample)
    ck = requests.head(hf_url(CKPT_REPO, CKPT_REV, CKPT_NAME, kind="models"), allow_redirects=False, timeout=60)
    s3_bw = bw["s3_frame"]["MB_per_s"] * 1e6
    hf_bw = bw["hf_tar"]["MB_per_s"] * 1e6
    fb = prim["frame_bytes"].to_numpy(float)
    one_off = TAR_SIZE / hf_bw + CKPT_SIZE / hf_bw + SEC_MODEL
    per_sample = SEC_LOAD + SEC_FORWARD + SEC_SCORE
    dl = fb / s3_bw  # seconds to download each frame; frame j+1 downloads while image j is scored
    total_s = one_off + (dl[0] if len(dl) else 0.0) + len(dl) * per_sample + float(np.maximum(0, dl[1:] - per_sample).sum())
    est = {"assumptions_s": {"forward": SEC_FORWARD, "frame_read_and_normalise": SEC_LOAD, "scoring": SEC_SCORE,
                             "model_build_and_load": SEC_MODEL},
           "measured_bandwidth": bw,
           "one_off_minutes_tar_checkpoint_model": round(one_off / 60, 1),
           "per_sample_minutes": round(per_sample / 60, 2),
           "frame_download_minutes_each_mean": round(float(dl.mean()) / 60, 2) if len(dl) else 0,
           "frame_download_hidden_by_prefetch": bool(dl.max() < per_sample) if len(dl) else True,
           "total_minutes": round(total_s / 60, 1)}
    disk = {"frames_total_bytes_downloaded_over_run": int(fb.sum()),
            "frames_max_on_disk_bytes": int(max([fb[i] + fb[i + 1] for i in range(len(fb) - 1)] or [fb.max()])),
            "checkpoint_bytes": CKPT_SIZE, "checkpoint_head_x_linked_size": ck.headers.get("x-linked-size"),
            "masks_bytes_approx": int(len(sample) * 45000), "tar_streamed_not_stored_bytes": TAR_SIZE}
    disk["peak_disk_gb"] = round((disk["frames_max_on_disk_bytes"] + CKPT_SIZE + disk["masks_bytes_approx"]) / 1e9, 2)
    meta.update({"dry_run": True, "estimate": est, "disk": disk, "config_model_diff_config_vs_config_infer": cfg_diff,
                 "forward_passes_planned": int(len(prim)), "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}",
                 "labels": f"{LABEL_REPO}@{LABEL_REV}", "upstream_dir": UPSTREAM})
    json.dump(meta, open(os.path.join(out_dir, f"sampling_{args.split}.json"), "w"), indent=1, default=str)
    log(f"split {args.split}: {meta['candidate_info']['candidates']} candidates, {len(prim)} primaries + "
        f"{int((sample.role == 'reserve').sum())} reserves, {len(meta['skipped'])} skipped, "
        f"join check ok: {meta['dataset_join_all_ok']}")
    log(f"frames to download over the run: {fb.sum() / 1e9:.2f} GB (at most {disk['frames_max_on_disk_bytes'] / 1e9:.2f}"
        f" GB on disk at once); checkpoint {CKPT_SIZE / 1e9:.2f} GB; tar streamed {TAR_SIZE / 1e9:.2f} GB")
    log(f"estimated {est['total_minutes']} min ({len(prim)} forward passes); peak disk ~{disk['peak_disk_gb']} GB")
    print(sample[["role", "stratum", "block", "timestamp", "frame_bytes"]].to_string(index=False))


def run(args, out_dir):
    import torch
    from torch.utils.data import DataLoader
    from surya.utils.data import build_scalers
    sys.path.insert(0, TASK_DIR)
    from dataset import ArDSDataset
    from finetune import DiceLoss, IoULoss, custom_collate_fn

    cfg, cfg_inf, _ = load_configs()
    cfg_inf["data"]["scalers"] = yaml.safe_load(open(os.path.join(ASSETS, "scalers.yaml")))  # as infer.main
    cfg_inf["dtype"] = torch.bfloat16  # as infer.main; HelioSpectFormer ignores it ("not used")
    eps, smooth = float(cfg["model"]["iou"]["eps"]), float(cfg["model"]["dice"]["smooth"])
    scalers = build_scalers(info=yaml.safe_load(open(cfg["data"]["scalers_path"])))
    metric_self_test(out_dir)
    sample, meta = get_sample(args, out_dir, cfg, scalers)
    if not meta.get("dataset_join_all_ok", False):
        raise RuntimeError("dataset join check failed in the sample file; rerun --dry-run")
    recs_path = per_sample_path(out_dir, args.split)
    previous, n_foreign = records_in_sample(read_records(recs_path), sample)
    if n_foreign:
        log(f"ignoring {n_foreign} records in {recs_path} that are not rows of the current sample file")

    # frames left behind by an interrupted run are not known to the new FrameCache objects and would never be
    # deleted, so the two cache folders are emptied first (this keeps the bound of two frames on disk)
    for sub in ("a", "b"):
        shutil.rmtree(os.path.join(WORK, "frames", sub), ignore_errors=True)
    check_resources(args, sample)

    # one pass over the archive serves every sample file present in out_dir (test and validation)
    wanted = list(sample["mask_file_path"])
    for other in ("test", "validation"):
        p = os.path.join(out_dir, f"sample_{other}.csv")
        if other != args.split and os.path.exists(p):
            wanted += list(pd.read_csv(p)["mask_file_path"])
    manifest = ensure_masks(sorted(set(wanted)), os.path.join(WORK, "meta", "masks_manifest.json"))
    have_mask = {fp: os.path.exists(os.path.join(LABEL_DIR, fp)) for fp in sample["mask_file_path"]}

    recs_path = per_sample_path(out_dir, args.split)
    previous = read_records(recs_path)
    done = {t for t, r in previous.items() if r.get("status") == "ok"}
    ok_blocks = {(int(r["stratum"]), int(r["block"])) for r in previous.values() if r.get("status") == "ok"}
    blocks = {}
    for r in sample.to_dict("records"):
        blocks.setdefault((int(r["stratum"]), int(r["block"])), {})[r["role"]] = r
    work, notes, complete = [], [], 0
    for key, b in sorted(blocks.items()):
        if key in ok_blocks:  # one image per block: a block is finished once its primary or reserve is scored
            complete += 1
            continue
        prim, res = b.get("primary"), b.get("reserve")
        if prim is not None and have_mask[prim["mask_file_path"]]:
            work.append(prim)
        elif res is not None and have_mask[res["mask_file_path"]]:
            notes.append({"block": list(key), "primary": prim and prim["timestamp"], "used_reserve": res["timestamp"],
                          "reason": "primary mask not in data.tar.gz"})
            work.append(res)
        else:
            notes.append({"block": list(key), "reason": "no mask in data.tar.gz for primary or reserve"})
    pending = [r for r in work if r["timestamp"] not in done][: args.max_forward]
    log(f"{len(blocks)} blocks in the sample, {complete} already scored, {len(pending)} images to run now")
    if not pending:
        summarize(out_dir, args.seed)
        return

    ckpt = ensure_checkpoint()
    t0 = time.time()
    model, minfo = load_model(cfg_inf, ckpt)
    minfo["seconds_build_and_load"] = round(time.time() - t0, 1)
    env = {"torch": torch.__version__, "threads": torch.get_num_threads(), "precision": "float32, no autocast",
           "upstream_head": os.popen(f"git -C {UPSTREAM} rev-parse --short HEAD").read().strip(),
           "checkpoint": f"{CKPT_REPO}@{CKPT_REV}/{CKPT_NAME}", "checkpoint_sha256": CKPT_SHA256,
           "labels": f"{LABEL_REPO}@{LABEL_REV}", "tar_manifest": {k: v for k, v in manifest.items()
                                                                     if k not in ("members",)},
           "reserve_substitutions": notes, "model": minfo, "peak_rss_gb_after_model_load": C.peak_rss_gb()}
    json.dump(env, open(os.path.join(out_dir, "environment.json"), "w"), indent=1, default=str)
    log(f"model ready in {minfo['seconds_build_and_load']} s, peak RSS {C.peak_rss_gb()} GB")

    os.makedirs(os.path.join(out_dir, "arrays"), exist_ok=True)
    caches = [C.FrameCache(os.path.join(WORK, "frames", "a")), C.FrameCache(os.path.join(WORK, "frames", "b"))]
    hmi_m = cfg["data"]["channels"].index("hmi_m")
    pool = cf.ThreadPoolExecutor(1)
    fut = pool.submit(caches[0].ensure, [pd.Timestamp(pending[0]["timestamp"])])
    j = 0
    while j < len(pending):
        row = pending[j]
        t = pd.Timestamp(row["timestamp"])
        cache = caches[j % 2]
        rec = {k: row.get(k) for k in ("split", "role", "stratum", "block", "timestamp", "mask_file_path",
                                       "frame_key", "frame_bytes", "stored_mb_per_channel")}
        t_start = time.time()
        ds = loader = batch = x = logits = target = None
        try:
            dl_s = fut.result()
        except Exception as e:
            dl_s, err = None, e
        else:
            err = None
        if j + 1 < len(pending):  # prefetch the next frame into the other cache while this one is scored
            fut = pool.submit(caches[(j + 1) % 2].ensure, [pd.Timestamp(pending[j + 1]["timestamp"])])
        try:
            if err is not None:
                raise err
            t1 = time.time()
            ds = build_ar_dataset(cfg, scalers, args.split, t, cache.root, ArDSDataset)
            assert len(ds) == 1 and pd.Timestamp(ds.valid_indices[0]) == t, "dataset did not select the time stamp"
            loader = DataLoader(ds, batch_size=1, num_workers=0, shuffle=False, collate_fn=custom_collate_fn)
            batch, metadata = next(iter(loader))
            t2 = time.time()
            x = batch["ts"]
            rec["input"] = {"shape": list(x.shape), "dtype": str(x.dtype), "nonfinite": int((~torch.isfinite(x)).sum()),
                            "per_channel_normalised": {}}
            for ci, ch in enumerate(cfg["data"]["channels"]):  # mean / std / non-finite count after the scalers
                v = x[0, ci, 0].double()
                m = float(torch.nanmean(v))
                rec["input"]["per_channel_normalised"][ch] = [
                    round(m, 4), round(float(torch.nanmean((v - m) ** 2)) ** 0.5, 4), int((~torch.isfinite(v)).sum())]
                del v
            with torch.no_grad():
                logits = model(batch)
                if logits.ndim == 5:
                    logits = logits[:, 0]
            t3 = time.time()
            target = batch["forecast"].unsqueeze(1)  # as finetune.py: mask / 255, shape (1, 1, H, W)
            rec["target_values"] = sorted(float(v) for v in torch.unique(target))
            rec["output"] = {"shape": list(logits.shape), "dtype": str(logits.dtype),
                             "nonfinite": int((~torch.isfinite(logits)).sum()),
                             "logit_min": float(logits.min()), "logit_max": float(logits.max()),
                             "logit_mean": float(logits.mean())}
            with torch.no_grad():
                rec["counts"] = image_counts(logits, target)
                rec["repo_loss_check"] = {  # the authors' loss classes on the actual float32 tensors
                    "soft_iou_from_IoULoss": 1.0 - float(IoULoss(eps=eps)(logits, target)),
                    "soft_dice_from_DiceLoss": 1.0 - float(DiceLoss(smooth=smooth)(logits, target))}
                rec["bce_with_logits"] = float(torch.nn.BCEWithLogitsLoss()(logits, target))
                h = rec["counts"]["hard"]["0.5"]
                i5, d5 = iou_dice_counts(h["tp"], h["fp"], h["fn"])
                rec["hard_iou"], rec["hard_dice"] = float(i5), float(d5)
                pool8 = lambda a: torch.nn.functional.avg_pool2d(a.float(), 8)[0, 0]  # noqa: E731
                np.savez_compressed(os.path.join(out_dir, "arrays", f"{args.split}_{stamp(t)}.npz"),
                                    prob_512=(pool8(torch.sigmoid(logits)) * 255).round().byte().numpy(),
                                    pred_frac_512=(pool8((logits > 0).float()) * 255).round().byte().numpy(),
                                    mask_frac_512=(pool8(target) * 255).round().byte().numpy(),
                                    hmi_m_norm_512=pool8(x[:, hmi_m, 0][:, None]).half().numpy())
            try:
                rec["metadata"] = {k: [str(v) for v in np.ravel(metadata[k])] for k in metadata}
            except Exception:
                rec["metadata"] = str(metadata)[:500]
            rec["seconds"] = {"frame_download": dl_s, "dataset_and_load": round(t2 - t1, 1),
                              "forward": round(t3 - t2, 1), "score": round(time.time() - t3, 1),
                              "total": round(time.time() - t_start, 1)}
            rec["status"] = "ok"
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} IoU {rec['hard_iou']:.3f} Dice {rec['hard_dice']:.3f} "
                f"mask px {rec['counts']['target_px']} forward {rec['seconds']['forward']} s")
        except Exception as e:  # record, substitute the block's reserve once, carry on
            rec["status"] = "failed"
            rec["error"] = repr(e)[:2000]
            log(f"[{j + 1}/{len(pending)}] {row['timestamp']} FAILED: {e!r}")
            res = blocks[(int(row["stratum"]), int(row["block"]))].get("reserve")
            if (row["role"] == "primary" and res is not None and have_mask.get(res["mask_file_path"])
                    and res["timestamp"] not in done and all(p["timestamp"] != res["timestamp"] for p in pending)):
                pending.append(res)
                if j + 1 == len(pending) - 1:  # nothing was prefetched for it yet
                    fut = pool.submit(caches[(j + 1) % 2].ensure, [pd.Timestamp(res["timestamp"])])
        finally:
            ds = loader = batch = x = logits = target = None  # release the 0.9 GB input and the outputs
        rec["peak_rss_gb"] = C.peak_rss_gb()
        with open(recs_path, "a") as fp:
            fp.write(json.dumps(rec, default=str) + "\n")
        gc.collect()
        j += 1
    pool.shutdown(wait=True)
    for c in caches:
        c.clear()
    if args.delete_checkpoint:
        os.remove(ckpt)
    s = summarize(out_dir, args.seed)
    sp = s["splits"].get(args.split, {})
    if sp:
        m = sp["metrics"]
        log(f"{args.split}: n={sp['n_images_scored']} IoU {m['hard_iou_mean_over_images']['estimate']:.3f} "
            f"{m['hard_iou_mean_over_images']['ci95']} Dice {m['hard_dice_mean_over_images']['estimate']:.3f} "
            f"{m['hard_dice_mean_over_images']['ci95']} (paper 0.768 / 0.853)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="select and check the sample; no weights, frames or model")
    ap.add_argument("--split", choices=["test", "validation"], default="test")
    ap.add_argument("--n", type=int, default=None, help="images in the sample (default 30 test, 10 validation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "results", "Surya", "ar_segmentation"))
    ap.add_argument("--max-forward", type=int, default=40, help="forward passes allowed in this run")
    ap.add_argument("--no-channel-check", action="store_true", help="skip the netCDF header check of each frame")
    ap.add_argument("--delete-checkpoint", action="store_true", help="remove the 1.8 GB checkpoint at the end")
    ap.add_argument("--summarize-only", action="store_true", help="recompute summary.json from the jsonl files")
    args = ap.parse_args()
    args.n = args.n or DEFAULT_N[args.split]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(WORK, exist_ok=True)
    os.chdir(WORK)  # ArDSDataset reads masks from ./assets/... and writes logs/data/ relative to the cwd
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
