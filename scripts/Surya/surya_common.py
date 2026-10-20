"""Shared helpers for running released Surya models on small samples of the benchmark data, on a CPU box.

Inputs come from the public bucket s3://nasa-surya-bench (no requester-pays). Its frames decode to exactly the
same values as the Hugging Face copies (all 13 channels bitwise equal for 2014-01-07 16:00, see
results/Surya/s3_vs_hf_frames.json); only the compression differs.

Nothing here changes how the authors' code computes anything; it only fetches files, keeps the disk use bounded and
writes index files in the format HelioNetCDFDataset expects.
"""
import os, resource, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # the xet backend keeps a second copy of every file in a cache

import pandas as pd
import requests

S3 = "https://nasa-surya-bench.s3.amazonaws.com"
CHANNELS = ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600",
            "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v"]


def s3_key(ts):
    """pd.Timestamp -> 'YYYY/MM/YYYYMMDD_HHMM.nc' (the bucket layout and the path style of the core-sdo indexes)."""
    ts = pd.Timestamp(ts)
    return ts.strftime("%Y/%m/%Y%m%d_%H%M.nc")


def s3_exists(ts, timeout=60):
    r = requests.head(f"{S3}/{s3_key(ts)}", timeout=timeout)
    return r.status_code == 200, int(r.headers.get("Content-Length", 0))


def fetch_s3(ts, root, retries=4):
    """Download one frame to root/YYYY/MM/..., verify the byte count against Content-Length, return the path."""
    key = s3_key(ts)
    dest = os.path.join(root, key)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    for attempt in range(retries):
        try:
            with requests.get(f"{S3}/{key}", stream=True, timeout=120) as r:
                r.raise_for_status()
                expected = int(r.headers["Content-Length"])
                if os.path.exists(dest) and os.path.getsize(dest) == expected:
                    return dest
                tmp = dest + ".part"
                with open(tmp, "wb") as fp:
                    for chunk in r.iter_content(1 << 24):
                        fp.write(chunk)
            if os.path.getsize(tmp) != expected:
                raise IOError(f"short read for {key}: {os.path.getsize(tmp)} of {expected}")
            os.replace(tmp, dest)
            return dest
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** (attempt + 1))


class FrameCache:
    """Keeps only the frames the current sample needs on disk.

    ensure(timestamps) downloads the missing ones and deletes every other cached frame first, so the disk
    holds at most one sample's worth of frames (about 0.6 GB per frame on S3).
    """

    def __init__(self, root):
        self.root = os.path.abspath(root)
        os.makedirs(self.root, exist_ok=True)
        self.held = {}

    def ensure(self, timestamps):
        want = {s3_key(t) for t in timestamps}
        for key in list(self.held):
            if key not in want:
                try:
                    os.remove(self.held.pop(key))
                except FileNotFoundError:
                    pass
        t0 = time.time()
        for t in timestamps:
            key = s3_key(t)
            if key not in self.held:
                self.held[key] = fetch_s3(t, self.root)
        return time.time() - t0

    def clear(self):
        self.ensure([])


def write_index(timestamps, path):
    """Index CSV in the format of core-sdo's *_index_surya_1_0.csv: path (relative to sdo_data_root_path),
    timestep, present. Pass the FrameCache root as sdo_data_root_path."""
    ts = sorted(pd.Timestamp(t) for t in timestamps)
    df = pd.DataFrame({"path": [s3_key(t) for t in ts], "timestep": [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts],
                       "present": 1})
    df.to_csv(path)  # same layout as the core-sdo index files (unnamed integer column first)
    return path


def peak_rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)
