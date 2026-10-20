"""Do the SDO frames in the public S3 bucket decode to the same values as the Hugging Face copies?

The downstream evaluations here read their inputs from s3://nasa-surya-bench (public, no requester-pays), because
the Hugging Face core-sdo validation/test data only exist as 10 GB tar parts. The S3 files are about 12% smaller
than the HF files of the same timestamp, so this script compares them channel by channel after decoding.

Usage: python compare_s3_hf_frames.py out.json 20140107_1600 [more stamps ...]
Each stamp YYYYMMDD_HHMM must exist in nasa-ibm-ai4science/Surya-1.0_validation_data (HF) and on S3.
"""
import json, os, sys, tempfile

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import hdf5plugin  # noqa: F401  (registers the HDF5 compression filters the HF files use)
import numpy as np
import requests
import xarray as xr
from huggingface_hub import hf_hub_download

S3 = "https://nasa-surya-bench.s3.amazonaws.com"


def fetch(url, dest):
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(dest, "wb") as fp:
            for chunk in r.iter_content(1 << 24):
                fp.write(chunk)
    return dest


def main():
    out, stamps = sys.argv[1], sys.argv[2:]
    work = tempfile.mkdtemp(prefix="s3hf_", dir=os.environ.get("SCRATCH", None))
    res = []
    for st in stamps:
        y, m = st[:4], st[4:6]
        s3_path = fetch(f"{S3}/{y}/{m}/{st}.nc", os.path.join(work, f"s3_{st}.nc"))
        hf_path = hf_hub_download("nasa-ibm-ai4science/Surya-1.0_validation_data", f"{st}.nc", repo_type="dataset",
                                  local_dir=os.path.join(work, "hf"), token=None)
        a, b = xr.open_dataset(s3_path, engine="h5netcdf"), xr.open_dataset(hf_path, engine="h5netcdf")
        rec = {"stamp": st, "bytes_s3": os.path.getsize(s3_path), "bytes_hf": os.path.getsize(hf_path),
               "vars_s3": sorted(a.data_vars), "vars_hf": sorted(b.data_vars),
               "attrs_equal": a.attrs == b.attrs, "channels": {}}
        enc = {}
        for v in sorted(set(a.data_vars) & set(b.data_vars)):
            x, z = a[v].values, b[v].values
            same_nan = bool(np.array_equal(np.isnan(x), np.isnan(z)))
            fin = ~np.isnan(x) & ~np.isnan(z)
            rec["channels"][v] = {"shape": list(x.shape), "dtype_s3": str(x.dtype), "dtype_hf": str(z.dtype),
                                  "bitwise_equal": bool(np.array_equal(x, z, equal_nan=True)),
                                  "nan_mask_equal": same_nan,
                                  "max_abs_diff": float(np.abs(x[fin] - z[fin]).max()) if fin.any() else None}
            enc[v] = {"s3": {k: str(a[v].encoding.get(k)) for k in ("compression", "compression_opts", "chunksizes",
                                                                   "shuffle", "zlib", "complevel")},
                      "hf": {k: str(b[v].encoding.get(k)) for k in ("compression", "compression_opts", "chunksizes",
                                                                   "shuffle", "zlib", "complevel")}}
            del x, z
        rec["encoding_first_channel"] = enc[sorted(enc)[0]] if enc else None
        rec["all_channels_bitwise_equal"] = all(c["bitwise_equal"] for c in rec["channels"].values())
        a.close(); b.close()
        os.remove(s3_path); os.remove(hf_path)
        res.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "channels"}, indent=1), flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
