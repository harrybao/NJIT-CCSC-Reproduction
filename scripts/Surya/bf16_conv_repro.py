"""Minimal check of Surya's patch-embedding convolution in bfloat16 on this CPU.

The module-by-module comparison (diagnose_cpu_bf16.py) shows that under CPU bfloat16 autocast the very first
layer, Conv2d(26, 1280, kernel_size=16, stride=16) applied to a 1 x 26 x 4096 x 4096 input, already differs from
float32 by a relative RMS of about 1, i.e. its output is wrong rather than slightly rounded. This script applies
the same convolution, with the released weights, to random inputs of several sizes and compares
  fp32, bf16 via autocast, bf16 with explicit casts, and bf16 with oneDNN (mkldnn) disabled
against a float64 reference. Relative RMS error ||y - y_ref|| / ||y_ref||; bfloat16 rounding alone gives ~1e-2 or less.

Usage: python bf16_conv_repro.py /path/to/surya.366m.v1.pt out.json
"""
import json, sys, time

import torch
import torch.nn.functional as F


def rel(y, ref):
    y = y.double()
    return float((y - ref).norm() / ref.norm())


def main():
    ckpt, out = sys.argv[1], sys.argv[2]
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    w = sd["embedding.patch_embed.proj.weight"].float().clone()
    b = sd["embedding.patch_embed.proj.bias"].float().clone()
    del sd
    res = {"torch": torch.__version__, "mkldnn_available": torch.backends.mkldnn.is_available(),
           "weight_shape": list(w.shape), "weight_dtype_in_checkpoint": None, "cases": []}
    cfg = torch.__config__.show()
    res["onednn_line"] = [l.strip() for l in cfg.splitlines() if "oneDNN" in l or "MKLDNN" in l or "CPU capability" in l]
    g = torch.Generator().manual_seed(0)
    for H in (64, 256, 1024, 2048, 4096):
        x = torch.randn(1, w.shape[1], H, H, generator=g)
        ref = F.conv2d(x.double(), w.double(), b.double(), stride=16)
        case = {"H": H}
        t = time.time(); case["fp32"] = rel(F.conv2d(x, w, b, stride=16), ref); case["t_fp32"] = round(time.time() - t, 2)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            t = time.time(); y = F.conv2d(x, w, b, stride=16); case["t_bf16_autocast"] = round(time.time() - t, 2)
        case["bf16_autocast"] = rel(y, ref); case["bf16_autocast_out_dtype"] = str(y.dtype)
        case["bf16_explicit"] = rel(F.conv2d(x.bfloat16(), w.bfloat16(), b.bfloat16(), stride=16), ref)
        with torch.backends.mkldnn.flags(enabled=False):
            case["bf16_explicit_mkldnn_off"] = rel(F.conv2d(x.bfloat16(), w.bfloat16(), b.bfloat16(), stride=16), ref)
        # rounding the inputs and weights to bf16 but convolving in float64: the error bf16 storage alone causes
        case["bf16_rounding_only"] = rel(F.conv2d(x.bfloat16().double(), w.bfloat16().double(), b.bfloat16().double(),
                                                  stride=16), ref)
        res["cases"].append(case)
        print(json.dumps(case), flush=True)
        del x, ref, y
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cases"}, indent=1))


if __name__ == "__main__":
    main()
