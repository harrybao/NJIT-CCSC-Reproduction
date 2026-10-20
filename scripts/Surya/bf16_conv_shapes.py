"""Which convolution shapes give wrong bfloat16 results on this CPU? (random weights, 64 x 64 inputs)

Compares a bfloat16 Conv2d through oneDNN (PyTorch's default CPU path) and with oneDNN disabled against a float64
reference. bfloat16 rounding alone gives a relative RMS error of about 0.003; values near 1 mean a wrong result.
Usage: python bf16_conv_shapes.py out.json
"""
import json, platform, sys

import torch
import torch.nn.functional as F

SHAPES = [(26, 1280, 16), (32, 1280, 16), (13, 1280, 16), (26, 64, 16), (16, 16, 16), (3, 64, 16),
          (26, 1280, 8), (26, 1280, 4), (26, 1280, 2), (26, 1280, 3)]


def rel(y, ref):
    return float((y.double() - ref).norm() / ref.norm())


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def main():
    torch.manual_seed(0)
    out = {"torch": torch.__version__, "cpu_capability": torch.backends.cpu.get_cpu_capability(),
           "cpu_model": cpu_model(), "cases": []}
    for cin, cout, k in SHAPES:
        stride = k if k != 3 else 1
        x = torch.randn(1, cin, 64, 64)
        w = torch.randn(cout, cin, k, k) * 0.02
        b = torch.randn(cout) * 0.02
        ref = F.conv2d(x.double(), w.double(), b.double(), stride=stride)
        on = rel(F.conv2d(x.bfloat16(), w.bfloat16(), b.bfloat16(), stride=stride), ref)
        with torch.backends.mkldnn.flags(enabled=False):
            off = rel(F.conv2d(x.bfloat16(), w.bfloat16(), b.bfloat16(), stride=stride), ref)
        out["cases"].append({"in_channels": cin, "out_channels": cout, "kernel": k, "stride": stride,
                             "rel_error_bf16_onednn": on, "rel_error_bf16_without_onednn": off})
        print(f"in={cin:3d} out={cout:5d} kernel={k:2d} stride={stride:2d}  oneDNN {on:.4f}  without oneDNN {off:.4f}")
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
