"""Run DeepGeoCME test.py under torch 1.12 (CPU).

torch<1.13 has no `weights_only` argument; we drop it in a thin wrapper.
The checkpoint and tensors are the authors' GitHub release files, verified
against assets_manifest.json SHA-256 hashes by test.py itself.
Usage (cwd = DeepGeoCME working copy): python run_test_torch112.py [test.py args]
"""
import os
import runpy
import sys

import torch

_load = torch.load


def _compat_load(*a, **k):
    k.pop("weights_only", None)
    return _load(*a, **k)


torch.load = _compat_load
sys.argv = ["test.py"] + sys.argv[1:]
sys.path.insert(0, os.getcwd())
runpy.run_path("test.py", run_name="__main__")
