"""Why does the Surya-1.0 validation loss blow up under CPU bfloat16 autocast?

On this machine, the test's float32 forward pass reproduces the reference loss closely, while the same pass under
torch.autocast(device_type="cpu", dtype=torch.bfloat16) gives a loss more than ten times larger. CUDA and CPU
autocast use different op lists: on CUDA, layer_norm and softmax (among others) are run in float32 and return
float32; on CPU they run in the dtype of their input. The released model code also has the explicit
softmax(..., dtype=torch.float32) calls commented out ("Changed when experimenting with mixed precision").

For one validation sample this script runs batch_step (the authors' test function) in five variants:
  fp32           no autocast (what the test does on a CPU-only machine)
  bf16           CPU autocast, as is
  bf16_ln32      CPU autocast, layer_norm computed in float32 and returned as float32 (the CUDA policy)
  bf16_sm32      CPU autocast, Tensor.softmax computed in float32 and returned as float32 (the CUDA policy)
  bf16_ln32_sm32 both
Forward hooks on the model's modules (to depth 4 by default) record, for every call, the output dtype and a fixed strided
subsample of 65,536 output values; each variant is compared with fp32 module by module (relative RMS error).

Usage: python diagnose_cpu_bf16.py --upstream ... --workdir ... --out results/Surya/bf16_diagnosis [--sample 1648]
"""
import argparse, contextlib, json, os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_validation_test_grouped as R  # noqa: E402  (download/build helpers of the main runner)


@contextlib.contextmanager
def fp32_layer_norm():
    orig = F.layer_norm

    def ln(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        with torch.autocast(device_type="cpu", enabled=False):
            return orig(input.float(), normalized_shape, None if weight is None else weight.float(),
                        None if bias is None else bias.float(), eps)
    F.layer_norm = ln
    try:
        yield
    finally:
        F.layer_norm = orig


@contextlib.contextmanager
def fp32_softmax():
    orig = torch.Tensor.softmax

    def sm(self, dim, dtype=None):
        with torch.autocast(device_type="cpu", enabled=False):
            return orig(self.float(), dim)
    torch.Tensor.softmax = sm
    try:
        yield
    finally:
        torch.Tensor.softmax = orig


def variant_ctx(name):
    stack = contextlib.ExitStack()
    if name != "fp32":
        stack.enter_context(torch.autocast(device_type="cpu", dtype=torch.bfloat16))
    if "ln32" in name:
        stack.enter_context(fp32_layer_norm())
    if "sm32" in name:
        stack.enter_context(fp32_softmax())
    return stack


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", default="1648")
    ap.add_argument("--variants", default="fp32,bf16,bf16_ln32,bf16_sm32,bf16_ln32_sm32")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--keep-files", action="store_true")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    upstream = os.path.abspath(a.upstream)
    sys.path.insert(0, upstream)
    sys.path.insert(0, os.path.join(upstream, "tests"))
    os.chdir(a.workdir)
    import test_surya as T
    from surya.utils.data import build_scalers, custom_collate_fn
    from torch.utils.data import DataLoader

    config = yaml.safe_load(open("data/Surya-1.0/config.yaml"))
    scalers = build_scalers(info=yaml.safe_load(open("data/Surya-1.0/scalers.yaml")))
    full_index = pd.read_csv(os.path.join(upstream, "tests/test_surya_index.csv"), index_col=0)
    files = R.files_for([a.sample])
    shas = R.lfs_sha256()
    local = []
    for fn in files:
        p = R.hf_hub_download(repo_id=R.DATA_REPO, repo_type="dataset", filename=fn,
                              local_dir="data/Surya-1.0_validation_data", token=None)
        assert R.sha256(p) == shas[fn], fn
        local.append(p)
    rows = full_index[full_index["path"].str.endswith(tuple(files))]
    idx_path = os.path.join(out, f"index_{a.sample}.csv")
    rows.to_csv(idx_path)
    ds = R.build_dataset(config, scalers, idx_path)
    assert [pd.Timestamp(t).strftime("%H%M") for t in ds.valid_indices] == [a.sample]
    batch_data, batch_metadata = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                                      collate_fn=custom_collate_fn)))
    k = sorted(h for g in R.GROUPS for h in g).index(a.sample)

    model = R.build_model(T, config)
    model.load_state_dict(torch.load("data/Surya-1.0/surya.366m.v1.pt", map_location="cpu", weights_only=True),
                          strict=True)
    model.requires_grad_(False)
    rec = {"calls": []}

    def make_hook(name):
        def hook(m, i, o):
            if not torch.is_tensor(o):
                return
            flat = o.detach().reshape(-1)
            step = max(1, flat.numel() // 65536)
            sub = flat[::step][:65536].float().clone()
            rec["calls"].append({"name": name, "cls": type(m).__name__, "dtype": str(o.dtype),
                                 "shape": list(o.shape), "sub": sub,
                                 "nonfinite": int((~torch.isfinite(flat)).sum().item())})
        return hook

    for name, m in model.named_modules():
        if name and name.count(".") < a.depth:
            m.register_forward_hook(make_hook(name))

    variants = a.variants.split(",")
    results, subs = {}, {}
    for v in variants:
        rec["calls"] = []
        t0 = time.time()
        with variant_ctx(v):
            with torch.no_grad():
                loss, _ = T.batch_step(model, batch_data, batch_metadata, "cpu", rollout=1)
        results[v] = {"loss": loss, "reference_loss": float(T.REFERENCE_LOSS[k]),
                      "delta_vs_reference": loss - float(T.REFERENCE_LOSS[k]), "seconds": round(time.time() - t0, 1)}
        subs[v] = rec["calls"]
        print(f"{v}: loss {loss:.6f} (reference {T.REFERENCE_LOSS[k]:.6f}) {results[v]['seconds']} s", flush=True)

    # module-by-module comparison with fp32 (call order is identical across variants)
    base = subs.get("fp32")
    table = []
    if base is not None:
        for v in variants:
            if v == "fp32":
                continue
            assert [c["name"] for c in subs[v]] == [c["name"] for c in base]
            for j, (c0, c1) in enumerate(zip(base, subs[v])):
                d = (c1["sub"] - c0["sub"]).norm() / (c0["sub"].norm() + 1e-30)
                table.append({"variant": v, "call": j, "module": c0["name"], "cls": c0["cls"],
                              "dtype_fp32_run": c0["dtype"], "dtype_variant": c1["dtype"],
                              "rel_rms_error_vs_fp32": float(d), "nonfinite_variant": c1["nonfinite"]})
    json.dump({"sample": a.sample, "results": results, "module_comparison": table,
               "torch": torch.__version__}, open(os.path.join(out, f"diagnosis_{a.sample}.json"), "w"), indent=1)
    for v in variants:
        if v == "fp32":
            continue
        rows_v = [r for r in table if r["variant"] == v]
        first = next((r for r in rows_v if r["rel_rms_error_vs_fp32"] > 0.05), None)
        print(v, "first module with >5% error:", first and (first["call"], first["module"], first["cls"],
              round(first["rel_rms_error_vs_fp32"], 4), first["dtype_variant"]), flush=True)
    if not a.keep_files:
        for p in local:
            os.remove(p)


if __name__ == "__main__":
    main()
