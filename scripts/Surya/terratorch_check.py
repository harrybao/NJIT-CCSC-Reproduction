"""Load the released Surya checkpoints through TerraTorch's registry and see what actually gets loaded.

The IBM blog says Surya is available "via IBM's TerraTorch library". TerraTorch 1.2.13 (with the `surya` extra,
which installs terratorch-surya 0.1.0) registers five Surya backbones. Its load_weights() downloads the checkpoint
from Hugging Face and calls model.load_state_dict(..., strict=False), ignoring the result. This script builds each
registry model exactly as TerraTorch does (registry defaults, pretrained=True), but serves the checkpoint from a
local copy instead of downloading it again, and records:
  * the missing and unexpected keys that strict=False hides,
  * how many model tensors were actually changed by loading (compared before and after).
With --forward, it also runs the base model on the repository test's sample 2014-01-07 16:48 (frames from S3,
TerraTorch's own HelioNetCDFDataset) and reports the 1-hour MSE, once with TerraTorch's default attention settings
(num_heads 8, dp_rank 8) and once with the values in the released config.yaml (num_heads 16, dp_rank 4). The
upstream model gives 0.231124 on this sample in float32 (results/Surya/bf16_diagnosis/conv32).

Run with the TerraTorch environment (Python 3.12), e.g.
  /home/user/envs/terratorch/bin/python terratorch_check.py --name heliofm_backbone_surya \
      --ckpt /home/user/work/surya_run/data/Surya-1.0/surya.366m.v1.pt --out results/Surya/terratorch/base.json --forward
"""
import argparse, gc, json, os, sys, time

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
import huggingface_hub
import torch
import torch.nn.functional as F


def fingerprint(model):
    return {k: (tuple(v.shape), float(v.float().sum()), float(v.float().abs().sum())) for k, v in model.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--forward", action="store_true")
    ap.add_argument("--fm-dir", default="/home/user/work/surya_run/data/Surya-1.0")
    ap.add_argument("--cache", default="/home/user/work/surya_ft/terratorch/frames")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)

    real_download = huggingface_hub.hf_hub_download
    served = []

    def local_download(**kw):
        served.append(kw)
        return a.ckpt
    huggingface_hub.hf_hub_download = local_download

    import terratorch  # noqa: F401  (registers the backbones)
    from importlib.metadata import version
    from terratorch.registry import BACKBONE_REGISTRY

    captured = {}
    orig_load = torch.nn.Module.load_state_dict

    def recording_load(self, state_dict, strict=True, assign=False):
        res = orig_load(self, state_dict, strict=strict, assign=assign)
        captured.update({"strict": strict, "n_checkpoint_tensors": len(state_dict),
                         "missing_keys": list(res.missing_keys), "unexpected_keys": list(res.unexpected_keys)})
        return res
    torch.nn.Module.load_state_dict = recording_load

    # same random initialisation for both builds, so "changed by loading" is a fair comparison
    torch.manual_seed(0)
    before = fingerprint(BACKBONE_REGISTRY.build(f"terratorch_{a.name}", pretrained=False))
    torch.manual_seed(0)
    model = BACKBONE_REGISTRY.build(f"terratorch_{a.name}", pretrained=True)
    after = fingerprint(model)
    torch.nn.Module.load_state_dict = orig_load
    changed = [k for k in after if before.get(k) != after[k]]
    res = {
        "registry_name": f"terratorch_{a.name}", "terratorch": version("terratorch"),
        "terratorch_surya": version("terratorch-surya"), "torch": torch.__version__,
        "checkpoint_requested_from_hf": served, "local_checkpoint": a.ckpt, "model_class": type(model).__name__,
        "n_model_tensors": len(after), "n_model_tensors_changed_by_loading": len(changed),
        "n_checkpoint_tensors": captured.get("n_checkpoint_tensors"), "strict": captured.get("strict"),
        "n_missing_keys": len(captured.get("missing_keys", [])), "n_unexpected_keys": len(captured.get("unexpected_keys", [])),
        "missing_keys_first10": captured.get("missing_keys", [])[:10],
        "unexpected_keys_first10": captured.get("unexpected_keys", [])[:10],
    }
    print(json.dumps(res, indent=1), flush=True)
    del before

    if a.forward:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import pandas as pd
        import yaml
        import surya_common as C
        from terratorch.datasets.helio import HelioNetCDFDataset
        huggingface_hub.hf_hub_download = real_download

        config = yaml.safe_load(open(os.path.join(a.fm_dir, "config.yaml")))
        t = pd.Timestamp("2014-01-07 16:48")
        need = [t + pd.Timedelta(minutes=d) for d in (-60, 0, 60)]
        cache = C.FrameCache(a.cache)
        cache.ensure(need)
        # TerraTorch's dataset has no data-root argument, so the index holds absolute paths
        idx = os.path.join(os.path.dirname(os.path.abspath(a.out)), "index_terratorch.csv")
        pd.DataFrame({"path": [os.path.join(cache.root, C.s3_key(x)) for x in need],
                      "timestep": [x.strftime("%Y-%m-%d %H:%M:%S") for x in need], "present": 1}).to_csv(idx)
        ds = HelioNetCDFDataset(index_path=idx, time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
                                time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
                                n_input_timestamps=2, rollout_steps=0,
                                scalers=os.path.join(a.fm_dir, "scalers.yaml"),
                                channels=config["data"]["sdo_channels"], phase="valid")
        assert [pd.Timestamp(v) for v in ds.valid_indices] == [t], ds.valid_indices
        sample, meta = ds[0]
        assert pd.Timestamp(meta["timestamps_input"][-1]) == t
        x = {"ts": torch.as_tensor(sample["ts"])[None], "time_delta_input": torch.as_tensor(sample["time_delta_input"])[None]}
        target = torch.as_tensor(sample["forecast"])[None][:, :, 0]
        res["forward_sample"] = str(t)
        model.requires_grad_(False)
        t1 = time.time()
        with torch.no_grad():
            res["mse_lead_1h_registry_defaults"] = F.mse_loss(model(x), target).item()
        res["seconds_registry_defaults"] = round(time.time() - t1, 1)
        print("registry defaults:", res["mse_lead_1h_registry_defaults"], flush=True)
        del model
        gc.collect()
        torch.manual_seed(0)
        huggingface_hub.hf_hub_download = local_download
        m2 = BACKBONE_REGISTRY.build(f"terratorch_{a.name}", pretrained=True, num_heads=config["model"]["num_heads"],
                                     dp_rank=config["model"]["dp_rank"])
        m2.requires_grad_(False)
        t1 = time.time()
        with torch.no_grad():
            res["mse_lead_1h_config_yaml_heads"] = F.mse_loss(m2(x), target).item()
        res["seconds_config_yaml_heads"] = round(time.time() - t1, 1)
        res["upstream_float32_reference"] = 0.2311243861913681
        print("config.yaml heads:", res["mse_lead_1h_config_yaml_heads"], flush=True)
        cache.clear()
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
