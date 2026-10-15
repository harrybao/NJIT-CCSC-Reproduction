"""Run the Surya-1.0 validation test (tests/test_surya.py in NASA-IMPACT/Surya) on a CPU machine with little disk.

The test scores 8 samples whose reference times are 2014-01-07 16:00, 16:12, ..., 17:24. With the released
config (inputs at t-60 and t, rollout_steps=1) each sample reads the frames at t-60, t, t+60 and t+120, so the
test needs all 23 files from 15:00 to 19:24 (15.4 GB). That does not fit on this disk at once. Each file is used
by at most two samples (t and t+60), so the samples are scored in five groups; a group's files are downloaded,
checked against the SHA-256 on Hugging Face, scored, and deleted before the next group.

Everything that affects the numbers is the authors' code, imported unchanged from the clone: the model class and
its constructor arguments (copied from the test fixture), the weights, the scalers, HelioNetCDFDataset with the
fixture's arguments, custom_collate_fn, batch_step and its MSE loss. Differences from the test, none of which
change the arithmetic of a forward pass:
  * the dataset is built per group from the rows of tests/test_surya_index.csv for that group's files;
  * the DataLoader uses num_workers=0 instead of 2 (memory: one sample is 3.5 GB in float32);
  * each batch is scored in up to two precision modes (see MODES).

A forward hook computes per-channel MSE of each model output as it is produced, and a persistence baseline
(the last input frame used as the forecast, in the same normalised units) is scored on the same targets.
Downsampled AIA 94 images (512 x 512) are saved for a figure like the one the test draws.

Memory (15 GB box): one sample is 3.5 GB of float32 inputs and targets and the float32 weights are 1.5 GB.
To keep the bfloat16 pass inside the limit, the parameters are set to requires_grad=False, so autocast does not
cache bfloat16 copies of every weight (under torch.no_grad this does not change any arithmetic), and each
precision mode is run in its own process (see run_validation_test.sh). Peak RSS is recorded per sample.

Usage:
  python run_validation_test_grouped.py --upstream /path/to/Surya --workdir /path/with/data/Surya-1.0 \
      --out results/Surya/validation_test --modes cuda_autocast_noop,cpu_bf16 [--groups 0,1] [--keep-files]
"""
import argparse, hashlib, json, os, resource, sys, time

# The xet download backend keeps a chunk cache (up to 10 GB by default) under ~/.cache/huggingface/xet,
# i.e. a second copy of every file, which this disk cannot hold. Use plain HTTP downloads instead.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
import pandas as pd
import requests
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download

DATA_REPO = "nasa-ibm-ai4science/Surya-1.0_validation_data"
GROUPS = [  # reference times (HHMM on 2014-01-07); samples t and t+60 share three files
    ["1600", "1700"],
    ["1612", "1712"],
    ["1624", "1724"],
    ["1636"],
    ["1648"],
]
# cuda_autocast_noop: exactly the test's context. On a machine without CUDA, autocast(device_type="cuda") is
#   disabled, so this is a float32 forward pass; it is what `pytest tests/test_surya.py` does on a CPU.
# cpu_bf16: the same context but with autocast on the CPU in bfloat16, i.e. the closest CPU analogue of the
#   GPU bfloat16 run the reference losses presumably come from.
MODES = {
    "cuda_autocast_noop": lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16),
    "cpu_bf16": lambda: torch.autocast(device_type="cpu", dtype=torch.bfloat16),
}


def hhmm_to_ts(hhmm):
    return pd.Timestamp(f"2014-01-07 {hhmm[:2]}:{hhmm[2:]}:00")


def files_for(group):
    out = []
    for hhmm in group:
        t = hhmm_to_ts(hhmm)
        for dt in (-60, 0, 60, 120):
            s = (t + pd.Timedelta(minutes=dt)).strftime("20140107_%H%M.nc")
            if s not in out:
                out.append(s)
    return sorted(out)


def lfs_sha256():
    r = requests.get(f"https://huggingface.co/api/datasets/{DATA_REPO}/tree/main", timeout=60)
    r.raise_for_status()
    return {x["path"]: (x.get("lfs") or {}).get("oid") for x in r.json()}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def build_model(T, config):
    # Same arguments as the `model` fixture in tests/test_surya.py.
    from surya.models.helio_spectformer import HelioSpectFormer
    return HelioSpectFormer(
        img_size=config["model"]["img_size"],
        patch_size=config["model"]["patch_size"],
        in_chans=len(config["data"]["sdo_channels"]),
        embed_dim=config["model"]["embed_dim"],
        time_embedding={"type": "linear", "time_dim": len(config["data"]["time_delta_input_minutes"])},
        depth=config["model"]["depth"],
        n_spectral_blocks=config["model"]["n_spectral_blocks"],
        num_heads=config["model"]["num_heads"],
        mlp_ratio=config["model"]["mlp_ratio"],
        drop_rate=config["model"]["drop_rate"],
        dtype=torch.bfloat16,
        window_size=config["model"]["window_size"],
        dp_rank=config["model"]["dp_rank"],
        learned_flow=config["model"]["learned_flow"],
        use_latitude_in_learned_flow=config["model"]["learned_flow"],
        init_weights=False,
        checkpoint_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        rpe=config["model"]["rpe"],
        ensemble=config["model"]["ensemble"],
        finetune=config["model"]["finetune"],
    )


def build_dataset(config, scalers, index_path):
    # Same arguments as the `dataset` fixture in tests/test_surya.py.
    from surya.datasets.helio import HelioNetCDFDataset
    return HelioNetCDFDataset(
        index_path=index_path,
        time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
        time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
        n_input_timestamps=len(config["data"]["time_delta_input_minutes"]),
        rollout_steps=1,
        channels=config["data"]["sdo_channels"],
        drop_hmi_probability=config["data"]["drop_hmi_probability"],
        num_mask_aia_channels=config["data"]["num_mask_aia_channels"],
        use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"],
        scalers=scalers,
        phase="valid",
        pooling=config["data"]["pooling"],
        random_vert_flip=config["data"]["random_vert_flip"],
    )


def down(x, f=8):
    # x: (H, W) float tensor -> (H/f, W/f) mean-pooled numpy float32
    return F.avg_pool2d(x[None, None].float(), f)[0, 0].numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--modes", default="cuda_autocast_noop,cpu_bf16")
    ap.add_argument("--groups", default=",".join(str(i) for i in range(len(GROUPS))))
    ap.add_argument("--keep-files", action="store_true")
    a = ap.parse_args()

    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "aia94_512"), exist_ok=True)
    upstream = os.path.abspath(a.upstream)
    sys.path.insert(0, upstream)
    sys.path.insert(0, os.path.join(upstream, "tests"))
    os.chdir(a.workdir)  # the authors' index uses paths relative to the repository root
    import test_surya as T  # authors' test module: batch_step, REFERENCE_LOSS, SDO_CHANNELS
    from surya.utils.data import build_scalers, custom_collate_fn
    from torch.utils.data import DataLoader

    modes = a.modes.split(",")
    groups = [int(g) for g in a.groups.split(",")]
    config = yaml.safe_load(open("data/Surya-1.0/config.yaml"))
    scalers = build_scalers(info=yaml.safe_load(open("data/Surya-1.0/scalers.yaml")))
    full_index = pd.read_csv(os.path.join(upstream, "tests/test_surya_index.csv"), index_col=0)
    ref_times = [hhmm for g in GROUPS for hhmm in g]
    ref_order = sorted(ref_times)  # dataloader order in the test (shuffle=False, index sorted by time)

    model = build_model(T, config)
    weights = torch.load("data/Surya-1.0/surya.366m.v1.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=True)
    del weights
    n_params = sum(p.numel() for p in model.parameters())
    model.requires_grad_(False)
    state = {"targets": None, "outputs": []}

    def hook(module, inputs, output):
        # score each output as soon as it exists; keep only small summaries
        s = len(state["outputs"])
        tgt = state["targets"][:, :, s]
        o = output.detach()
        per_chan = [F.mse_loss(o[:, c].float(), tgt[:, c]).item() for c in range(tgt.shape[1])]
        state["outputs"].append({"mse_fp32": F.mse_loss(o.float(), tgt).item(), "per_channel": per_chan,
                                 "dtype": str(o.dtype), "aia94_512": down(o[0, T.SDO_CHANNELS.index("aia94")])})

    model.register_forward_hook(hook)
    shas = lfs_sha256()
    env = {"torch": torch.__version__, "threads": torch.get_num_threads(), "n_parameters": n_params,
           "cuda_available": torch.cuda.is_available(), "model_training_flag": model.training}
    json.dump(env, open(os.path.join(out, "environment.json"), "w"), indent=1)
    log_path = os.path.join(out, "per_sample.jsonl")

    for gi in groups:
        group = GROUPS[gi]
        files = files_for(group)
        local = []
        t0 = time.time()
        for fn in files:
            p = hf_hub_download(repo_id=DATA_REPO, repo_type="dataset", filename=fn,
                                local_dir="data/Surya-1.0_validation_data", token=None)
            digest = sha256(p)
            if digest != shas.get(fn):
                raise RuntimeError(f"SHA-256 mismatch for {fn}: {digest} vs {shas.get(fn)}")
            local.append(p)
        t_dl = time.time() - t0
        rows = full_index[full_index["path"].str.endswith(tuple(files))]
        assert len(rows) == len(files), (len(rows), files)
        idx_path = os.path.join(out, f"index_group{gi}.csv")
        rows.to_csv(idx_path)
        dataset = build_dataset(config, scalers, idx_path)
        got = [pd.Timestamp(t).strftime("%H%M") for t in dataset.valid_indices]
        assert got == group, (got, group)
        loader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=0, pin_memory=False,
                            drop_last=False, collate_fn=custom_collate_fn)
        for bi, (batch_data, batch_metadata) in enumerate(loader):
            ref_hhmm = pd.Timestamp(batch_metadata["timestamps_input"][0][-1]).strftime("%H%M")
            assert ref_hhmm == group[bi], (ref_hhmm, group[bi])  # guard against the dataset's silent skip-ahead
            k = ref_order.index(ref_hhmm)
            x_last = batch_data["ts"][:, :, -1]
            targets = batch_data["forecast"]
            rec = {"group": gi, "reference_time": f"2014-01-07T{ref_hhmm[:2]}:{ref_hhmm[2:]}",
                   "sample_index_in_test": k, "reference_loss": float(T.REFERENCE_LOSS[k]),
                   "timestamps_input": [str(t) for t in batch_metadata["timestamps_input"][0]],
                   "timestamps_targets": [str(t) for t in batch_metadata["timestamps_targets"][0]],
                   "download_seconds_group": round(t_dl, 1)}
            # persistence baseline in the same normalised units (MSE averaged over the two lead times)
            pers = [F.mse_loss(x_last, targets[:, :, s]).item() for s in range(targets.shape[2])]
            rec["persistence_mse_per_step"] = pers
            rec["persistence_loss"] = sum(pers) / len(pers)
            c94 = T.SDO_CHANNELS.index("aia94")
            imgs = {"input_t-60": down(batch_data["ts"][0, c94, 0]), "input_t": down(batch_data["ts"][0, c94, 1]),
                    "target_t+60": down(targets[0, c94, 0]), "target_t+120": down(targets[0, c94, 1])}
            for mode in modes:
                state["targets"], state["outputs"] = targets, []
                t1 = time.time()
                with MODES[mode]():
                    with torch.no_grad():
                        loss, _ = T.batch_step(model, batch_data, batch_metadata, "cpu", rollout=1)
                secs = time.time() - t1
                outs = state["outputs"]
                assert len(outs) == 2
                rec[mode] = {"loss": loss, "delta_vs_reference": loss - float(T.REFERENCE_LOSS[k]),
                             "mse_per_step_fp32_recomputed": [o["mse_fp32"] for o in outs],
                             "mse_per_channel_per_step": [o["per_channel"] for o in outs],
                             "output_dtype": outs[0]["dtype"], "seconds_two_forward_passes": round(secs, 1),
                             "peak_rss_gb_so_far": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
                imgs[f"{mode}_pred_t+60"] = outs[0]["aia94_512"]
                imgs[f"{mode}_pred_t+120"] = outs[1]["aia94_512"]
                print(f"[{ref_hhmm}] {mode}: loss {loss:.6f} ref {T.REFERENCE_LOSS[k]:.6f} "
                      f"delta {loss - T.REFERENCE_LOSS[k]:+.2e} ({secs:.0f} s, peak RSS "
                      f"{rec[mode]['peak_rss_gb_so_far']} GB)", flush=True)
            tag = "_".join(modes)
            np.savez_compressed(os.path.join(out, "aia94_512", f"sample_{ref_hhmm}_{tag}.npz"), **imgs)
            with open(log_path.replace(".jsonl", f"_{tag}.jsonl"), "a") as fp:
                fp.write(json.dumps(rec) + "\n")
            state["targets"], state["outputs"] = None, []
            del batch_data, targets, x_last
        if not a.keep_files:
            for p in local:
                os.remove(p)
        print(f"group {gi} done; files {'kept' if a.keep_files else 'deleted'}", flush=True)


if __name__ == "__main__":
    main()
