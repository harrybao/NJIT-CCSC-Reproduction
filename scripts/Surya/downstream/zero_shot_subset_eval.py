"""Zero-shot forecasting with the released Surya-1.0 on a sample of the paper's validation period (Table 9).

Table 9 of arXiv 2508.14112 v2 gives the MSE (normalised model units, all 13 channels) per lead time for the
released model ("5 hours" column) and for persistence, on the validation period (Jan 15-31 of 2011-2019). The
paper does not say how many initialisation times it used, and the period holds about 17,700 frames (roughly
10 TB), so this script evaluates a seeded random sample of initialisation times and reports the mean with a
bootstrap interval. Leads of 1 h and 2 h are evaluated (rollout_steps=1, as in tests/test_surya.py).

Per initialisation time t the inputs are the frames at t-60 and t and the targets are t+60 and t+120 (the
released config: time_delta_input_minutes [-60, 0], time_delta_target_minutes +60). Frames are read from the
public S3 bucket, which holds bitwise-identical data to the HF copies. The model, scalers, dataset class and
batch_step are the authors' code; precision is float32 (CPU bfloat16 autocast breaks this model, see
results/Surya/bf16_diagnosis). Persistence (the frame at t used as the forecast for every lead) is scored on the
same targets.

Sampling: only times on the full hour are used; 2 per year 2011-2019 plus 2 more drawn from all years
(20 in total by default), with all four frames present in the core-sdo validation index and on S3.

Usage:
  python zero_shot_subset_eval.py --dry-run --out results/Surya/zero_shot
  python zero_shot_subset_eval.py --out results/Surya/zero_shot [--n-per-year 2 --extra 2 --seed 0]
"""
import argparse, gc, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import surya_common as C  # noqa: E402  (sets HF_HUB_DISABLE_XET before huggingface_hub is imported)

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download

UPSTREAM = "/home/user/upstream/surya/Surya"
FM_DIR = "/home/user/work/surya_run/data/Surya-1.0"
TABLE9 = {"surya_5h_column": {1: 0.20172, 2: 0.22351}, "persistence": {1: 0.59868, 2: 0.70594}}
DELTAS = [-60, 0, 60, 120]


def sample_times(index_csv, n_per_year, extra, seed, check_s3=True):
    idx = pd.read_csv(index_csv)
    idx = idx[idx["present"] == 1]
    present = set(pd.to_datetime(idx["timestep"]))
    cands = sorted(t for t in present if t.minute == 0 and
                   all(t + pd.Timedelta(minutes=d) in present for d in DELTAS))
    rng = np.random.default_rng(seed)
    by_year = {}
    for t in cands:
        by_year.setdefault(t.year, []).append(t)
    picked, skipped = [], []

    def take(pool, k):
        order = list(rng.permutation(len(pool)))
        got = 0
        for i in order:
            t = pool[i]
            if t in picked:
                continue
            if check_s3:
                missing = [d for d in DELTAS if not C.s3_exists(t + pd.Timedelta(minutes=d))[0]]
                if missing:
                    skipped.append({"time": str(t), "reason": f"not on S3 at offsets {missing}"})
                    continue
            picked.append(t)
            got += 1
            if got == k:
                break

    for y in sorted(by_year):
        take(by_year[y], n_per_year)
    take(cands, extra)
    return sorted(picked), skipped, len(cands)


def bootstrap_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n, len(x)))].mean(1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-per-year", type=int, default=2)
    ap.add_argument("--extra", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default="/home/user/work/surya_ft/zero_shot/frames")
    a = ap.parse_args()
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)

    index_csv = hf_hub_download("nasa-ibm-ai4science/core-sdo", "valid_index_surya_1_0.csv", repo_type="dataset",
                                local_dir="/home/user/work/surya_ft/zero_shot/meta", token=None)
    sample_path = os.path.join(out, "sample_validation.csv")
    if os.path.exists(sample_path):
        times = [pd.Timestamp(t) for t in pd.read_csv(sample_path)["init_time"]]
        skipped, n_cands = None, None
    else:
        times, skipped, n_cands = sample_times(index_csv, a.n_per_year, a.extra, a.seed)
        pd.DataFrame({"init_time": [str(t) for t in times]}).to_csv(sample_path, index=False)
        json.dump({"candidates_on_the_hour_with_all_frames": n_cands, "skipped": skipped, "seed": a.seed,
                   "n_per_year": a.n_per_year, "extra": a.extra},
                  open(os.path.join(out, "sampling.json"), "w"), indent=1)
    sizes = [C.s3_exists(t + pd.Timedelta(minutes=d))[1] for t in times[:3] for d in DELTAS]
    print(f"{len(times)} initialisation times; ~{np.mean(sizes) / 1e9:.2f} GB per frame, 4 frames each; "
          f"about {len(times) * 2 * 135 / 60:.0f} min of forward passes")
    if a.dry_run:
        print("\n".join(str(t) for t in times))
        return

    sys.path.insert(0, UPSTREAM)
    sys.path.insert(0, os.path.join(UPSTREAM, "tests"))
    sys.path.insert(0, os.path.dirname(HERE))
    import test_surya as T
    import run_validation_test_grouped as R
    from surya.datasets.helio import HelioNetCDFDataset
    from surya.utils.data import build_scalers, custom_collate_fn
    from torch.utils.data import DataLoader

    config = yaml.safe_load(open(os.path.join(FM_DIR, "config.yaml")))
    scalers = build_scalers(info=yaml.safe_load(open(os.path.join(FM_DIR, "scalers.yaml"))))
    model = R.build_model(T, config)
    model.load_state_dict(torch.load(os.path.join(FM_DIR, "surya.366m.v1.pt"), map_location="cpu",
                                     weights_only=True), strict=True)
    model.requires_grad_(False)
    state = {"targets": None, "steps": []}

    def hook(m, i, o):
        s = len(state["steps"])
        state["steps"].append(F.mse_loss(o.detach().float(), state["targets"][:, :, s]).item())
    model.register_forward_hook(hook)

    cache = C.FrameCache(a.cache)
    log_path = os.path.join(out, "per_init.jsonl")
    done = {json.loads(l)["init_time"] for l in open(log_path)} if os.path.exists(log_path) else set()
    for t in times:
        if str(t) in done:
            continue
        need = [t + pd.Timedelta(minutes=d) for d in DELTAS]
        t_dl = cache.ensure(need)
        idx_path = C.write_index(need, os.path.join(out, "index_current.csv"))
        ds = HelioNetCDFDataset(
            index_path=idx_path, time_delta_input_minutes=config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=config["data"]["time_delta_target_minutes"],
            n_input_timestamps=len(config["data"]["time_delta_input_minutes"]), rollout_steps=1,
            channels=config["data"]["sdo_channels"], drop_hmi_probability=config["data"]["drop_hmi_probability"],
            num_mask_aia_channels=config["data"]["num_mask_aia_channels"],
            use_latitude_in_learned_flow=config["data"]["use_latitude_in_learned_flow"], scalers=scalers,
            phase="valid", pooling=config["data"]["pooling"], random_vert_flip=config["data"]["random_vert_flip"],
            sdo_data_root_path=cache.root)
        assert [pd.Timestamp(v) for v in ds.valid_indices] == [t], ds.valid_indices
        batch, meta = next(iter(DataLoader(ds, batch_size=1, shuffle=False, num_workers=0,
                                           collate_fn=custom_collate_fn)))
        assert pd.Timestamp(meta["timestamps_input"][0][-1]) == t  # guard against the dataset's skip-ahead
        targets = batch["forecast"]
        pers = [F.mse_loss(batch["ts"][:, :, -1], targets[:, :, s]).item() for s in range(2)]
        state["targets"], state["steps"] = targets, []
        t1 = time.time()
        with torch.no_grad():
            loss, returned = T.batch_step(model, batch, meta, "cpu", rollout=1)
        del returned
        rec = {"init_time": str(t), "surya_mse_lead_1h": state["steps"][0], "surya_mse_lead_2h": state["steps"][1],
               "batch_step_loss": loss, "persistence_mse_lead_1h": pers[0], "persistence_mse_lead_2h": pers[1],
               "seconds_forward": round(time.time() - t1, 1), "seconds_download": round(t_dl, 1),
               "peak_rss_gb": C.peak_rss_gb()}
        with open(log_path, "a") as fp:
            fp.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        state["targets"], state["steps"] = None, []
        del batch, targets, ds
        gc.collect()
    cache.clear()

    recs = [json.loads(l) for l in open(log_path)]
    summ = {"n": len(recs), "table9_reference": TABLE9}
    for k in ["surya_mse_lead_1h", "surya_mse_lead_2h", "persistence_mse_lead_1h", "persistence_mse_lead_2h"]:
        v = [r[k] for r in recs]
        summ[k] = {"mean": float(np.mean(v)), "median": float(np.median(v)), "bootstrap95": bootstrap_ci(v),
                   "min": float(np.min(v)), "max": float(np.max(v))}
    json.dump(summ, open(os.path.join(out, "summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
