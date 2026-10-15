"""Summarise the grouped Surya-1.0 validation run and draw an AIA 94 figure like the one in tests/test_surya.py.

Reads results/Surya/validation_test/per_sample_<mode>.jsonl (one line per sample) and writes
  summary.json   per-sample losses, means, the test's pass criterion (|mean - reference mean| < 1e-4),
                 the persistence baseline, and per-channel MSE averaged over the samples
  summary.md     the same as a table
  aia94_validation.png
Usage: python summarise_validation.py results/Surya/validation_test
"""
import glob, json, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CHANNELS = ["aia94", "aia131", "aia171", "aia193", "aia211", "aia304", "aia335", "aia1600",
            "hmi_m", "hmi_bx", "hmi_by", "hmi_bz", "hmi_v"]
REFERENCE_MEAN_TOL = 1.0e-4  # assert in tests/test_surya.py


def load(d):
    recs = {}
    for p in sorted(glob.glob(os.path.join(d, "per_sample_*.jsonl"))):
        for line in open(p):
            r = json.loads(line)
            t = r["reference_time"]
            recs.setdefault(t, {}).update(r)
    return [recs[t] for t in sorted(recs)]


def main():
    d = sys.argv[1]
    recs = load(d)
    modes = [m for m in ("cuda_autocast_noop", "cpu_bf16") if any(m in r for r in recs)]
    out = {"n_samples": len(recs), "samples": [], "modes": {}}
    for r in recs:
        row = {"reference_time": r["reference_time"], "reference_loss": r["reference_loss"],
               "persistence_loss": r["persistence_loss"], "persistence_mse_per_step": r["persistence_mse_per_step"]}
        for m in modes:
            if m in r:
                row[m] = r[m]["loss"]
        out["samples"].append(row)
    ref = np.array([r["reference_loss"] for r in recs])
    for m in modes:
        have = [r for r in recs if m in r]
        loss = np.array([r[m]["loss"] for r in have])
        refm = np.array([r["reference_loss"] for r in have])
        per_chan = np.array([r[m]["mse_per_channel_per_step"] for r in have])  # n x 2 x 13
        out["modes"][m] = {
            "n": len(have), "mean_loss": float(loss.mean()), "mean_reference": float(refm.mean()),
            "mean_delta": float(loss.mean() - refm.mean()),
            "test_criterion_pass": bool(len(have) == 8 and abs(loss.mean() - refm.mean()) < REFERENCE_MEAN_TOL),
            "max_abs_per_sample_delta": float(np.abs(loss - refm).max()),
            "mean_seconds_two_forward_passes": float(np.mean([r[m]["seconds_two_forward_passes"] for r in have])),
            "peak_rss_gb": float(max(r[m]["peak_rss_gb_so_far"] for r in have)),
            "mse_per_step_mean": [float(x) for x in np.mean([r[m]["mse_per_step_fp32_recomputed"] for r in have], 0)],
            "mse_per_channel_mean": {f"t+{60 * (s + 1)}min": dict(zip(CHANNELS, map(float, per_chan[:, s].mean(0))))
                                     for s in range(per_chan.shape[1])},
        }
    pers = np.array([r["persistence_loss"] for r in recs])
    out["persistence"] = {"mean_loss": float(pers.mean()),
                          "mse_per_step_mean": [float(x) for x in np.mean([r["persistence_mse_per_step"] for r in recs], 0)]}
    out["reference_mean_all8"] = float(ref.mean()) if len(recs) == 8 else None
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)

    lines = ["| Reference time | Reference loss | " + " | ".join(modes) + " | Persistence |",
             "|---|---|" + "---|" * len(modes) + "---|"]
    for row in out["samples"]:
        lines.append(f"| {row['reference_time']} | {row['reference_loss']:.6f} | "
                     + " | ".join(f"{row[m]:.6f}" if m in row else "-" for m in modes)
                     + f" | {row['persistence_loss']:.4f} |")
    lines.append(f"| mean | {ref.mean():.6f} | " + " | ".join(f"{out['modes'][m]['mean_loss']:.6f}" for m in modes)
                 + f" | {pers.mean():.4f} |")
    open(os.path.join(d, "summary.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(json.dumps({m: {k: v for k, v in out["modes"][m].items() if k != "mse_per_channel_mean"} for m in modes},
                     indent=1))

    # figure: one row per sample; inputs, fp32 predictions, targets (normalised units, 512 x 512 means)
    try:
        import sunpy.visualization.colormaps as sunpy_cm
        cmap = sunpy_cm.cmlist["sdoaia94"]
    except Exception:
        cmap = "magma"
    files = sorted(glob.glob(os.path.join(d, "aia94_512", "sample_*_cuda_autocast_noop.npz")))
    if files:
        cols = ["input_t-60", "input_t", "cuda_autocast_noop_pred_t+60", "target_t+60",
                "cuda_autocast_noop_pred_t+120", "target_t+120"]
        titles = ["input t-60", "input t", "Surya t+60", "observed t+60", "Surya t+120", "observed t+120"]
        arrs = [np.load(f) for f in files]
        allv = np.concatenate([a[c].ravel() for a in arrs for c in cols])
        vmin, vmax = np.quantile(allv, [0.01, 0.995])
        fig, ax = plt.subplots(len(arrs), len(cols), figsize=(2.2 * len(cols), 2.2 * len(arrs)))
        ax = np.atleast_2d(ax)
        for i, (f, a) in enumerate(zip(files, arrs)):
            hhmm = os.path.basename(f).split("_")[1]
            for j, c in enumerate(cols):
                ax[i, j].imshow(a[c], origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
                ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(titles[j], fontsize=9)
            ax[i, 0].set_ylabel(f"t = {hhmm[:2]}:{hhmm[2:]}", fontsize=9)
        fig.suptitle("Surya-1.0 on the repository's validation samples, AIA 94 (normalised units, 2014-01-07)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(d, "aia94_validation.png"), dpi=90)


if __name__ == "__main__":
    main()
