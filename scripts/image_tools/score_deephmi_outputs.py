"""DeepHMI: re-score the authors' shipped super-resolution outputs (sr_out/) against
the GST/NIRIS ground truth in the shipped test FITS, and compare with the HMI input
on the same 720x720 grid (the "no super-resolution" baseline). Field components are
derived exactly as in DeepHMI/test.py:
  HMI: Bz = B cos(inc_deg), Bt = B sin(inc_deg)      (FITS planes 1, 3, 5)
  GST: Bz = B cos(pi - inc_rad), Bt = B sin(pi - inc_rad)  (planes 0, 2, 4)
A full CPU re-run of the 1,000-step diffusion sampler was not feasible
(about 67 s per U-Net pass on this machine => >11 h for two components).
Usage (cwd = DeepHMI repo): python score_deephmi_outputs.py out.json
"""
import json
import sys

import numpy as np
from astropy.io import fits
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

f = fits.open("dataset_test/niris_hmi_bto_azi_inc_150622_164315.fits")[0].data.astype(np.float64)
hmi_inc = np.clip(f[5], 0, 180) * np.pi / 180
gst_inc = np.pi - np.clip(f[4], 0, np.pi)
truth = {"bz": f[0] * np.cos(gst_inc), "bt": f[0] * np.sin(gst_inc)}
hmi = {"bz": f[1] * np.cos(hmi_inc), "bt": f[1] * np.sin(hmi_inc)}
res = {}
for comp in ["bz", "bt"]:
    y = truth[comp]
    p = fits.open(f"sr_out/{comp}_150622_164315.fits")[0].data.astype(np.float64)
    rng = float(y.max() - y.min())
    row = {}
    for tag, img in [("HMI_input_same_grid", hmi[comp]), ("DeepHMI_author_output", p)]:
        row[tag] = {"PCC": float(np.corrcoef(y.ravel(), img.ravel())[0, 1]),
                    "RMSE_G": float(np.sqrt(np.mean((y - img) ** 2))),
                    "SSIM": float(structural_similarity(y, img, data_range=rng)),
                    "PSNR": float(peak_signal_noise_ratio(y, img, data_range=rng))}
    res[comp] = row
    print(comp, json.dumps(row))
json.dump(res, open(sys.argv[1], "w"), indent=1)
