"""Run the SolarUnet demo pipeline with deterministic (sorted) directory listing.

The upstream code iterates over os.listdir(), whose order is filesystem-
dependent (Linux ext4/overlay returned frame_1, frame_3, frame_2), which pairs
predicted masks with the wrong FITS frames before tracking. We patch
os.listdir to return sorted names; the algorithm itself is unchanged.
Usage (cwd = SolarUnet working copy): python run_solarunet_sorted.py
"""
import os
import sys

_orig = os.listdir
os.listdir = lambda p=".": sorted(_orig(p))
sys.path.insert(0, os.getcwd())

from solarunet import model_predicting, post_processing, pre_processing  # noqa: E402
from magnetic_tracking import magnetic_tracking  # noqa: E402
from statistics_analysis import analysis  # noqa: E402

pre_processing("data/exmaple_data_preprocess/3_class/", "data/exmaple_data_preprocess/2_class/")
model_predicting("data/magnetic/", "results/predicted_mask/", pretrain=True)
post_processing()
magnetic_tracking("results/processed_data_for_tracking/", "results/tracking_results/")
analysis()
