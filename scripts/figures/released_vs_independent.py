"""Dumbbell chart: skill under the released evaluation vs. an independent,
leakage-aware re-evaluation of the same model (values from results/*.json).
Usage: python released_vs_independent.py out.png
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_REL, C_IND = "#6da7ec", "#184f95"   # blue ramp 300 / 600 (ordinal pair, validated)

rows = [  # label, metric, released, independent, note
    ("MIST (≥M, 24 h)", "TSS", 0.906, 0.197, "random split of overlapping windows → 2015–18 data"),
    ("Dst Transformer, 1 h", "R²", 0.942, 0.063, "label-post-processed outputs → raw network (MC mean)"),
    ("Dst Transformer, 6 h", "R²", 0.932, 0.338, "label-post-processed outputs → raw network (MC mean)"),
    ("SolarFlareNet (≥C, 24 h)", "TSS", 0.792, 0.697, "all bundled test rows → rows not in training file"),
    ("LSTM 2019 (≥M5.0)", "TSS", 0.877, 0.825, "threshold chosen on test → on 2014 validation"),
    ("FlareML RF (4-class)", "macro TSS", 0.338, 0.246, "random 10-fold → AR-grouped 10-fold"),
    ("LSTM 2019 (≥C)", "TSS", 0.589, 0.584, "threshold chosen on test → on 2014 validation"),
]
fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=160)
fig.subplots_adjust(left=0.36, right=0.97)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for i, (lab, met, a, b, note) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.plot([b, a], [y, y], color="#c3c2b7", lw=2, zorder=1, solid_capstyle="round")
    ax.scatter([a], [y], s=70, color=C_REL, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter([b], [y], s=70, color=C_IND, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.text(a + 0.018, y, f"{a:.2f}", va="center", ha="left", fontsize=8.5, color=INK2)
    ax.text(b - 0.018, y, f"{b:.2f}", va="center", ha="right", fontsize=8.5, color=INK2)
    tr = ax.get_yaxis_transform()  # x in axes fraction, y in data
    ax.text(-0.03, y + 0.13, lab, va="center", ha="right", fontsize=9.5, color=INK, transform=tr)
    ax.text(-0.03, y - 0.22, f"{met}: {note}", va="center", ha="right", fontsize=7.5, color=INK2, transform=tr)
ax.set_xlim(-0.02, 1.05); ax.set_ylim(-0.7, len(rows) - 0.3)
ax.set_yticks([]); ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.tick_params(axis="x", colors=INK2, labelsize=8.5)
for x in [0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    ax.axvline(x, color=GRID, lw=0.8, zorder=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Skill score (TSS or R², higher is better)", color=INK2, fontsize=9)
ax.scatter([], [], s=70, color=C_REL, label="released evaluation (as shipped)")
ax.scatter([], [], s=70, color=C_IND, label="independent, leakage-aware re-evaluation")
leg = ax.legend(loc="lower left", bbox_to_anchor=(-0.55, 1.0), ncol=2, frameon=False, fontsize=8.5)
for t in leg.get_texts():
    t.set_color(INK)
fig.suptitle("Same model, different evaluation: how much of the reported skill survives?", x=0.02, ha="left",
             fontsize=11.5, color=INK, y=1.04)
fig.savefig(sys.argv[1], facecolor=SURFACE, bbox_inches="tight")
