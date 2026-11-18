"""06_paper_figure.py -- the single column-width figure used in the paper (3.35 in wide). Reads analysis/paired_change.csv and results/raw_results.csv."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
EXP = Path(__file__).resolve().parent.parent
ch = pd.read_csv(EXP / "analysis" / "paired_change.csv"); raw = pd.read_csv(EXP / "results" / "raw_results.csv")
rep = pd.read_csv(EXP / "validation" / "dataset_report.csv").set_index("sample_id")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 6.5, "axes.labelsize": 6.5, "axes.titlesize": 7,
                    "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.linewidth": 0.5, "lines.linewidth": 1.1, "pdf.fonttype": 42})
BLUE, ORANGE = "#0072B2", "#D55E00"
E = {"hue": ("primary_hue_informative", "Hue rotation (deg)", "(a) Hue", {0.0: "0", 15.0: "16", 30.0: "30", 45.0: "44", 60.0: "60", 90.0: "90"}, 0.0),
     "blur": ("primary", "Blur kernel size", "(b) Blur", None, 1.0), "noise": ("primary", "Noise sigma", "(c) Noise", None, 0.0),
     "wrong_garment": ("primary", "Wrong-garment weight", "(d) Wrong garment", None, 0.0)}
prim = sorted(raw[(raw.analysis_set == "primary_candidate")].sample_id.unique())
fig, axs = plt.subplots(2, 2, figsize=(3.35, 3.35))
for ax, (exp, (sname, xl, title, lab, neu)) in zip(axs.ravel(), E.items()):
    ids = [s for s in prim if (exp != "hue" or rep.loc[s, "hue_informative(mean_sat>=40)"])]
    W = raw[(raw.experiment == exp) & raw.sample_id.isin(ids)].pivot(index="sample_id", columns="level_requested", values="GREM"); W = W.sub(W[neu], axis=0)
    for s in W.index: ax.plot(range(W.shape[1]), W.loc[s].values, color="0.72", lw=0.4, alpha=0.8, zorder=0)
    for met, col, ls in (("GREM", BLUE, "-"), ("SSIM_whole", ORANGE, "--")):
        d = ch[(ch.set == sname) & (ch.experiment == exp) & (ch.metric == met)].sort_values("level")
        m = np.r_[0, d.mean_change.values]; lo = np.r_[0, d.ci95_lo.values]; hi = np.r_[0, d.ci95_hi.values]; x = np.arange(len(m))
        ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0); ax.plot(x, m, color=col, ls=ls, marker="o", ms=2.2)
    lv = [neu] + sorted(ch[(ch.set == sname) & (ch.experiment == exp)].level.unique())
    ax.set_xticks(range(len(lv))); ax.set_xticklabels([lab[l] if lab else f"{l:g}" for l in lv]); ax.axhline(0, color="k", lw=0.4)
    ax.set_xlabel(xl); ax.set_title(f"{title} (n={len(ids)})", loc="left", pad=2)
for ax in axs[:, 0]: ax.set_ylabel("Change from own baseline")
fig.legend(handles=[Line2D([0], [0], color=BLUE, marker="o", ms=2.2, label="GREM"), Line2D([0], [0], color=ORANGE, ls="--", marker="o", ms=2.2, label="Whole-image SSIM"),
                    Line2D([0], [0], color="0.72", lw=0.8, label="Single outfits (GREM)")], loc="lower center", ncol=3, frameon=False, fontsize=5.8, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.045, 1, 1), pad=0.4, w_pad=0.6, h_pad=0.7)
fig.savefig(EXP / "figures" / "fig_paper_main.png", dpi=600); fig.savefig(EXP / "figures" / "fig_paper_main.pdf"); print("saved", fig.get_size_inches())
