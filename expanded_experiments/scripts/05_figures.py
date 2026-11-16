"""
05_figures.py -- STEP 3b: publication figures from the analysis tables (IEEE sizes: 3.5 in column, 7.16 in double column).

    python3 scripts/05_figures.py        -> figures/*.png (600 dpi) and figures/*.pdf (vector)

All numbers are read from analysis/*.csv and results/raw_results.csv. Lines are mean change from each sample's OWN baseline with 95%
bootstrap CI (resampling samples); grey lines are the individual samples. No figure uses any value not in those files.
"""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

EXP = Path(__file__).resolve().parent.parent; fig_dir = EXP / "figures"; fig_dir.mkdir(exist_ok=True)
ch = pd.read_csv(EXP / "analysis" / "paired_change.csv"); raw = pd.read_csv(EXP / "results" / "raw_results.csv")
mm = pd.read_csv(EXP / "results" / "mismatched_reference.csv"); lv_df = pd.read_csv(EXP / "analysis" / "per_level_summary.csv")
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"], "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
                    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.4, "pdf.fonttype": 42})
COL = {"GREM": "#0072B2", "SSIM_whole": "#D55E00", "CP": "#009E73", "TF": "#CC79A7"}
EXPS = {"hue": ("primary_hue_informative", "Hue rotation (deg, effective)"), "blur": ("primary", "Gaussian blur kernel size"),
        "noise": ("primary", "Noise sigma (gray levels)"), "wrong_garment": ("primary", "Wrong-garment blend $\\alpha$")}
NEUTRAL = {"hue": 0.0, "blur": 1.0, "noise": 0.0, "wrong_garment": 0.0}
TITLE = {"hue": "(a) Hue", "blur": "(b) Blur", "noise": "(c) Noise", "wrong_garment": "(d) Wrong garment"}
panel_n = {}

def series(exp, met):
    sname = EXPS[exp][0]; d = ch[(ch.set == sname) & (ch.experiment == exp) & (ch.metric == met)].sort_values("level")
    lv = [NEUTRAL[exp]] + d.level.tolist(); mean = [0.0] + d.mean_change.tolist(); lo = [0.0] + d.ci95_lo.tolist(); hi = [0.0] + d.ci95_hi.tolist()
    return lv, np.array(mean), np.array(lo), np.array(hi), int(d.n.iloc[0])

def per_sample(exp, met):
    sname = EXPS[exp][0]
    ids = ch[(ch.set == sname)].index  # not used; recompute from raw below
    primary = sorted(raw[raw.analysis_set == "primary_candidate"].sample_id.unique())
    if exp == "hue":
        rep = pd.read_csv(EXP / "validation" / "dataset_report.csv").set_index("sample_id")
        primary = [s for s in primary if rep.loc[s, "hue_informative(mean_sat>=40)"]]
    W = raw[(raw.experiment == exp) & raw.sample_id.isin(primary)].pivot(index="sample_id", columns="level_requested", values=met)
    return W.sub(W[NEUTRAL[exp]], axis=0)

def eff_labels(exp, levels):
    if exp == "hue": return [{0.0: "0", 15.0: "16", 30.0: "30", 45.0: "44", 60.0: "60", 90.0: "90"}[l] for l in levels]
    return [f"{l:g}" for l in levels]

def draw(ax, exp, mets, show_samples=None, styles=None):
    lv = None
    for met in mets:
        lv, m, lo, hi, n = series(exp, met); x = np.arange(len(lv)); panel_n[exp] = n
        ax.fill_between(x, lo, hi, color=COL[met], alpha=0.18, linewidth=0)
        ax.plot(x, m, color=COL[met], ls=(styles or {}).get(met, "-"), marker="o", ms=3, label=met.replace("_whole", " (whole image)").replace("_", " "))
    if show_samples:
        P = per_sample(exp, show_samples)
        for s in P.index: ax.plot(np.arange(P.shape[1]), P.loc[s].values, color="0.65", lw=0.5, alpha=0.7, zorder=0)
    ax.axhline(0, color="k", lw=0.5); ax.set_xticks(np.arange(len(lv))); ax.set_xticklabels(eff_labels(exp, lv)); ax.set_xlabel(EXPS[exp][1])
    ax.set_title(f"{TITLE[exp]} (n={panel_n[exp]})", loc="left")

# ---- Figure 1: GREM and whole-image SSIM change under each degradation (the main figure) ----
fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.5), sharey=False)
for ax, exp in zip(axs.ravel(), EXPS): draw(ax, exp, ["GREM", "SSIM_whole"], show_samples="GREM", styles={"SSIM_whole": "--"})
for ax in axs[:, 0]: ax.set_ylabel("Change from own baseline")
h = [Line2D([0], [0], color=COL["GREM"], marker="o", ms=3, label="GREM (mean, 95% CI)"), Line2D([0], [0], color=COL["SSIM_whole"], ls="--", marker="o", ms=3, label="Whole-image SSIM (mean, 95% CI)"),
     Line2D([0], [0], color="0.65", lw=0.8, label="Individual samples (GREM)")]
fig.legend(handles=h, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01)); fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(fig_dir / "fig_main_grem_vs_ssim.png", dpi=600); fig.savefig(fig_dir / "fig_main_grem_vs_ssim.pdf"); plt.close(fig)

# ---- Figure 2: components ----
fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.5))
for ax, exp in zip(axs.ravel(), EXPS): draw(ax, exp, ["CP", "TF"], show_samples=None)
for ax in axs[:, 0]: ax.set_ylabel("Change from own baseline")
axs[0, 0].legend(loc="lower left", frameon=False); fig.tight_layout()
fig.savefig(fig_dir / "fig_components_cp_tf.png", dpi=600); fig.savefig(fig_dir / "fig_components_cp_tf.pdf"); plt.close(fig)

# ---- Figure 3: per-sample heatmaps of the change in GREM ----
rep = pd.read_csv(EXP / "validation" / "dataset_report.csv").set_index("sample_id")
prim = sorted(raw[raw.analysis_set == "primary_candidate"].sample_id.unique())
fig, axs = plt.subplots(1, 4, figsize=(7.16, 3.3), sharey=True)
for ax, exp in zip(axs, EXPS):
    W = raw[(raw.experiment == exp) & raw.sample_id.isin(prim)].pivot(index="sample_id", columns="level_requested", values="GREM").loc[prim]
    D = W.sub(W[NEUTRAL[exp]], axis=0).drop(columns=[NEUTRAL[exp]])
    im = ax.imshow(D.values, cmap="RdBu", vmin=-0.5, vmax=0.5, aspect="auto")
    ax.set_xticks(range(D.shape[1])); ax.set_xticklabels(eff_labels(exp, D.columns), rotation=0); ax.set_title(TITLE[exp][4:], loc="left")
    for i in range(D.shape[0]):
        for j in range(D.shape[1]): ax.text(j, i, f"{D.values[i, j]:+.2f}".replace("+0.", "+.").replace("-0.", "-."), ha="center", va="center", fontsize=4.8, color="k")
    ax.set_xlabel(EXPS[exp][1].split("(")[0].strip())
labels = [f"{s}{'†' if not rep.loc[s, 'hue_informative(mean_sat>=40)'] else ''}" for s in prim]
axs[0].set_yticks(range(len(prim))); axs[0].set_yticklabels(labels)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.01); cb.set_label("Change in GREM"); cb.ax.tick_params(labelsize=6)
fig.text(0.01, -0.06, "† low-saturation garment: hue rotation barely changes it (excluded from the hue summary).", fontsize=6)
fig.savefig(fig_dir / "fig_per_sample_heatmap.png", dpi=600, bbox_inches="tight"); fig.savefig(fig_dir / "fig_per_sample_heatmap.pdf", bbox_inches="tight"); plt.close(fig)

# ---- Figure 4: matched vs mismatched reference ----
prim_mm = mm[mm.sample_id.isin(prim)]
fig, ax = plt.subplots(figsize=(3.5, 2.6)); ids = sorted(prim_mm.sample_id.unique())
for i, s in enumerate(ids):
    g = prim_mm[prim_mm.sample_id == s]; o = g[~g.matched].GREM; m = g[g.matched].GREM.iloc[0]
    ax.plot([i, i], [o.min(), o.max()], color="0.7", lw=0.8, zorder=1); ax.scatter([i] * len(o), o, s=7, color="0.45", zorder=2)
    ax.scatter([i], [m], s=22, color=COL["GREM"], marker="D", zorder=3)
ax.set_xticks(range(len(ids))); ax.set_xticklabels(ids, rotation=60, fontsize=6); ax.set_ylabel("GREM"); ax.set_ylim(0.2, 0.95)
ax.legend(handles=[Line2D([0], [0], marker="D", color="w", markerfacecolor=COL["GREM"], ms=5, label="Own (matched) garment"), Line2D([0], [0], marker="o", color="w", markerfacecolor="0.45", ms=3.5, label="Other garments (mismatched)")],
          frameon=False, loc="lower right", fontsize=6); fig.tight_layout()
fig.savefig(fig_dir / "fig_mismatched_reference.png", dpi=600); fig.savefig(fig_dir / "fig_mismatched_reference.pdf"); plt.close(fig)
print("figures written:", sorted(p.name for p in fig_dir.glob("*.png")))
