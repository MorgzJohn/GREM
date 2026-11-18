"""
04_analyze.py -- STEP 3a: the pre-registered statistical analysis (config/analysis_plan.json, version 2).

    python3 scripts/04_analyze.py            (a few seconds; writes analysis/*.csv and analysis/analysis_log.txt)

Reads results/raw_results.csv, results/mismatched_reference.csv, results/expected_failures.csv, validation/dataset_report.csv.
Everything is exploratory. Nothing here is a claim about human perception or about GREM being better than SSIM.
"""
import hashlib, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

EXP = Path(__file__).resolve().parent.parent
res, out = EXP / "results", EXP / "analysis"; out.mkdir(exist_ok=True)
plan_bytes = (EXP / "config" / "analysis_plan.json").read_bytes(); plan_hash = hashlib.sha256(plan_bytes).hexdigest()
meta = json.load(open(res / "run_meta.json"))
if meta["plan_sha256"] != plan_hash: sys.exit("STOP: analysis_plan.json differs from the plan used for the run.")
if "ALL CHECKS PASSED" not in (res / "verification_report.txt").read_text(): sys.exit("STOP: verification did not pass.")
B, SEED = plan_bytes and json.loads(plan_bytes)["statistics"]["bootstrap"]["resamples"], json.loads(plan_bytes)["statistics"]["bootstrap"]["seed"]
raw = pd.read_csv(res / "raw_results.csv"); mm = pd.read_csv(res / "mismatched_reference.csv"); ef = pd.read_csv(res / "expected_failures.csv")
rep = pd.read_csv(EXP / "validation" / "dataset_report.csv").set_index("sample_id")
EXPS = ["hue", "blur", "noise", "wrong_garment"]; NEUTRAL = {"hue": 0.0, "blur": 1.0, "noise": 0.0, "wrong_garment": 0.0}
TESTED = ["CP", "TF", "GF", "GREM", "SSIM_whole", "SSIM_garment_region"]       # NP and non-garment SSIM are invariant by construction
ALLM = TESTED + ["NP", "SSIM_nongarment_region"]
info = raw.drop_duplicates("sample_id").set_index("sample_id")[["analysis_set", "mask_quality", "baseline_quality", "person_file", "garment_file"]]
info["hue_informative"] = rep["hue_informative(mean_sat>=40)"].reindex(info.index)
info["mask_fraction"] = rep["mask_fraction_of_image"].reindex(info.index)
primary = info[(info.analysis_set == "primary_candidate") & info.mask_quality.isin(["good", "partial"])].index.tolist()
stress = info[info.analysis_set == "mask_stress_subset"].index.tolist()
SETS = {"primary": primary, "primary_hue_informative": [s for s in primary if info.loc[s, "hue_informative"]],
        "primary_hue_NOT_informative": [s for s in primary if not info.loc[s, "hue_informative"]],
        "sens_mask_good": [s for s in primary if info.loc[s, "mask_quality"] == "good"],
        "sens_baseline_good": [s for s in primary if info.loc[s, "baseline_quality"] == "good"],
        "sens_excl_non_3x4_garment(IMG_7461)": [s for s in primary if info.loc[s, "garment_file"] != "IMG_7461.jpg"]}
for p in sorted(info.loc[primary, "person_file"].unique()): SETS[f"loo_person_excl_{p}"] = [s for s in primary if info.loc[s, "person_file"] != p]
for g in sorted(info.loc[primary, "garment_file"].unique()): SETS[f"log_garment_excl_{g}"] = [s for s in primary if info.loc[s, "garment_file"] != g]
for q in ("good", "partial", "bad"): SETS[f"stress_mask_{q}"] = [s for s in stress if info.loc[s, "mask_quality"] == q]

def wide(ids, exp, metric):
    d = raw[(raw.experiment == exp) & raw.sample_id.isin(ids)].pivot(index="sample_id", columns="level_requested", values=metric)
    return d.loc[[s for s in ids if s in d.index]]
def rng_for(key): return np.random.default_rng(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") ^ SEED)
def boot_ci(x, key, fn=np.mean):
    x = np.asarray(x, float); n = len(x)
    if n < 2: return (np.nan, np.nan)
    idx = rng_for(key).integers(0, n, (B, n)); v = np.apply_along_axis(fn, 1, x[idx]) if fn is not np.mean else x[idx].mean(axis=1)
    return tuple(np.percentile(v, [2.5, 97.5]))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(np.where(np.isnan(p), 2, p)); m = int(np.sum(~np.isnan(p))); adj = np.full(len(p), np.nan); run = 0.0
    for rank, i in enumerate(o[:m]): run = max(run, (m - rank) * p[i]); adj[i] = min(1.0, run)
    return adj

# ---------------- per-level summary + paired change ----------------
level_rows, change_rows = [], []
for sname, ids in SETS.items():
    if not ids: continue
    for exp in EXPS:
        for met in ALLM:
            W = wide(ids, exp, met)
            if W.empty: continue
            base = W[NEUTRAL[exp]]
            for lv in W.columns:
                v = W[lv]; q1, q3 = np.percentile(v, [25, 75])
                level_rows.append(dict(set=sname, experiment=exp, metric=met, level=lv, n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan, median=v.median(), q1=q1, q3=q3))
                if lv == NEUTRAL[exp]: continue
                d = (v - base).values; inv = np.all(d == 0)
                lo, hi = boot_ci(d, f"{sname}|{exp}|{met}|{lv}")
                try: p = np.nan if inv else stats.wilcoxon(d, alternative="two-sided").pvalue
                except ValueError: p = np.nan
                change_rows.append(dict(set=sname, experiment=exp, metric=met, level=lv, n=len(d), mean_change=d.mean(), ci95_lo=lo, ci95_hi=hi, median_change=np.median(d),
                                        pct_strict_decrease=100 * np.mean(d < 0), pct_non_increase=100 * np.mean(d <= 0), pct_strict_increase=100 * np.mean(d > 0),
                                        invariant=inv, wilcoxon_p=p))
lv_df, ch = pd.DataFrame(level_rows), pd.DataFrame(change_rows)
ch["wilcoxon_p_holm"] = np.nan
for (s, e), g in ch[ch.metric.isin(TESTED)].groupby(["set", "experiment"]): ch.loc[g.index, "wilcoxon_p_holm"] = holm(g.wilcoxon_p.values)
lv_df.to_csv(out / "per_level_summary.csv", index=False); ch.to_csv(out / "paired_change.csv", index=False)

# ---------------- per-sample Spearman between degradation intensity and each metric ----------------
cor_rows = []
for sname in ("primary", "primary_hue_informative", "primary_hue_NOT_informative", "stress_mask_good", "stress_mask_partial", "stress_mask_bad"):
    for exp in EXPS:
        for met in TESTED:
            W = wide(SETS[sname], exp, met)
            if W.empty: continue
            rhos = np.array([stats.spearmanr(W.columns.values, W.loc[s].values)[0] if W.loc[s].nunique() > 1 else np.nan for s in W.index])
            ok = rhos[~np.isnan(rhos)]
            lo, hi = boot_ci(ok, f"cor|{sname}|{exp}|{met}", fn=np.median) if len(ok) > 1 else (np.nan, np.nan)
            cor_rows.append(dict(set=sname, experiment=exp, metric=met, n_samples=len(W), n_defined=len(ok), median_rho=np.median(ok) if len(ok) else np.nan, ci95_lo=lo, ci95_hi=hi,
                                 pct_rho_negative=100 * np.mean(ok < 0) if len(ok) else np.nan, pct_rho_minus1=100 * np.mean(ok == -1) if len(ok) else np.nan))
pd.DataFrame(cor_rows).to_csv(out / "correlations.csv", index=False)

# ---------------- sensitivity: leave-one-out ranges for the headline contrasts ----------------
HEAD = {"hue": 90.0, "blur": 25.0, "noise": 40.0, "wrong_garment": 1.0}
sens = []
for sname, ids in SETS.items():
    if sname.startswith(("stress", "primary_hue")) or not ids: continue
    for exp in EXPS:
        for met in ("GREM", "CP", "TF", "SSIM_whole"):
            W = wide(ids, exp, met); d = (W[HEAD[exp]] - W[NEUTRAL[exp]]).values
            sens.append(dict(set=sname, experiment=exp, metric=met, level_max=HEAD[exp], n=len(d), mean_change=d.mean()))
sens = pd.DataFrame(sens); sens.to_csv(out / "sensitivity_max_level.csv", index=False)

# ---------------- mismatched reference ----------------
mmp = mm[mm.sample_id.isin(primary)].copy(); mrows = []
for s, g in mmp.groupby("sample_id"):
    m = g[g.matched]; o = g[~g.matched]
    mrows.append(dict(sample_id=s, garment=m.own_garment.iloc[0], matched_GREM=m.GREM.iloc[0], mismatched_mean=o.GREM.mean(), mismatched_min=o.GREM.min(), mismatched_max=o.GREM.max(), n_mismatched=len(o),
                      pct_mismatched_below_matched=100 * np.mean(o.GREM.values < m.GREM.iloc[0]), matched_rank=1 + int(np.sum(o.GREM.values > m.GREM.iloc[0])),
                      matched_CP=m.CP.iloc[0], mismatched_CP_mean=o.CP.mean(), matched_TF=m.TF.iloc[0], mismatched_TF_mean=o.TF.mean(),
                      SSIM_whole_matched=m.SSIM_whole.iloc[0], SSIM_whole_mismatched_range=o.SSIM_whole.max() - o.SSIM_whole.min()))
mdf = pd.DataFrame(mrows); mdf.to_csv(out / "mismatched_reference_summary.csv", index=False)
pairs = [(m, o) for s, g in mmp.groupby("sample_id") for m in [g[g.matched].GREM.iloc[0]] for o in g[~g.matched].GREM.values]
auc = np.mean([1.0 if m > o else 0.5 if m == o else 0.0 for m, o in pairs])

# ---------------- exploratory, NOT pre-registered: baseline GREM by the author's baseline rating ----------------
bl = raw[(raw.experiment == "hue") & raw.is_baseline_level & raw.sample_id.isin(primary)].groupby("baseline_quality").GREM.agg(["count", "mean", "min", "max"])
bl.to_csv(out / "exploratory_baseline_GREM_by_rating.csv")

# ---------------- compact table for the paper ----------------
rows = []
for exp in EXPS:
    sname = "primary_hue_informative" if exp == "hue" else "primary"
    for lv in sorted(ch[(ch.set == sname) & (ch.experiment == exp)].level.unique()):
        r = {"experiment": exp, "set": sname, "level": lv}
        for met in ("GREM", "CP", "TF", "SSIM_whole"):
            c = ch[(ch.set == sname) & (ch.experiment == exp) & (ch.metric == met) & (ch.level == lv)].iloc[0]
            r[f"d{met}"] = round(c.mean_change, 4); r[f"d{met}_ci"] = f"[{c.ci95_lo:.3f}, {c.ci95_hi:.3f}]"; r[f"{met}_pct_decrease"] = round(c.pct_strict_decrease)
        r["n"] = int(c.n); rows.append(r)
pd.DataFrame(rows).to_csv(out / "summary_for_paper.csv", index=False)

log = [f"plan sha256 {plan_hash}", f"primary n={len(primary)}: {primary}", f"hue-informative n={len(SETS['primary_hue_informative'])}; not informative: {SETS['primary_hue_NOT_informative']}",
       f"stress n={len(stress)}; expected failures: {ef.sample_id.tolist()}", f"bootstrap: {B} resamples, seed {SEED}, resampling samples",
       f"mismatched reference: matched > mismatched in {100 * auc:.1f}% of {len(pairs)} (matched, mismatched) pairs (ties count 0.5)"]
(out / "analysis_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8"); print("\n".join(log))
