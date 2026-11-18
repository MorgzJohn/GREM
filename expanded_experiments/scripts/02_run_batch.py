"""
02_run_batch.py -- STEP 2b: score every baseline and degraded image with grem_core.evaluate_grem (about 8-10 minutes).

    python3 scripts/02_run_batch.py

Scores are computed FROM THE SAVED PNG FILES, so what is on disk is what was scored. Neutral levels reuse the baseline score.
Also runs the mismatched-reference test (each primary baseline output scored against other garments; GREM only).
Outputs: results/raw_results.csv, results/mismatched_reference.csv, results/expected_failures.csv, results/run_meta.json
"""
import argparse, csv, json, platform, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
import exp_lib as L
C = L.C

ap = argparse.ArgumentParser(); ap.add_argument("--base", default=None); ap.add_argument("--only", default=None)
a = ap.parse_args()
base, fold, rows, plan, plan_hash = L.load_all(a.base)
full, fail = L.select(rows, a.only)
lev = L.levels(plan); deg = L.EXP / "degraded"; res = L.EXP / "results"; res.mkdir(exist_ok=True)
meta = json.load(open(deg / "meta.json", encoding="utf-8"))
core_hash = L.sha256_file(L.CODE / "grem_core.py")
if meta["plan_sha256"] != plan_hash: sys.exit("STOP: analysis_plan.json changed since the degraded images were made. Re-run 01_make_degradations.py.")
if meta["grem_core_sha256"] != core_hash: sys.exit("STOP: grem_core.py changed since the degraded images were made. Re-run 01_make_degradations.py.")
manifest = {}
for m in csv.DictReader(open(deg / "manifest.csv", encoding="utf-8")): manifest[(m["sample_id"], m["experiment"], str(m["level_requested"]))] = m
t0 = time.time(); out_rows = []; sets = {r["sample_id"]: r["analysis_set"] for r in rows}

for n, r in enumerate(full, 1):
    sid = r["sample_id"]
    person = L.load_img(base, fold, "person", r["person_file"]); garment = L.load_img(base, fold, "garment", r["garment_file"])
    output = L.load_img(base, fold, "output", r["output_file"])
    bl = L.metrics(C.evaluate_grem(person, garment, output))
    ident = {"sample_id": sid, "analysis_set": r["analysis_set"], "mask_quality": r["mask_quality"], "baseline_quality": r["baseline_quality"], "person_file": r["person_file"], "garment_file": r["garment_file"], "output_file": r["output_file"]}
    for exp in L.EXPERIMENTS:
        for lv in lev[exp]:
            neutral = lv == L.NEUTRAL[exp]
            if neutral: m = bl; eff = 0 if exp != "blur" else 1; f = ""
            else:
                key = (sid, exp, str(lv))
                if key not in manifest: continue                      # e.g. wrong_garment without a partner
                f = manifest[key]["file"]; eff = manifest[key]["level_effective"]
                m = L.metrics(C.evaluate_grem(person, garment, L.C.load_pil_image(Image.open(deg / f))))
            out_rows.append({**ident, "experiment": exp, "level_requested": lv, "level_effective": eff, "is_baseline_level": neutral, "file": f, **m,
                             "empty_mask": m["mask_pixels"] < 50})
    print(f"[{n}/{len(full)}] {sid} scored ({time.time() - t0:.0f}s)")

# expected failures: baseline only, reported as failures
fl = []
for r in fail:
    person = L.load_img(base, fold, "person", r["person_file"]); garment = L.load_img(base, fold, "garment", r["garment_file"])
    m = L.metrics(C.evaluate_grem(person, garment, L.load_img(base, fold, "output", r["output_file"])))
    fl.append({"sample_id": r["sample_id"], "output_file": r["output_file"], "garment_file": r["garment_file"], "reason": r["analysis_set"], **m})

# mismatched-reference test (primary baseline outputs vs every OTHER garment file in the primary set). GREM only.
prim = [r for r in full if r["analysis_set"] == "primary_candidate"]
garments = sorted({r["garment_file"] for r in prim}); mm = []
for r in prim:
    person = L.load_img(base, fold, "person", r["person_file"]); output = L.load_img(base, fold, "output", r["output_file"])
    for g in garments:
        m = L.metrics(C.evaluate_grem(person, L.load_img(base, fold, "garment", g), output))
        mm.append({"sample_id": r["sample_id"], "output_file": r["output_file"], "own_garment": r["garment_file"], "reference_garment": g,
                   "matched": g == r["garment_file"], **m})
print(f"mismatched-reference: {len(mm)} scorings")

def dump(path, data):
    if not data: return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)
dump(res / "raw_results.csv", out_rows); dump(res / "expected_failures.csv", fl); dump(res / "mismatched_reference.csv", mm)
import skimage, sklearn, cv2
json.dump({"plan_sha256": plan_hash, "grem_core_sha256": core_hash, "python": sys.version.split()[0], "platform": platform.platform(),
           "numpy": np.__version__, "opencv": cv2.__version__, "scikit_image": skimage.__version__, "scikit_learn": sklearn.__version__,
           "n_result_rows": len(out_rows), "n_mismatched_rows": len(mm), "seconds": round(time.time() - t0, 1)}, open(res / "run_meta.json", "w"), indent=2)
print(f"saved raw_results.csv ({len(out_rows)} rows), mismatched_reference.csv ({len(mm)}), expected_failures.csv ({len(fl)})")
