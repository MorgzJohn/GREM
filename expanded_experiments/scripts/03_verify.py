"""
03_verify.py -- STEP 2c: independent checks that the experiment did what it claims. Run after 02. Exit code 1 if any check fails.

    python3 scripts/03_verify.py [--recheck 5]
"""
import argparse, csv, json, sys
import numpy as np
from PIL import Image
import exp_lib as L
C = L.C

ap = argparse.ArgumentParser(); ap.add_argument("--base", default=None); ap.add_argument("--recheck", type=int, default=5)
a = ap.parse_args()
base, fold, rows, plan, plan_hash = L.load_all(a.base)
deg, res = L.EXP / "degraded", L.EXP / "results"
man = list(csv.DictReader(open(deg / "manifest.csv", encoding="utf-8")))
raw = list(csv.DictReader(open(res / "raw_results.csv", encoding="utf-8")))
meta = json.load(open(res / "run_meta.json")); dmeta = json.load(open(deg / "meta.json"))
byid = {r["sample_id"]: r for r in rows}
log, bad = [], 0
def check(name, ok, detail=""):
    global bad; bad += (not ok); line = f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""); print(line); log.append(line)

check("analysis plan unchanged since the run", meta["plan_sha256"] == plan_hash == dmeta["plan_sha256"])
check("grem_core.py unchanged since the run", meta["grem_core_sha256"] == L.sha256_file(L.CODE / "grem_core.py") == dmeta["grem_core_sha256"])
# 1. file integrity, locality of changes, and that something really changed
bad_hash = bad_out = bad_in = 0; cache = {}
for m in man:
    sid = m["sample_id"]; r = byid[sid]
    if sid not in cache:
        person = L.load_img(base, fold, "person", r["person_file"]); garment = L.load_img(base, fold, "garment", r["garment_file"])
        cache[sid] = (L.load_img(base, fold, "output", r["output_file"]), C.get_garment_mask(person, garment).astype(bool))
    out, mask = cache[sid]
    p = deg / m["file"]
    if L.sha256_file(p) != m["sha256"]: bad_hash += 1
    img = np.array(Image.open(p).convert("RGB"))
    if not np.array_equal(img[~mask], out[~mask]): bad_out += 1
    if not np.any(img[mask] != out[mask]): bad_in += 1
check("every degraded file matches its recorded SHA-256", bad_hash == 0, f"({len(man)} files, {bad_hash} mismatches)")
check("pixels OUTSIDE the garment mask are bit-identical to the baseline in every degraded file", bad_out == 0, f"({bad_out} violations)")
check("every degraded file differs from the baseline INSIDE the mask", bad_in == 0, f"({bad_in} files unchanged)")
check("manifest agrees: zero changed pixels outside the mask", all(int(m["pixels_changed_outside_mask"]) == 0 for m in man))
# 2. invariances that must hold by construction
base_rows = {(x["sample_id"], x["experiment"]): x for x in raw if x["is_baseline_level"] == "True"}
inv_np = inv_ssn = inv_mask = inv_base = 0
for x in raw:
    b = base_rows[(x["sample_id"], x["experiment"])]
    if x["is_baseline_level"] == "True":
        inv_base += any(x[k] != b[k] for k in ("GREM", "CP", "TF", "NP", "SSIM_whole")); continue
    inv_np += abs(float(x["NP"]) - float(b["NP"])) > 1e-9
    inv_ssn += (x["SSIM_nongarment_region"] != b["SSIM_nongarment_region"])
    inv_mask += (x["mask_pixels"] != b["mask_pixels"])
check("NP is unchanged by every garment-only degradation", inv_np == 0, f"({inv_np} violations)")
check("non-garment-region SSIM is unchanged by every garment-only degradation", inv_ssn == 0, f"({inv_ssn} violations)")
check("mask size is identical across degradation levels of a sample", inv_mask == 0, f"({inv_mask} violations)")
check("baseline-level rows equal the baseline score", inv_base == 0)
# 3. hue effective angles and noise seeds
hue = {(m["level_requested"], m["level_effective"]) for m in man if m["experiment"] == "hue"}
check("hue effective angles follow shift = round(deg/2)", all(int(e) == 2 * round(int(q) / 2) for q, e in hue), str(sorted(hue)))
check("noise seeds are recorded and reproducible", all(int(m["noise_seed"]) == L.seed_for(m["sample_id"], int(m["level_requested"])) for m in man if m["experiment"] == "noise"))
# 4. arithmetic and no empty masks among scored rows
def arith(x):
    cp, tf, npv = float(x["CP"]), float(x["TF"]), float(x["NP"]); pen = 0.12 if (cp < .3 and tf < .3) else 0.06 if cp < .4 else 0.0
    gf = max(0, 0.55 * cp + 0.45 * tf - pen); g = 0.8 * gf + (min(0.2 * npv, 0.1) if gf < 0.55 else 0.2 * npv); return abs(g - float(x["GREM"])) < 2e-3
check("GREM reproduces from its components in every row", all(arith(x) for x in raw))
check("no scored row has an empty mask", not any(x["empty_mask"] == "True" for x in raw))
# 5. determinism: re-score a few saved files and require identical numbers
import random; random.seed(1); sample = random.sample([x for x in raw if x["is_baseline_level"] == "False"], min(a.recheck, len(raw)))
same = 0
for x in sample:
    r = byid[x["sample_id"]]
    person = L.load_img(base, fold, "person", r["person_file"]); garment = L.load_img(base, fold, "garment", r["garment_file"])
    m = L.metrics(C.evaluate_grem(person, garment, L.C.load_pil_image(Image.open(deg / x["file"]))))
    same += all(abs(float(x[k]) - m[k]) < 1e-12 for k in ("CP", "TF", "NP", "GREM", "SSIM_whole"))
check("re-scoring saved files reproduces identical numbers", same == len(sample), f"({same}/{len(sample)})")
(res / "verification_report.txt").write_text("\n".join(log) + f"\n\n{'ALL CHECKS PASSED' if not bad else str(bad) + ' CHECK(S) FAILED'}\n", encoding="utf-8")
print("\n" + ("ALL CHECKS PASSED" if not bad else f"{bad} CHECK(S) FAILED"))
sys.exit(1 if bad else 0)
