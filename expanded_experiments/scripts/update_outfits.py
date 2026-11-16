"""
update_outfits.py -- keep config/outfits.csv in sync when you add more generated outputs.

  python3 scripts/update_outfits.py              # 1) finds outputs not yet in the table and appends blank rows for them
  (open config/outfits.csv and fill person_file, garment_file, baseline_quality, vton_model for the new rows)
  python3 scripts/update_outfits.py --recompute  # 2) checks the table, recomputes partners and analysis sets, saves

Only run step 2 BEFORE the degraded images are scored (step 2 of the experiments). After that the dataset is frozen.
"""
import argparse, csv, json, shutil, sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
CSV = EXP / "config" / "outfits.csv"
ap = argparse.ArgumentParser(); ap.add_argument("--recompute", action="store_true"); ap.add_argument("--base", default=None)
a = ap.parse_args()
paths = json.load(open(EXP / "config" / "paths.json", encoding="utf-8"))
base = Path(a.base or paths["base_dir"]); fold = paths["folders"]
rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
fields = list(rows[0].keys())
EXT = (".png", ".jpg", ".jpeg")

def save(rows):
    shutil.copy(CSV, str(CSV) + ".bak")
    with open(CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

if not a.recompute:
    have = {r["output_file"] for r in rows}
    imgs = sorted(p.name for p in (base / fold["output"]).iterdir() if p.suffix.lower() in EXT)
    new = [n for n in imgs if n not in have]
    gone = sorted(have - set(imgs))
    if gone: print("WARNING: in the table but not in the output folder:", gone)
    if not new: print("No new outputs found. Table has", len(rows), "rows."); sys.exit(0)
    for n in new:
        r = {k: "" for k in fields}
        r.update(sample_id=f"S{len(rows) + 1:02d}", output_file=n, mapping_status="NEW - author must fill person_file and garment_file")
        rows.append(r)
    save(rows)
    print(f"Added {len(new)} new row(s):")
    for n in new: print("  ", n)
    print("\nNOW open config/outfits.csv and fill, for each new row:\n  person_file, garment_file (the files you gave the VTON tool),\n  baseline_quality (good/partial/poor, judged by eye BEFORE any score), vton_model.\n"
          "Then verify with:  python3 check_rows.py config/outfits.csv \"<your SAMPLE IMAGES folder>\"\nand run:  python3 scripts/update_outfits.py --recompute")
    sys.exit(0)

# ---------------- --recompute ----------------
def first(rows_, key, match_key, match_val):
    for r in rows_:
        if r[match_key] == match_val and r[key]: return r[key]
    return ""
for r in rows:                                   # reuse facts we already know about the same person / garment
    if not r["person_framing"]: r["person_framing"] = first(rows, "person_framing", "person_file", r["person_file"])
    for k in ("garment_colour_class", "garment_description"):
        if not r[k]: r[k] = first(rows, k, "garment_file", r["garment_file"])
errs = []
for r in rows:
    sid = r["sample_id"]
    for k in ("person_file", "garment_file", "baseline_quality", "vton_model"):
        if not r[k].strip(): errs.append(f"{sid} {r['output_file']}: '{k}' is empty")
    if r["baseline_quality"] and r["baseline_quality"] not in ("good", "partial", "poor"): errs.append(f"{sid}: baseline_quality must be good/partial/poor")
    if r["person_framing"] not in ("studio", "non-studio"): errs.append(f"{sid} {r['output_file']}: person_framing must be 'studio' or 'non-studio' (new person: fill it by hand)")
    for key, sub in (("person_file", "person"), ("garment_file", "garment"), ("output_file", "output")):
        if r[key] and not (base / fold[sub] / r[key]).is_file(): errs.append(f"{sid}: file not found: {sub}/{r[key]}")
if errs:
    print("NOT SAVED. Fix these first:"); [print("  -", e) for e in errs]; sys.exit(1)

by_person = defaultdict(list)
for r in rows: by_person[r["person_file"]].append(r["output_file"])
changes = []
for r in rows:
    outs = sorted(by_person[r["person_file"]])
    partner = outs[(outs.index(r["output_file"]) + 1) % len(outs)] if len(outs) > 1 else ""
    if r["wrong_garment_partner"] != partner: changes.append((r["sample_id"], r["output_file"], r["wrong_garment_partner"] or "none", partner or "none"))
    r["wrong_garment_partner"] = partner
failed = set()
rep = EXP / "validation" / "dataset_report.csv"
if rep.exists():
    for x in csv.DictReader(open(rep, encoding="utf-8")):
        if any(t in x.get("flags", "") for t in ("EMPTY_MASK", "TINY_MASK", "CP_DEFAULTS")): failed.add(x["output_file"])
for r in rows:
    r["analysis_set"] = ("expected_failure (empty / tiny mask or no usable reference colours)" if r["output_file"] in failed
                         else "primary_candidate" if r["person_framing"] == "studio" else "mask_stress_subset")
    if r["mapping_status"].startswith("NEW"): r["mapping_status"] = "confirmed by author (filled by hand; verify with check_rows.py)"
save(rows)
print(f"Saved {len(rows)} rows (backup: outfits.csv.bak)\n")
sets = defaultdict(list)
for r in rows: sets[r["analysis_set"]].append(r)
for s, rs in sets.items():
    print(f"{s}: {len(rs)} samples | {len({r['person_file'] for r in rs})} persons | {len({r['garment_file'] for r in rs})} garments | with wrong-garment partner: {sum(bool(r['wrong_garment_partner']) for r in rs)}")
print("\nbaseline quality:", {q: sum(r['baseline_quality'] == q for r in rows) for q in ('good', 'partial', 'poor')})
if not rep.exists(): print("NOTE: validation/dataset_report.csv not found, so white-garment failures are not yet flagged; re-run 00_validate_dataset.py then this script again.")
if changes:
    print("\nPartner assignments that changed because people gained outputs (fixed rule: next output of the same person by file name):")
    for c in changes: print("  ", c)
