"""
00_validate_dataset.py  --  STEP 1 of the expanded experiments (read-only: changes nothing in your image folders).

    cd GREM_revision_v2/expanded_experiments
    python3 scripts/00_validate_dataset.py

For every row of config/outfits.csv it checks the files, runs GREM's own mask function (grem_core.get_garment_mask)
and writes into ./validation/ :
  dataset_report.csv        one row per sample: image sizes, mask size, validity flags, mean saturation in the mask ...
  mask_overlay_<n>.jpg      person | garment | output, with the estimated garment mask outlined in red
  validation_log.txt        environment, hash of the analysis plan, summary
It does NOT score anything and does not modify GREM.
"""
import argparse, csv, hashlib, json, platform, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent; EXP = HERE.parent
sys.path.insert(0, str(EXP.parent / "code"))
import grem_core as C

ap = argparse.ArgumentParser()
ap.add_argument("--base", default=None, help="override base_dir from config/paths.json")
ap.add_argument("--out", default=str(EXP / "validation"))
a = ap.parse_args()

paths = json.load(open(EXP / "config" / "paths.json", encoding="utf-8"))
base = Path(a.base or paths["base_dir"]); fold = paths["folders"]
rows = list(csv.DictReader(open(EXP / "config" / "outfits.csv", encoding="utf-8")))
plan_bytes = (EXP / "config" / "analysis_plan.json").read_bytes()
plan = json.loads(plan_bytes); plan_hash = hashlib.sha256(plan_bytes).hexdigest()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
log = []
def say(s=""): print(s); log.append(s)

by_output = {r["output_file"]: r for r in rows}
report, problems = [], 0
tiles = []
for r in rows:
    sid = r["sample_id"]; flags = []
    p = {k: base / fold[k] / r[f"{k}_file"] for k in ("person", "garment", "output")}
    missing = [str(v) for v in p.values() if not v.is_file()]
    if missing:
        problems += 1; flags.append("MISSING_FILE"); say(f"{sid}: MISSING {missing}")
        report.append({"sample_id": sid, "output_file": r["output_file"], "flags": ";".join(flags)}); continue
    sizes = {k: Image.open(v).size for k, v in p.items()}
    for k, (w, h) in sizes.items():
        if abs(w / h - 0.75) > 0.02: flags.append(f"ASPECT_{k.upper()}_{w}x{h}")
    arr = {k: C.load_pil_image(Image.open(v)) for k, v in p.items()}
    mask = C.get_garment_mask(arr["person"], arr["garment"]); mpx = int(mask.sum())
    H, Wd = mask.shape
    ys, xs = np.where(mask > 0)
    bbox = (ys.min() / H, ys.max() / H, xs.min() / Wd, xs.max() / Wd) if mpx else (np.nan,) * 4
    g = arr["garment"].reshape(-1, 3)
    ref_nonwhite = int((~np.all(g > 220, axis=1)).sum())       # same rule as the colour term in grem_core.py
    hsv_g = cv2.cvtColor(arr["garment"], cv2.COLOR_RGB2HSV)
    ref_fg = int(((hsv_g[:, :, 1] > 30) | (hsv_g[:, :, 2] < 220)).sum())
    mean_sat = float(cv2.cvtColor(arr["output"], cv2.COLOR_RGB2HSV)[:, :, 1][mask > 0].mean()) if mpx else float("nan")
    if mpx < 50: flags.append("EMPTY_MASK")
    tiny = plan.get("validity_rules", {}).get("tiny_mask_fraction", 0.01)
    if mpx >= 50 and mpx / (H * Wd) < tiny: flags.append("TINY_MASK")
    if ref_nonwhite < 50: flags.append("CP_DEFAULTS_TO_0.5")
    part = r["wrong_garment_partner"]
    if part:
        pr = by_output.get(part)
        if pr is None or pr["person_file"] != r["person_file"]: flags.append("PARTNER_NOT_SAME_PERSON"); problems += 1
        elif not (base / fold["output"] / part).is_file(): flags.append("PARTNER_FILE_MISSING"); problems += 1
    report.append({
        "sample_id": sid, "output_file": r["output_file"], "person_file": r["person_file"], "garment_file": r["garment_file"],
        "analysis_set": r["analysis_set"], "baseline_quality": r["baseline_quality"],
        "person_size": "x".join(map(str, sizes["person"])), "garment_size": "x".join(map(str, sizes["garment"])),
        "output_size": "x".join(map(str, sizes["output"])),
        "mask_pixels": mpx, "mask_fraction_of_image": round(mpx / (H * Wd), 4),
        "mask_bbox_y0": round(bbox[0], 3), "mask_bbox_y1": round(bbox[1], 3), "mask_bbox_x0": round(bbox[2], 3), "mask_bbox_x1": round(bbox[3], 3),
        "reference_nonwhite_pixels": ref_nonwhite, "reference_foreground_pixels": ref_fg,
        "mean_saturation_in_mask_0_255": round(mean_sat, 1),
        "hue_informative(mean_sat>=40)": bool(mean_sat >= 40) if mpx else False,
        "has_wrong_garment_partner": bool(part), "flags": ";".join(flags)})
    # overlay tiles: mask outline on person and output
    def outlined(img):
        im = img.copy(); cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(im, cnts, -1, (255, 0, 0), 3); return im
    tiles.append((sid, r, mpx, H * Wd, flags, [outlined(arr["person"]), arr["garment"], outlined(arr["output"])]))

# ---- overlay sheets (5 samples per sheet) ----
w, h, per = 200, 267, 5
for s in range(0, len(tiles), per):
    chunk = tiles[s:s + per]
    sheet = Image.new("RGB", (3 * (w + 8), len(chunk) * (h + 30)), "white"); d = ImageDraw.Draw(sheet)
    for i, (sid, r, mpx, tot, flags, ims) in enumerate(chunk):
        y = i * (h + 30)
        d.text((4, y + 3), f"{sid}  {r['output_file']}  mask={mpx}px ({100 * mpx / tot:.1f}% of image)  " + (" | ".join(flags) if flags else "no flags"), fill="black")
        d.text((4, y + 15), f"person={r['person_file']}  garment={r['garment_file']}", fill="black")
        for j, im in enumerate(ims):
            t = Image.fromarray(im); t.thumbnail((w, h)); sheet.paste(t, (j * (w + 8), y + 28))
    sheet.save(out / f"mask_overlay_{s // per + 1}.jpg", quality=88)

if report:
    keys = list({k: 1 for r_ in report for k in r_}.keys())
    with open(out / "dataset_report.csv", "w", newline="", encoding="utf-8") as f:
        wri = csv.DictWriter(f, fieldnames=keys); wri.writeheader(); wri.writerows(report)

import skimage, sklearn
say("=== environment ==="); say(f"python {sys.version.split()[0]} | {platform.platform()} | numpy {np.__version__} | opencv {cv2.__version__} | scikit-image {skimage.__version__} | scikit-learn {sklearn.__version__}")
say(f"analysis_plan.json sha256 = {plan_hash}")
say("=== summary ===")
say(f"samples: {len(rows)} | problems (missing files / bad partners): {problems}")
for r_ in report:
    if r_.get("flags"): say(f"  {r_['sample_id']} {r_['output_file']}: {r_['flags']}")
say("=== NEXT: open the mask_overlay_*.jpg sheets, and for each sample write mask_quality in config/outfits.csv ===")
(out / "validation_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
sys.exit(1 if problems else 0)
