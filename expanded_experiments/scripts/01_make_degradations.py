"""
01_make_degradations.py -- STEP 2a: create the degraded copies of every baseline output (lossless PNG) + a manifest + contact sheets.

    python3 scripts/01_make_degradations.py            (all samples; takes ~1-2 minutes)

Originals are never modified. Every change is confined to the garment mask (the same mask GREM uses). Neutral levels
(hue 0, blur 1, noise 0, alpha 0) are NOT saved: they ARE the baseline. Expected-failure samples are skipped here.
Outputs: degraded/<sample_id>/<experiment>_<level>.png, degraded/manifest.csv, degraded/meta.json, contact/<sample_id>.jpg
"""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import exp_lib as L
C = L.C

ap = argparse.ArgumentParser(); ap.add_argument("--base", default=None); ap.add_argument("--only", default=None)
a = ap.parse_args()
base, fold, rows, plan, plan_hash = L.load_all(a.base)
full, fail = L.select(rows, a.only)
lev = L.levels(plan); out_dir = L.EXP / "degraded"; sheet_dir = L.EXP / "contact"
out_dir.mkdir(exist_ok=True); sheet_dir.mkdir(exist_ok=True)
manifest = []
print(f"plan sha256 {plan_hash[:12]}... | {len(full)} samples get degradations | {len(fail)} expected-failure samples are skipped")

for r in full:
    sid = r["sample_id"]
    person = L.load_img(base, fold, "person", r["person_file"]); garment = L.load_img(base, fold, "garment", r["garment_file"])
    output = L.load_img(base, fold, "output", r["output_file"])
    mask = C.get_garment_mask(person, garment).astype(bool)
    partner = L.load_img(base, fold, "output", r["wrong_garment_partner"]) if r["wrong_garment_partner"] else None
    (out_dir / sid).mkdir(exist_ok=True)
    tiles = {}
    ys, xs = np.where(mask); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    my, mx = int(0.08 * (y1 - y0)), int(0.08 * (x1 - x0))
    box = (max(0, y0 - my), min(output.shape[0], y1 + my), max(0, x0 - mx), min(output.shape[1], x1 + mx))
    for exp in L.EXPERIMENTS:
        if exp == "wrong_garment" and partner is None:
            print(f"  {sid}: no partner -> wrong_garment skipped"); continue
        for lv in lev[exp]:
            seed = None
            if exp == "hue": img, eff = L.shift_hue(output, mask, lv)
            elif exp == "blur": img, eff = L.blur_garment(output, mask, lv)
            elif exp == "noise": img, eff, seed = L.add_noise(output, mask, lv, sid)
            else: img, eff = L.blend_partner(output, mask, partner, lv)
            neutral = (lv == L.NEUTRAL[exp])
            tiles.setdefault(exp, []).append((lv, img))
            if neutral: continue
            fn = f"{exp}_{lv}.png"; Image.fromarray(img).save(out_dir / sid / fn)
            diff = np.any(img != output, axis=2)
            manifest.append({"sample_id": sid, "experiment": exp, "level_requested": lv, "level_effective": eff, "file": f"{sid}/{fn}",
                             "sha256": L.sha256_file(out_dir / sid / fn), "mask_pixels": int(mask.sum()),
                             "pixels_changed_in_mask": int((diff & mask).sum()), "pixels_changed_outside_mask": int((diff & ~mask).sum()),
                             "mean_abs_change_in_mask": round(float(np.abs(img[mask].astype(float) - output[mask].astype(float)).mean()), 3),
                             "partner_output": r["wrong_garment_partner"] if exp == "wrong_garment" else "", "noise_seed": seed if seed is not None else ""})
    # contact sheet: one row per experiment, cropped to the mask region
    th = 170; rowsimg = []
    for exp, items in tiles.items():
        ims = []
        for lv, img in items:
            c = Image.fromarray(img[box[0]:box[1], box[2]:box[3]]); c = c.resize((int(c.width * th / c.height), th)); ims.append((lv, c))
        rowsimg.append((exp, ims))
    W = max(sum(c.width + 4 for _, c in ims) for _, ims in rowsimg) + 90
    sheet = Image.new("RGB", (W, len(rowsimg) * (th + 16) + 14), "white"); d = ImageDraw.Draw(sheet)
    d.text((4, 2), f"{sid} {r['output_file']}  person={r['person_file']}  garment={r['garment_file']}  (crop = garment mask region; first tile of each row = baseline)", fill="black")
    for i, (exp, ims) in enumerate(rowsimg):
        y = 14 + i * (th + 16); d.text((4, y + th // 2), exp, fill="black"); x = 90
        for lv, c in ims: sheet.paste(c, (x, y)); d.text((x + 2, y + th + 2), str(lv), fill="black"); x += c.width + 4
    sheet.save(sheet_dir / f"{sid}.jpg", quality=88)
    print(f"  {sid} done")

keys = list(manifest[0].keys()) if manifest else []
with open(out_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(manifest)
json.dump({"plan_sha256": plan_hash, "grem_core_sha256": L.sha256_file(L.CODE / "grem_core.py"), "samples_degraded": [r["sample_id"] for r in full],
           "expected_failures_skipped": [r["sample_id"] for r in fail], "n_images": len(manifest)}, open(out_dir / "meta.json", "w"), indent=2)
print(f"wrote {len(manifest)} degraded images, manifest.csv and contact sheets in contact/")
