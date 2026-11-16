"""Shared helpers for the expanded experiments. NO scoring logic lives here: every score comes from grem_core.evaluate_grem."""
import csv, hashlib, json, sys
from pathlib import Path
import numpy as np, cv2
from PIL import Image

EXP = Path(__file__).resolve().parent.parent
CODE = EXP.parent / "code"
sys.path.insert(0, str(CODE))
import grem_core as C

EXPERIMENTS = ("hue", "blur", "noise", "wrong_garment")
NEUTRAL = {"hue": 0, "blur": 1, "noise": 0, "wrong_garment": 0}      # level that means "untouched baseline"

def sha256_bytes(b): return hashlib.sha256(b).hexdigest()
def sha256_file(p): return sha256_bytes(Path(p).read_bytes())

def load_all(base_override=None):
    paths = json.load(open(EXP / "config" / "paths.json", encoding="utf-8"))
    base = Path(base_override or paths["base_dir"]); fold = paths["folders"]
    rows = list(csv.DictReader(open(EXP / "config" / "outfits.csv", encoding="utf-8-sig")))
    pb = (EXP / "config" / "analysis_plan.json").read_bytes()
    return base, fold, rows, json.loads(pb), sha256_bytes(pb)

def load_img(base, fold, kind, name):
    return C.load_pil_image(Image.open(base / fold[kind] / name))

def select(rows, only=None):
    """Samples that get the full degradation battery: primary + mask-stress. Expected failures are baseline-only."""
    full = [r for r in rows if r["analysis_set"] in ("primary_candidate", "mask_stress_subset")]
    fail = [r for r in rows if r["analysis_set"].startswith("expected_failure")]
    if only:
        keep = set(only.split(","))
        full = [r for r in full if r["sample_id"] in keep]; fail = [r for r in fail if r["sample_id"] in keep]
    return full, fail

def levels(plan):
    e = plan["experiments"]
    return {"hue": e["B_hue"]["requested_degrees"], "blur": e["C_blur"]["kernel_sizes"],
            "noise": e["A_noise"]["sigma_gray_levels"], "wrong_garment": e["D_wrong_garment"]["alphas"]}

# ---------------- degradations (garment-mask pixels only; semantics follow the author's degrade.py) ----------------
def shift_hue(img, mask, degrees):
    units = round(degrees / 2)                       # OpenCV hue unit = 2 degrees (Python round: 7.5 -> 8, 22.5 -> 22)
    if units == 0: return img.copy(), 0
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv[:, :, 0] = ((hsv[:, :, 0].astype(np.int16) + units) % 180).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    out = img.copy(); out[mask] = rgb[mask]
    return out, 2 * units

def blur_garment(img, mask, k):
    if k == 1: return img.copy(), 1
    b = cv2.GaussianBlur(img, (k, k), 0); out = img.copy(); out[mask] = b[mask]
    return out, k

def seed_for(sample_id, sigma):
    return int.from_bytes(hashlib.sha256(f"{sample_id}|noise|{sigma}".encode()).digest()[:8], "big")

def add_noise(img, mask, sigma, sample_id):
    if sigma == 0: return img.copy(), 0, None
    seed = seed_for(sample_id, sigma); rng = np.random.default_rng(seed)
    out = img.copy(); px = out[mask].astype(np.float32)
    px += rng.normal(0.0, sigma, px.shape).astype(np.float32)
    out[mask] = np.clip(np.rint(px), 0, 255).astype(np.uint8)
    return out, sigma, seed

def blend_partner(img, mask, partner, alpha):
    if alpha == 0: return img.copy(), 0.0
    out = img.copy()
    mixed = (1 - alpha) * img[mask].astype(np.float32) + alpha * partner[mask].astype(np.float32)
    out[mask] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
    return out, alpha

# ---------------- metric record from evaluate_grem (no re-implementation of the metric) ----------------
def metrics(r):
    cp, tf = r["garment_color"], r["garment_texture"]
    gf_raw = round(0.55 * cp + 0.45 * tf, 4)
    pen = 0.12 if (cp < 0.3 and tf < 0.3) else 0.06 if cp < 0.4 else 0.0
    gf = max(0.0, gf_raw - pen); npv = r["identity_score"]
    grem = 0.8 * gf + (min(0.2 * npv, 0.10) if gf < 0.55 else 0.2 * npv)
    if abs(gf - r["garment_fidelity"]) > 1.5e-3 or abs(grem - r["overall_grem"]) > 1.5e-3:
        raise RuntimeError("components do not reproduce GF/GREM: the experiment must stop")
    return {"CP": cp, "TF": tf, "GF_raw": gf_raw, "penalty": pen, "GF": r["garment_fidelity"], "NP": npv, "GREM": r["overall_grem"],
            "SSIM_whole": r["ssim_whole"], "SSIM_garment_region": r["ssim_garment"], "SSIM_nongarment_region": r["ssim_identity"],
            "mask_pixels": int(np.asarray(r["garment_mask"]).sum())}
