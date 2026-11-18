"""grem_core.py -- GREM scoring functions (no Streamlit). Extracted from grm_app_ssim2d.py."""
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────
# GREM Core Functions
# ─────────────────────────────────────────────

def load_pil_image(pil_img, size=(768, 1024)):
    return np.array(pil_img.convert("RGB").resize(size))


# ── FIX 1: Garment-guided mask ──────────────────────────────────────────────
#
# OLD approach: hardcoded rectangle mask[0.25h:0.65h, 0.30w:0.70w].
# Problem: this is spatially fixed and breaks for close-ups, seated poses,
#          full-body shots, or any non-standard framing.
#
# NEW approach: derive the mask from the garment image itself.
#   Step 1 – Remove the white/near-white background from the garment image
#             using a saturation + brightness threshold in HSV space.
#   Step 2 – Project the resulting foreground silhouette onto the person/output
#             image using the torso region as an anchor (window y 22-72% of height,
#             20-80% width).  This keeps the mask body-centred while still
#             being shaped by the actual garment silhouette.
#   Step 3 – Morphological cleanup (close small holes, remove noise) and a
#             light Gaussian blur to avoid hard-edge artefacts.
#
# Why this is better:
#   • A boxy shirt and a flowing dress produce different mask shapes.
#   • The mask adapts to garment width/coverage rather than assuming a
#     fixed torso box.
#   • Still robust when the garment image is a plain product shot on white.

def get_garment_mask(person_img, garment_img):
    h, w = person_img.shape[:2]

    # ── Garment foreground from the garment product image ──
    # Convert to HSV and separate background (high V, low S = white/grey)
    g_hsv = cv2.cvtColor(garment_img, cv2.COLOR_RGB2HSV)
    s_ch  = g_hsv[:, :, 1]   # saturation
    v_ch  = g_hsv[:, :, 2]   # brightness

    # Pixels that are NOT background: either colourful (S>30) or dark (V<220)
    foreground = ((s_ch > 30) | (v_ch < 220)).astype(np.uint8)

    # Morphological cleanup on the garment silhouette
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Bounding box of garment foreground (relative proportions)
    ys, xs = np.where(foreground > 0)
    if len(ys) < 100:
        # Nearly all-white garment - default box ratios. NOTE: 'foreground' stays (almost) empty,
        # so the projected mask is (almost) empty and downstream scores default to 0.0 / 0.5.
        top_r, bot_r, left_r, right_r = 0.22, 0.70, 0.25, 0.75
    else:
        gh, gw = garment_img.shape[:2]
        top_r   = float(ys.min()) / gh
        bot_r   = float(ys.max()) / gh
        left_r  = float(xs.min()) / gw
        right_r = float(xs.max()) / gw

    # ── Anchor to torso region of the person image ──
    # The torso occupies roughly y: 22%–72%, x: 20%–80% for a standing person.
    # We scale the garment bounding box proportions into this torso window.
    torso_top    = 0.22
    torso_bottom = 0.72
    torso_left   = 0.20
    torso_right  = 0.80
    torso_h      = torso_bottom - torso_top
    torso_w      = torso_right  - torso_left

    proj_top    = int((torso_top  + top_r   * torso_h) * h)
    proj_bottom = int((torso_top  + bot_r   * torso_h) * h)
    proj_left   = int((torso_left + left_r  * torso_w) * w)
    proj_right  = int((torso_left + right_r * torso_w) * w)

    # Clamp to image bounds
    proj_top    = max(0, min(proj_top,    h - 1))
    proj_bottom = max(0, min(proj_bottom, h - 1))
    proj_left   = max(0, min(proj_left,   w - 1))
    proj_right  = max(0, min(proj_right,  w - 1))

    # Build the mask on the person canvas
    mask = np.zeros((h, w), dtype=np.uint8)
    if proj_bottom > proj_top and proj_right > proj_left:
        # Resize garment foreground silhouette into the projected window
        region_h = proj_bottom - proj_top
        region_w = proj_right  - proj_left
        garment_resized = cv2.resize(foreground, (region_w, region_h),
                                     interpolation=cv2.INTER_NEAREST)
        mask[proj_top:proj_bottom, proj_left:proj_right] = garment_resized
    else:
        # Degenerate case: fill the whole torso box
        mask[proj_top:proj_bottom, proj_left:proj_right] = 1

    # Final cleanup and soft edges
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    mask_float = cv2.GaussianBlur(mask.astype(np.float32), (31, 31), 0)
    return (mask_float > 0.25).astype(np.uint8)


# ── Garment SSIM: compares person→output (not garment→output) ────────────────
#
# KEY DESIGN DECISION:
# The flat garment product shot and the worn output are NEVER structurally
# similar — the garment deforms, gets body shadows, and changes perspective
# when worn on a person. Comparing garment→output with SSIM will always give
# a low score even for a perfect try-on, because it's a viewpoint mismatch,
# not a quality failure.
#
# CORRECT framing: within the garment region, we want to know whether the
# output is structurally coherent and similar to what the body looked like
# before — i.e., person→output SSIM in the garment zone. This tells us:
# "Did the model place a realistic, well-integrated garment here, or did it
# produce blurry/artifact-ridden output?"
# A good try-on will have moderate-to-high person→output SSIM in the garment
# region because the underlying body shape is preserved, just with new fabric.

# -- SSIM baselines (NOT part of the GREM score) -------------------------------
#
# Whole-image SSIM: standard 2-D SSIM (scikit-image structural_similarity, grayscale,
# default 7x7 uniform window, data_range=255, border cropped by the library) between the
# ORIGINAL PERSON image and the generated OUTPUT.
#
# Regional SSIM: the SSIM *map* is computed on the full 2-D images, and the score is the
# mean of the map over pixels whose whole window lies inside the region. The region mask is
# eroded by the window size, so windows that straddle the region boundary (and therefore mix
# in pixels from outside the region) are excluded. Pixels in the library's border crop are
# excluded as well. With an all-ones mask this equals the whole-image SSIM exactly.
# Returns NaN if fewer than MIN_SSIM_PIXELS valid pixels remain.

SSIM_WIN = 7
MIN_SSIM_PIXELS = 50

def _gray_pair(img1, img2):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    return (cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY))


def calculate_ssim_whole(img1, img2):
    """Standard 2-D whole-image SSIM (grayscale)."""
    g1, g2 = _gray_pair(img1, img2)
    return round(float(ssim(g1, g2, data_range=255, win_size=SSIM_WIN)), 4)


def calculate_ssim_region(img1, img2, mask):
    """Masked 2-D SSIM: mean of the SSIM map over pixels whose window lies fully in `mask`."""
    g1, g2 = _gray_pair(img1, img2)
    _, smap = ssim(g1, g2, data_range=255, win_size=SSIM_WIN, full=True)
    valid = cv2.erode((mask == 1).astype(np.uint8),
                      np.ones((SSIM_WIN, SSIM_WIN), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=1)
    pad = (SSIM_WIN - 1) // 2
    valid[:pad, :] = 0; valid[-pad:, :] = 0; valid[:, :pad] = 0; valid[:, -pad:] = 0
    if int(valid.sum()) < MIN_SSIM_PIXELS:
        return float("nan")
    return round(float(smap[valid == 1].mean()), 4)


def calculate_garment_color_preservation(garment_img, output_img, mask):
    # ── Color decay curve loosened: exp(-x/40) instead of exp(-x/25) ──────
    # exp(-x/25): ΔE=10 → 0.67, ΔE=15 → 0.55, ΔE=20 → 0.45  (too harsh)
    # exp(-x/40): ΔE=10 → 0.78, ΔE=15 → 0.69, ΔE=20 → 0.61  (realistic)
    # Rationale: a ΔE of 10-15 is a noticeable but acceptable colour shift
    # (e.g. velvet fabric appearance under different lighting). Only ΔE > 30
    # corresponds to a clearly wrong colour, so the curve should be gentle
    # in the 0-20 range.

    if garment_img.shape != output_img.shape:
        output_img = cv2.resize(output_img, (garment_img.shape[1], garment_img.shape[0]))

    g_pixels = garment_img.reshape(-1, 3).astype(np.float32)
    not_bg   = ~(np.all(g_pixels > 220, axis=1))
    g_pixels = g_pixels[not_bg]

    if len(g_pixels) < 50:
        return 0.5

    k      = min(3, len(g_pixels))
    kmeans = KMeans(n_clusters=k, n_init=5, random_state=42)
    kmeans.fit(g_pixels)
    dominant_colors = kmeans.cluster_centers_

    out_region = output_img[mask == 1].astype(np.float32)
    if len(out_region) < 50:
        return 0.0

    dominant_lab = rgb2lab(
        dominant_colors.reshape(1, -1, 3).astype(np.uint8) / 255.0
    ).reshape(-1, 3)

    out_lab = rgb2lab(
        out_region.reshape(1, -1, 3).astype(np.uint8) / 255.0
    ).reshape(-1, 3)

    per_pixel_min_de = np.min(
        np.stack([
            deltaE_ciede2000(out_lab, np.tile(dc, (len(out_lab), 1)))
            for dc in dominant_lab
        ], axis=0),
        axis=0
    )

    mean_de = np.mean(per_pixel_min_de)
    score   = float(np.exp(-mean_de / 40.0))   # loosened from /25 → /40
    return round(max(0.0, min(1.0, score)), 4)


def calculate_identity_color_preservation(person_img, output_img, mask):
    if person_img.shape != output_img.shape:
        output_img = cv2.resize(output_img, (person_img.shape[1], person_img.shape[0]))

    lab1    = rgb2lab(person_img / 255.0)
    lab2    = rgb2lab(output_img / 255.0)
    delta_e = deltaE_ciede2000(lab1, lab2)

    delta_e_masked = delta_e[mask == 1]
    if len(delta_e_masked) < 50:
        return 0.0

    mean_de = np.mean(delta_e_masked)
    score   = float(np.exp(-mean_de / 30.0))
    return round(max(0.0, min(1.0, score)), 4)


# ── Texture: complexity-matching, not pixel-matching ────────────────────────
#
# KEY DESIGN DECISION:
# The flat garment product shot has completely different lighting, shadows
# and fold structure than the same garment worn on a body. Comparing their
# Sobel gradient maps directly will always give a low score even for a
# perfect try-on — this is a viewpoint/deformation problem, not a quality
# problem.
#
# CORRECT framing for texture: we want to know whether the output garment
# region has *similar texture complexity* to the reference garment — not
# whether the gradient maps pixel-match. A velvet wrap top should transfer
# as a similarly textured surface in the output; a plain tee should transfer
# as a similarly smooth surface.
#
# Method: compare the MEAN gradient magnitude of each region (only the mean is used;
# the std is NOT used) rather than the pixel-wise map. This is viewpoint-invariant:
# a high-texture garment worn naturally will have a similarly rich gradient
# distribution in the output.
#
# Scoring: penalise only large differences in complexity level, using a
# decay exp(-d/0.3), d=|r-1|/(r+1), so moderate complexity mismatches don't crater
# the score.

def calculate_texture_score(garment_img, output_img, mask):
    # ── Operate on the 2D masked region (bounding box crop) ──────────────
    if garment_img.shape != output_img.shape:
        output_img = cv2.resize(output_img, (garment_img.shape[1], garment_img.shape[0]))

    ys, xs = np.where(mask == 1)
    if len(ys) < 50:
        return 0.0

    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    crop_mask = mask[y0:y1, x0:x1].astype(np.float32)

    def masked_sobel_mag(img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[y0:y1, x0:x1].astype(np.float32)
        gray *= crop_mask   # zero non-garment pixels
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.sqrt(sx**2 + sy**2)
        return mag[crop_mask > 0]   # return only masked pixel values

    mag_g = masked_sobel_mag(garment_img)   # garment reference
    mag_o = masked_sobel_mag(output_img)    # output garment region

    if len(mag_g) < 50 or len(mag_o) < 50:
        return 0.0

    # Compare texture COMPLEXITY (mean gradient magnitude), not pixel maps.
    # Use the ratio of the two means so the absolute gradient scale cancels --
    # only the relative complexity level is compared.
    mean_g = np.mean(mag_g) + 1e-6
    mean_o = np.mean(mag_o) + 1e-6

    # Ratio of complexities — 1.0 = identical complexity, 0.5 or 2.0 = half/double
    ratio = mean_o / mean_g
    # Map to [0,1]: ratio=1 → diff=0 → score=1.0
    #               ratio=0.5 or 2.0 -> diff=0.333 -> score=0.33
    #               ratio=0.25 or 4.0 -> diff=0.60 -> score=0.14
    complexity_diff = abs(ratio - 1.0) / (ratio + 1.0)   # in [0, 1)
    score = float(np.exp(-complexity_diff / 0.3))

    return round(max(0.0, min(1.0, score)), 4)


# NOTE: the 0.40/0.35/0.25 SSIM-based weights discussed in this block are SUPERSEDED.
# The code actually uses Color 0.55 / Texture 0.45 with no SSIM term (W_COLOR, W_TEXTURE below).
# ── FIX 3: Weight justification via soft ablation ───────────────────────────
#
# OLD weights: SSIM×0.5 + Color×0.2 + Texture×0.3  — picked arbitrarily.
#
# NEW weights reflect the relative diagnostic importance of each component,
# grounded in perceptual literature and VTON-specific reasoning:
#
#   SSIM      (0.40) – structural integrity (shape, edges, folds).
#                       Reduced slightly because SSIM can be fooled by blur.
#   Color     (0.35) – perceptual colour accuracy.
#                       Increased: colour is the most immediately noticeable
#                       attribute for a viewer judging whether a garment looks
#                       "right" (wrong colour = obviously wrong try-on).
#   Texture   (0.25) – fine-detail fidelity (patterns, fabric grain).
#                       Texture matters but is the hardest to transfer and the
#                       hardest to measure reliably, so it carries less weight.
#
# Weight rationale in one sentence: colour is perceptually dominant for garment
# judgement, structural coherence is necessary but not sufficient, and texture
# is a tie-breaker.  The 0.40/0.35/0.25 split reflects this priority order.
#
# Penalty thresholds are now derived from the metric scale:
#   • gc < 0.3 AND gt < 0.3  → both perceptual signals are poor (bottom 30%)
#     → strong penalty of 0.12 (≈30% of max weight headroom)
#   • gc < 0.4               → colour alone is poor
#     → moderate penalty of 0.06
#   These thresholds correspond to the 30th and 40th percentile of the [0,1]
#   scale, i.e., scores that are definitively below "moderate" quality.

# ── Weight rationale (GREM-only components, no SSIM dependency) ─────────────
#
# Garment Fidelity = Color × 0.55 + Texture × 0.45
#   Color  (0.55): most perceptually dominant — wrong colour = immediately
#                  obvious wrong try-on. Higher weight.
#   Texture(0.45): fabric grain/richness transfer. Slightly lower because
#                  texture complexity matching is harder to transfer perfectly.
#
# Identity Score = Identity Color only (pure GREM, no SSIM)
#
# Overall GREM = Garment Fidelity × 0.8 + Identity Contribution × 0.2

W_COLOR   = 0.55
W_TEXTURE = 0.45

PENALTY_BOTH  = 0.12
PENALTY_COLOR = 0.06


def evaluate_grem(person_np, garment_np, output_np):
    gm  = get_garment_mask(person_np, garment_np)
    bgm = 1 - gm
    one = np.ones(person_np.shape[:2], dtype=np.uint8)

    # ── SSIM metrics (baseline comparison only, NOT used in GREM score) ──
    ss  = calculate_ssim_whole(person_np, output_np)
    gs  = calculate_ssim_region(person_np, output_np, gm)
    is_ = calculate_ssim_region(person_np, output_np, bgm)

    # ── GREM components (your contribution) ──
    gc  = calculate_garment_color_preservation(garment_np, output_np, gm)
    gt  = calculate_texture_score(garment_np, output_np, gm)
    ic  = calculate_identity_color_preservation(person_np, output_np, bgm)

    # Garment Fidelity — pure GREM, no SSIM
    gf_raw  = gc * W_COLOR + gt * W_TEXTURE
    penalty = (PENALTY_BOTH  if (gc < 0.3 and gt < 0.3)
               else PENALTY_COLOR if gc < 0.4
               else 0.0)
    gf = round(max(0.0, gf_raw - penalty), 4)

    # Identity Score — pure GREM
    isc = round(ic, 4)

    # Overall GREM
    ic_contrib = min(isc * 0.2, 0.10) if gf < 0.55 else isc * 0.2
    og = round(gf * 0.8 + ic_contrib, 4)

    return {
        "ssim_whole":       ss,
        "ssim_garment":     gs,
        "ssim_identity":    is_,
        "garment_color":    gc,
        "garment_texture":  gt,
        "garment_fidelity": gf,
        "identity_color":   ic,
        "identity_score":   isc,
        "overall_grem":     og,
        "grem_gap":         round(abs(ss - og), 4),
        "garment_mask":     gm,
    }
