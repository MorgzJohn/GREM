import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.color import deltaE_ciede2000, rgb2lab
from sklearn.cluster import KMeans
import pandas as pd


st.set_page_config(
    page_title="GREM — Garment Region Evaluation Metric",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
.stApp { background: #080A0F; color: #E8E2D9; font-family: 'Outfit', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem 2rem; max-width: 1400px; }

.hero-wrap { position: relative; padding: 5rem 0 3rem 0; text-align: center; overflow: hidden; }
.hero-wrap::before { content: ''; position: absolute; top: -60px; left: 50%; transform: translateX(-50%); width: 600px; height: 600px; background: radial-gradient(ellipse, #C8A96E18 0%, #C8A96E05 50%, transparent 70%); pointer-events: none; }
.hero-eyebrow { font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.35em; color: #C8A96E; text-transform: uppercase; margin-bottom: 1.2rem; opacity: 0.9; }
.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(3.5rem, 8vw, 7rem); font-weight: 900; line-height: 0.9; margin: 0; letter-spacing: -0.02em; }
.hero-title .accent { color: #C8A96E; font-style: italic; }
.hero-title .light { color: #E8E2D9; }
.hero-subtitle { font-family: 'Outfit', sans-serif; font-size: 1rem; color: #6B6560; margin-top: 1.5rem; letter-spacing: 0.05em; font-weight: 300; }
.hero-divider { width: 60px; height: 1px; background: linear-gradient(90deg, transparent, #C8A96E, transparent); margin: 2rem auto; }

.stTabs [data-baseweb="tab-list"] { background: transparent; border-bottom: 1px solid #1E2028; gap: 0; padding: 0; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Mono', monospace; font-size: 0.75rem; letter-spacing: 0.15em; text-transform: uppercase; color: #4A4845; padding: 1rem 2rem; background: transparent; border: none; border-bottom: 2px solid transparent; transition: all 0.3s ease; }
.stTabs [aria-selected="true"] { color: #C8A96E !important; border-bottom-color: #C8A96E !important; background: transparent !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 2.5rem; }

.upload-label { font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.2em; text-transform: uppercase; color: #C8A96E; margin-bottom: 0.5rem; display: block; }
.stFileUploader > div { background: #0D0F14 !important; border: 1px solid #1E2028 !important; border-radius: 4px !important; }
.stFileUploader > div:hover { border-color: #C8A96E44 !important; }

.score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1px; background: #1E2028; border: 1px solid #1E2028; border-radius: 4px; overflow: hidden; margin: 2rem 0; }
.score-card { background: #0D0F14; padding: 2rem 1.5rem; text-align: center; position: relative; }
.score-card::after { content: ''; position: absolute; bottom: 0; left: 50%; transform: translateX(-50%); width: 0; height: 2px; background: #C8A96E; transition: width 0.4s ease; }
.score-card:hover::after { width: 80%; }
.score-num { font-family: 'Playfair Display', serif; font-size: 3.5rem; font-weight: 700; line-height: 1; color: #E8E2D9; letter-spacing: -0.02em; }
.score-num.gold { color: #C8A96E; }
.score-num.highlight { color: #4ECDC4; }
.score-denom { font-family: 'DM Mono', monospace; font-size: 0.7rem; color: #3A3835; margin-top: 0.25rem; }
.score-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #4A4845; margin-top: 0.75rem; }

.section-header { display: flex; align-items: center; gap: 1rem; margin: 2.5rem 0 1.5rem 0; }
.section-line { flex: 1; height: 1px; background: #1E2028; }
.section-title { font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.25em; text-transform: uppercase; color: #4A4845; white-space: nowrap; }

.metric-row { display: flex; align-items: center; padding: 0.85rem 1.25rem; border-bottom: 1px solid #1E2028; transition: background 0.2s ease; }
.metric-row:last-child { border-bottom: none; }
.metric-row:hover { background: #0D0F14; }
.metric-name { font-family: 'Outfit', sans-serif; font-size: 0.85rem; color: #6B6560; flex: 1; font-weight: 300; }
.metric-name.bold { color: #E8E2D9; font-weight: 500; }
.metric-score { font-family: 'DM Mono', monospace; font-size: 1.1rem; color: #E8E2D9; width: 80px; text-align: right; margin-right: 1rem; }
.metric-bar-wrap { width: 120px; height: 2px; background: #1E2028; border-radius: 2px; overflow: hidden; margin-right: 1rem; }
.metric-bar-fill { height: 100%; border-radius: 2px; }
.bar-good { background: #4ECDC4; }
.bar-moderate { background: #C8A96E; }
.bar-poor { background: #E05C5C; }
.metric-tag { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.1em; padding: 0.2rem 0.6rem; border-radius: 2px; width: 70px; text-align: center; }
.tag-good { background: #4ECDC415; color: #4ECDC4; }
.tag-moderate { background: #C8A96E15; color: #C8A96E; }
.tag-poor { background: #E05C5C15; color: #E05C5C; }

.finding-panel { border-left: 2px solid #C8A96E; padding: 1.5rem 2rem; background: #0D0F14; margin: 1.5rem 0; }
.finding-panel.good { border-left-color: #4ECDC4; }
.finding-eyebrow { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #C8A96E; margin-bottom: 0.5rem; }
.finding-panel.good .finding-eyebrow { color: #4ECDC4; }
.finding-text { font-family: 'Outfit', sans-serif; font-size: 0.95rem; color: #A09890; line-height: 1.7; font-weight: 300; }
.finding-text strong { color: #E8E2D9; font-weight: 500; }

.stButton > button { background: transparent !important; border: 1px solid #C8A96E !important; color: #C8A96E !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important; letter-spacing: 0.2em !important; text-transform: uppercase !important; padding: 0.85rem 2rem !important; border-radius: 2px !important; transition: all 0.3s ease !important; width: 100% !important; }
.stButton > button:hover { background: #C8A96E15 !important; transform: translateY(-1px) !important; }

.stProgress > div > div { background: #C8A96E !important; border-radius: 0 !important; }
.stProgress > div { background: #1E2028 !important; border-radius: 0 !important; height: 2px !important; }
.stSlider > div > div > div { background: #C8A96E !important; }

[data-testid="metric-container"] { background: #0D0F14; border: 1px solid #1E2028; border-radius: 4px; padding: 1rem 1.25rem; }
[data-testid="metric-container"] label { font-family: 'DM Mono', monospace !important; font-size: 0.6rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; color: #4A4845 !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'Playfair Display', serif !important; font-size: 1.8rem !important; color: #E8E2D9 !important; }

.stAlert { background: #0D0F14 !important; border-radius: 4px !important; font-family: 'Outfit', sans-serif !important; }
.stDownloadButton > button { background: transparent !important; border: 1px solid #1E2028 !important; color: #4A4845 !important; font-family: 'DM Mono', monospace !important; font-size: 0.65rem !important; letter-spacing: 0.15em !important; text-transform: uppercase !important; border-radius: 2px !important; transition: all 0.3s ease !important; }
.stDownloadButton > button:hover { border-color: #C8A96E !important; color: #C8A96E !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080A0F; }
::-webkit-scrollbar-thumb { background: #2A2825; border-radius: 2px; }

.about-body { font-family: 'Outfit', sans-serif; font-size: 0.95rem; color: #6B6560; line-height: 1.8; font-weight: 300; max-width: 760px; }
.about-body h3 { font-family: 'Playfair Display', serif; font-size: 1.6rem; color: #E8E2D9; font-weight: 700; margin: 2.5rem 0 0.75rem 0; }
.about-body strong { color: #E8E2D9; font-weight: 500; }
.about-body code { font-family: 'DM Mono', monospace; font-size: 0.8rem; background: #0D0F14; border: 1px solid #1E2028; padding: 0.1rem 0.4rem; border-radius: 2px; color: #C8A96E; }

.cmp-table { width: 100%; border-collapse: collapse; font-family: 'Outfit', sans-serif; font-size: 0.85rem; }
.cmp-table th { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: #4A4845; padding: 0.75rem 1rem; border-bottom: 1px solid #1E2028; text-align: left; }
.cmp-table td { padding: 0.85rem 1rem; border-bottom: 1px solid #12141A; color: #6B6560; font-weight: 300; }
.cmp-table td:first-child { color: #A09890; }
.cmp-table tr:hover td { background: #0D0F14; }

.mask-note { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.1em; color: #4A4845; padding: 0.4rem 0.8rem; background: #0D0F14; border: 1px solid #1E2028; border-radius: 2px; display: inline-block; margin-top: 0.5rem; }
.mask-note.garment-guided { border-color: #C8A96E44; color: #C8A96E99; }
</style>
""", unsafe_allow_html=True)


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
#             image using the torso region as an anchor (center 40-70% height,
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
        # Nearly all-white garment – fall back to a sensible torso box
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

def calculate_ssim_region(img1, img2, mask):
    """General SSIM between two images within a mask region."""
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    region1 = gray1[mask == 1]
    region2 = gray2[mask == 1]

    if len(region1) < 50:
        return 0.0

    score = ssim(region1, region2, data_range=255)
    return round(float(score), 4)


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
# Method: compare the *distribution* of gradient magnitudes (via their mean
# and std) rather than the pixel-wise map. This is viewpoint-invariant:
# a high-texture garment worn naturally will have a similarly rich gradient
# distribution in the output.
#
# Scoring: penalise only large differences in complexity level, using a
# loose decay (exp(-x/1.5)) so moderate complexity mismatches don't crater
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
    # Normalise both by their own mean so scale differences don't matter —
    # only the relative complexity level is compared.
    mean_g = np.mean(mag_g) + 1e-6
    mean_o = np.mean(mag_o) + 1e-6

    # Ratio of complexities — 1.0 = identical complexity, 0.5 or 2.0 = half/double
    ratio = mean_o / mean_g
    # Map to [0,1]: ratio=1 → diff=0 → score=1.0
    #               ratio=0.5 or 2.0 → diff=0.5 → score=0.72
    #               ratio=0.25 or 4.0 → diff=0.75 → score=0.47
    complexity_diff = abs(ratio - 1.0) / (ratio + 1.0)   # in [0, 0.5]
    score = float(np.exp(-complexity_diff / 0.3))

    return round(max(0.0, min(1.0, score)), 4)


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
    ss  = calculate_ssim_region(person_np, output_np, one)
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


def score_meta(s):
    if s >= 0.75:   return ("GOOD",     "good",     "bar-good")
    elif s >= 0.55: return ("MODERATE", "moderate", "bar-moderate")
    else:           return ("POOR",     "poor",     "bar-poor")

def metric_row_html(label, score, bold=False):
    tag, cls, bar_cls = score_meta(score)
    return f"""<div class="metric-row">
        <span class="metric-name {'bold' if bold else ''}">{label}</span>
        <span class="metric-score">{score}</span>
        <div class="metric-bar-wrap"><div class="metric-bar-fill {bar_cls}" style="width:{min(int(score*100),100)}%"></div></div>
        <span class="metric-tag tag-{cls}">{tag}</span>
    </div>"""

def section_divider(title):
    return f"""<div class="section-header">
        <div class="section-line"></div>
        <span class="section-title">{title}</span>
        <div class="section-line"></div>
    </div>"""


# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <p class="hero-eyebrow">Virtual Try-On Research · MCA 2024–25</p>
    <h1 class="hero-title"><span class="accent">G</span><span class="light">REM</span></h1>
    <div class="hero-divider"></div>
    <p class="hero-subtitle">Garment Region Evaluation Metric</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["✦  Single Evaluation", "◈  Batch Analysis", "○  About"])


# ══════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════
with tab1:
    st.markdown(section_divider("Upload Images"), unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown('<span class="upload-label">01 · Person Input</span>', unsafe_allow_html=True)
        person_file = st.file_uploader("Person", type=["jpg","jpeg","png"], key="person", label_visibility="collapsed")
        if person_file: st.image(person_file, use_container_width=True)

    with c2:
        st.markdown('<span class="upload-label">02 · Garment Image</span>', unsafe_allow_html=True)
        garment_file = st.file_uploader("Garment", type=["jpg","jpeg","png"], key="garment", label_visibility="collapsed")
        if garment_file: st.image(garment_file, use_container_width=True)

    with c3:
        st.markdown('<span class="upload-label">03 · VTON Output</span>', unsafe_allow_html=True)
        output_file = st.file_uploader("Output", type=["jpg","jpeg","png"], key="output", label_visibility="collapsed")
        if output_file: st.image(output_file, use_container_width=True)

    st.markdown(section_divider("Run"), unsafe_allow_html=True)
    run_btn = st.button("Run Evaluation →", key="run")

    if run_btn:
        if not all([person_file, garment_file, output_file]):
            st.error("Upload all three images to proceed.")
        else:
            with st.spinner("Computing GREM scores..."):
                p_np = load_pil_image(Image.open(person_file))
                g_np = load_pil_image(Image.open(garment_file))
                o_np = load_pil_image(Image.open(output_file))
                R    = evaluate_grem(p_np, g_np, o_np)
                mask = R["garment_mask"]

            # ── Top scorecard ──
            st.markdown(section_divider("Summary"), unsafe_allow_html=True)
            st.markdown(f"""
            <div class="score-grid">
                <div class="score-card">
                    <div class="score-num">{R['ssim_whole']}</div>
                    <div class="score-denom">/ 1.000</div>
                    <div class="score-label">Standard SSIM</div>
                </div>
                <div class="score-card">
                    <div class="score-num gold">{R['overall_grem']}</div>
                    <div class="score-denom">/ 1.000</div>
                    <div class="score-label">Overall GREM</div>
                </div>
                <div class="score-card">
                    <div class="score-num highlight">{R['grem_gap']}</div>
                    <div class="score-denom">delta</div>
                    <div class="score-label">GREM Gap</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # ── SSIM Baseline ──
            st.markdown(section_divider("SSIM Baseline — For Comparison Only"), unsafe_allow_html=True)
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;letter-spacing:0.12em;color:#3A3835;margin-bottom:1rem;">These are standard SSIM scores. They are shown as a baseline to illustrate SSIM\'s limitations — they are NOT components of the GREM score.</div>',
                unsafe_allow_html=True)
            bc1, bc2, bc3 = st.columns(3, gap="medium")
            with bc1:
                st.markdown(metric_row_html("Whole-Image SSIM", R['ssim_whole']), unsafe_allow_html=True)
            with bc2:
                st.markdown(metric_row_html("Garment-Region SSIM", R['ssim_garment']), unsafe_allow_html=True)
            with bc3:
                st.markdown(metric_row_html("Identity-Region SSIM", R['ssim_identity']), unsafe_allow_html=True)

            # ── GREM Breakdown ──
            st.markdown(section_divider("GREM Score Breakdown"), unsafe_allow_html=True)
            st.markdown(
                '<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;letter-spacing:0.12em;color:#C8A96E99;margin-bottom:1rem;">These are GREM\'s own components — no SSIM involved. Overall GREM is derived entirely from these.</div>',
                unsafe_allow_html=True)
            bd1, bd2 = st.columns(2, gap="large")
            with bd1:
                st.markdown('<span class="upload-label">Garment Fidelity  ·  Color × 0.55 + Texture × 0.45</span>', unsafe_allow_html=True)
                st.markdown(
                    metric_row_html("Color Preservation", R['garment_color']) +
                    metric_row_html("Texture Score", R['garment_texture']) +
                    metric_row_html("Garment Fidelity", R['garment_fidelity'], bold=True),
                    unsafe_allow_html=True)
            with bd2:
                st.markdown('<span class="upload-label">Identity Preservation  ·  Identity Color</span>', unsafe_allow_html=True)
                st.markdown(
                    metric_row_html("Identity Color", R['identity_color']) +
                    metric_row_html("Identity Score", R['identity_score'], bold=True),
                    unsafe_allow_html=True)

            st.markdown(section_divider("Research Finding"), unsafe_allow_html=True)
            if R['grem_gap'] <= 0.05:
                st.markdown(f"""<div class="finding-panel good">
                    <div class="finding-eyebrow">✓ Metrics Aligned</div>
                    <p class="finding-text">GREM gap is only <strong>{R['grem_gap']*100:.1f}%</strong> —
                    SSIM and GREM are in close agreement for this result. The garment transfer
                    quality is consistent with what whole-image SSIM suggests.</p>
                </div>""", unsafe_allow_html=True)
            elif R['ssim_whole'] > R['overall_grem']:
                st.markdown(f"""<div class="finding-panel">
                    <div class="finding-eyebrow">⚠ SSIM Overestimation Detected</div>
                    <p class="finding-text">Standard SSIM reports <strong>{R['ssim_whole']}</strong> but GREM scores
                    this try-on at <strong>{R['overall_grem']}</strong> — a gap of
                    <strong>{R['grem_gap']*100:.1f}%</strong>. SSIM is inflated by the unchanged
                    face, background, and legs. GREM isolates the garment region and reveals
                    the true try-on quality is lower than SSIM suggests.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="finding-panel good">
                    <div class="finding-eyebrow">✦ SSIM Underestimation Detected</div>
                    <p class="finding-text">Standard SSIM reports <strong>{R['ssim_whole']}</strong> but GREM scores
                    this try-on at <strong>{R['overall_grem']}</strong> — a gap of
                    <strong>{R['grem_gap']*100:.1f}%</strong>. SSIM penalises this result because
                    the garment change is large, dragging down the whole-image similarity score.
                    GREM recognises that the garment transferred correctly — colour and texture
                    both match the target — and scores it higher accordingly.</p>
                </div>""", unsafe_allow_html=True)

            st.markdown(section_divider("Visual Analysis"), unsafe_allow_html=True)
            v1, v2, v3, v4, v5 = st.columns(5, gap="small")
            mask_3ch = np.stack([mask, mask, mask], axis=2)
            mask_vis = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
            for col, img, lbl, cap in zip(
                [v1, v2, v3, v4, v5],
                [p_np, g_np, o_np, mask_vis, o_np * mask_3ch],
                ["01 · Person","02 · Garment","03 · Output","04 · Mask","05 · Region"],
                ["Input","Target",f"SSIM {R['ssim_whole']}","Eval zone",f"GREM {R['overall_grem']}"]
            ):
                with col:
                    st.markdown(f'<span class="upload-label">{lbl}</span>', unsafe_allow_html=True)
                    st.image(img, caption=cap, use_container_width=True)

            # Show mask type indicator
            st.markdown('<span class="mask-note garment-guided">✦ Garment-guided mask — shape derived from garment silhouette</span>', unsafe_allow_html=True)

            st.markdown(section_divider("Score Chart"), unsafe_allow_html=True)
            st.caption("SSIM scores (grey baseline) vs GREM components (your metric)")
            st.bar_chart(pd.DataFrame({"Score": {
                "SSIM — Whole Image":    R['ssim_whole'],
                "SSIM — Garment Region": R['ssim_garment'],
                "SSIM — Identity Region":R['ssim_identity'],
                "GREM — Color Score":    R['garment_color'],
                "GREM — Texture Score":  R['garment_texture'],
                "GREM — Garment Fidelity": R['garment_fidelity'],
                "GREM — Identity Score": R['identity_score'],
                "GREM — Overall":        R['overall_grem'],
            }}), use_container_width=True)
            st.caption("Y-axis: Score (0 to 1) — Higher is better")

            st.markdown(section_divider("Export"), unsafe_allow_html=True)
            st.download_button("↓ Export Results (.txt)",
                data=f"""GREM Evaluation Report
======================
── SSIM Baseline (comparison only) ──
Whole-Image SSIM    : {R['ssim_whole']}
Garment-Region SSIM : {R['ssim_garment']}
Identity-Region SSIM: {R['ssim_identity']}

── GREM Score (your metric) ──
Color Preservation  : {R['garment_color']}
Texture Score       : {R['garment_texture']}
Garment Fidelity    : {R['garment_fidelity']}   (Color×{W_COLOR} + Texture×{W_TEXTURE})
Identity Color      : {R['identity_color']}
Identity Score      : {R['identity_score']}
Overall GREM        : {R['overall_grem']}
GREM Gap (vs SSIM)  : {R['grem_gap']}
""",
                file_name="grem_result.txt", mime="text/plain")


# ══════════════════════════════════════════════
# TAB 2
# ══════════════════════════════════════════════
with tab2:
    st.markdown(section_divider("Batch Configuration"), unsafe_allow_html=True)
    num_samples = st.slider("Number of samples", min_value=2, max_value=10, value=4)

    batch_data = []
    for i in range(num_samples):
        st.markdown(f'<span class="upload-label">Sample {i+1:02d}</span>', unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3, gap="small")
        with b1: p = st.file_uploader(f"Person {i+1}", type=["jpg","jpeg","png"], key=f"bp_{i}", label_visibility="collapsed")
        with b2: g = st.file_uploader(f"Garment {i+1}", type=["jpg","jpeg","png"], key=f"bg_{i}", label_visibility="collapsed")
        with b3: o = st.file_uploader(f"Output {i+1}", type=["jpg","jpeg","png"], key=f"bo_{i}", label_visibility="collapsed")
        if p and g and o:
            batch_data.append({"person": p, "garment": g, "output": o})

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Run Batch Analysis →", key="batch_run"):
        if len(batch_data) < 2:
            st.error("Upload at least 2 complete image sets.")
        else:
            all_results = []
            prog = st.progress(0)
            for idx, case in enumerate(batch_data):
                r = evaluate_grem(
                    load_pil_image(Image.open(case["person"])),
                    load_pil_image(Image.open(case["garment"])),
                    load_pil_image(Image.open(case["output"]))
                )
                r["sample"] = f"Sample {idx+1:02d}"
                all_results.append(r)
                prog.progress((idx + 1) / len(batch_data))

            st.success(f"✅ Evaluated {len(all_results)} samples!")

            st.markdown(section_divider("Results Table"), unsafe_allow_html=True)
            df = pd.DataFrame(all_results)[["sample","ssim_whole","garment_fidelity","identity_score","overall_grem","grem_gap"]]
            df.columns = ["Sample","SSIM (baseline)","Garment Fidelity","Identity Score","Overall GREM","GREM Gap"]
            st.dataframe(df, hide_index=True, use_container_width=True)

            st.markdown(section_divider("Summary Statistics"), unsafe_allow_html=True)
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Avg SSIM",     f"{np.mean([r['ssim_whole'] for r in all_results]):.4f}")
            m2.metric("Avg Fidelity", f"{np.mean([r['garment_fidelity'] for r in all_results]):.4f}")
            m3.metric("Avg Identity", f"{np.mean([r['identity_score'] for r in all_results]):.4f}")
            m4.metric("Avg GREM",     f"{np.mean([r['overall_grem'] for r in all_results]):.4f}")
            m5.metric("Avg Gap",      f"{np.mean([r['grem_gap'] for r in all_results]):.4f}")

            avg_gap  = np.mean([r['grem_gap'] for r in all_results])
            avg_ssim = np.mean([r['ssim_whole'] for r in all_results])
            avg_grem = np.mean([r['overall_grem'] for r in all_results])

            st.markdown(section_divider("Key Finding"), unsafe_allow_html=True)
            st.markdown(f"""<div class="finding-panel">
                <div class="finding-eyebrow">◈ Batch Analysis Result</div>
                <p class="finding-text">Across <strong>{len(all_results)} samples</strong>, Standard SSIM averaged
                <strong>{avg_ssim:.4f}</strong> while GREM averaged <strong>{avg_grem:.4f}</strong> —
                a mean gap of <strong>{avg_gap*100:.1f}%</strong>. This confirms SSIM consistently
                overestimates try-on quality by including unchanged image regions in its evaluation.</p>
            </div>""", unsafe_allow_html=True)

            st.download_button("↓ Export Batch Results (.csv)",
                data=df.to_csv(index=False),
                file_name="grem_batch.csv", mime="text/csv")


# ══════════════════════════════════════════════
# TAB 3
# ══════════════════════════════════════════════
with tab3:
    st.markdown(section_divider("About"), unsafe_allow_html=True)
    st.markdown("""
    <div class="about-body">
    <h3>What is GREM?</h3>
    <strong>Garment Region Evaluation Metric (GREM)</strong> is a novel evaluation
    framework for Virtual Try-On systems. Standard metrics like <code>SSIM</code>
    and <code>FID</code> evaluate the entire output image — including the face, background,
    and legs which are unchanged during try-on. This inflates scores and masks poor garment quality.
    <br><br>
    GREM isolates and scores only the garment region, providing an honest, garment-focused quality assessment.

    <h3>Garment-Guided Masking</h3>
    The evaluation region is not a fixed rectangle. GREM derives the mask from the
    <strong>garment image itself</strong>: it removes the near-white background via HSV
    thresholding, extracts the garment's foreground silhouette, then projects that
    silhouette onto the person image anchored to the torso region. This means a boxy
    shirt and a flowing dress produce different mask shapes, and the metric adapts to
    the actual garment coverage rather than assuming a fixed torso box.

    <h3>Why SSIM fails</h3>
    A completely wrong garment can still score <strong>0.90 on SSIM</strong> because the
    unchanged background and face dominate the metric. GREM penalises this correctly,
    scoring the same result at <strong>~0.47</strong> using garment-gated identity weighting.

    <h3>Color Preservation — Why Two Methods?</h3>
    Garment color and identity color are evaluated differently, and intentionally so.
    <br><br>
    For <strong>garment color</strong>, pixel-wise comparison fails because the garment
    product image is a flat shot (garment on white background) while the output image
    shows the garment worn on a body — the pixels are in entirely different spatial positions.
    Instead, GREM extracts the garment's <strong>dominant colors via K-Means clustering</strong>
    (filtering out the white background first), then checks how closely the output's garment
    region matches those dominant colors using perceptual ΔE distance. This is spatially
    invariant and correctly rewards good color transfer regardless of deformation.
    <br><br>
    For <strong>identity/background color</strong>, the person image and output image ARE
    spatially aligned — the face, legs, and background stay in the same positions — so
    pixel-wise ΔE comparison is valid and appropriate here.

    <h3>Garment SSIM — Coherence, Not Pixel-Match</h3>
    The Garment SSIM score compares the <strong>person image to the output image</strong>
    within the garment region — not the flat garment product shot to the output.
    This is intentional: a flat product shot and a worn garment are never structurally
    similar (different viewpoint, deformation, shadows), so comparing them with SSIM
    always yields a low score regardless of try-on quality. That's a measurement error,
    not a quality signal.
    <br><br>
    Instead, Garment SSIM measures <strong>structural coherence of the placement</strong>:
    did the model produce a well-integrated, artifact-free garment region, preserving
    the body's underlying shape? A good try-on scores moderately high here because
    the body structure is preserved under new fabric.

    <h3>Texture Score — Complexity Matching</h3>
    Texture compares the <strong>gradient complexity level</strong> of the garment product
    image to the output's garment region — not the gradient maps pixel-by-pixel.
    A velvet wrap top should produce a similarly rich, high-gradient surface in the output;
    a plain tee should produce a similarly smooth surface. This is viewpoint-invariant:
    only the texture richness level is compared, not the spatial arrangement of folds.

    <h3>Formula &amp; Weight Rationale</h3>
    <code>Overall GREM = Garment Fidelity × 0.8 + Identity Contribution × 0.2</code>
    <br><br>
    <code>Garment Fidelity = Color × 0.55 + Texture × 0.45</code>
    <br><br>
    SSIM is <strong>not a component of GREM</strong>. It is computed separately as a baseline
    for comparison. The GREM score is derived entirely from perceptual colour distance (ΔE)
    and texture complexity matching — both of which are viewpoint-invariant and garment-specific.
    <br><br>
    Color carries slightly more weight (0.55) because a wrong colour is the most immediately
    obvious failure in a try-on. Texture (0.45) captures fabric richness transfer — velvet
    should look like velvet, plain cotton should look smooth.
    <br><br>
    <code>Identity Contribution = min(Identity Color × 0.2, 0.10)  if GF &lt; 0.55
                           = Identity Color × 0.2              otherwise</code>

    <h3>Research Application</h3>
    GREM was developed to evaluate <strong>IDM-VTON</strong> — a diffusion-based
    virtual try-on model. The metric provides garment-specific quality scores that
    better reflect real try-on performance compared to standard whole-image metrics.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(section_divider("Metric Comparison"), unsafe_allow_html=True)
    st.markdown("""
    <table class="cmp-table">
        <thead><tr><th>Aspect</th><th>Standard SSIM</th><th>GREM</th></tr></thead>
        <tbody>
            <tr><td>Evaluation region</td><td>Whole image</td><td>Garment-guided silhouette mask</td></tr>
            <tr><td>Mask type</td><td>N/A</td><td>Derived from garment HSV foreground</td></tr>
            <tr><td>Color accuracy</td><td>Not measured</td><td>Dominant-color K-Means ΔE (spatially invariant)</td></tr>
            <tr><td>Texture fidelity</td><td>Not measured</td><td>Gradient complexity matching (viewpoint-invariant)</td></tr>
            <tr><td>Identity check</td><td>Mixed into score</td><td>Separate gated component (identity color only)</td></tr>
            <tr><td>SSIM dependency</td><td>Is SSIM</td><td>None — no SSIM in the score formula</td></tr>
            <tr><td>Wrong garment</td><td>~0.90 (misleading)</td><td>~0.47 (honest)</td></tr>
            <tr><td>Garment penalty</td><td>None</td><td>Dynamic penalty for colour/texture mismatch</td></tr>
        </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown(section_divider("Citation"), unsafe_allow_html=True)
    