import streamlit as st
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import pandas as pd
import os

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GREM - Garment Region Evaluation Metric",
    page_icon="👗",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea22, #764ba222);
        border: 1px solid #667eea44;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #667eea;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        margin-top: 0.2rem;
    }
    .finding-box {
        background: linear-gradient(135deg, #11998e22, #38ef7d22);
        border: 1px solid #11998e44;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #f7971e22, #ffd20022);
        border: 1px solid #f7971e44;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# GREM Core Functions
# ─────────────────────────────────────────────

def load_pil_image(pil_img, size=(768, 1024)):
    img = pil_img.convert("RGB").resize(size)
    return np.array(img)

def get_garment_mask(person_img):
    h, w = person_img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    top    = int(h * 0.15)
    bottom = int(h * 0.70)
    left   = int(w * 0.20)
    right  = int(w * 0.80)
    mask[top:bottom, left:right] = 1
    return mask

def calculate_ssim_region(img1, img2, mask):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    r1 = img1 * mask_3ch
    r2 = img2 * mask_3ch
    score = ssim(r1, r2, channel_axis=2, data_range=255)
    return round(float(score), 4)

def calculate_color_preservation(img1, img2, mask):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    pixels1 = img1[mask == 1].astype(float)
    pixels2 = img2[mask == 1].astype(float)
    if len(pixels1) == 0:
        return 0.0
    color_diff = np.mean(np.abs(pixels1 - pixels2))
    return round(float(1.0 - color_diff / 255.0), 4)

def calculate_texture_score(img1, img2, mask):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    mask_3ch = np.stack([mask, mask, mask], axis=2)
    gray1 = cv2.cvtColor((img1 * mask_3ch).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor((img2 * mask_3ch).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    hist1 = cv2.calcHist([gray1], [0], mask, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], mask, [256], [0, 256])
    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return round(float(max(0.0, score)), 4)

def evaluate_grem(person_np, garment_np, output_np):
    garment_mask    = get_garment_mask(person_np)
    background_mask = 1 - garment_mask
    ones_mask       = np.ones(person_np.shape[:2], dtype=np.uint8)

    garment_ssim    = calculate_ssim_region(garment_np, output_np, garment_mask)
    garment_color   = calculate_color_preservation(garment_np, output_np, garment_mask)
    garment_texture = calculate_texture_score(garment_np, output_np, garment_mask)
    identity_ssim   = calculate_ssim_region(person_np, output_np, background_mask)
    identity_color  = calculate_color_preservation(person_np, output_np, background_mask)
    standard_ssim   = calculate_ssim_region(person_np, output_np, ones_mask)

    garment_fidelity = round(garment_ssim*0.4 + garment_color*0.3 + garment_texture*0.3, 4)
    identity_score   = round(identity_ssim*0.6 + identity_color*0.4, 4)
    overall_grem = round(garment_fidelity * 0.8 + identity_score * 0.2, 4)
    grem_gap         = round(abs(standard_ssim - overall_grem), 4)

    return {
        "standard_ssim":     standard_ssim,
        "garment_ssim":      garment_ssim,
        "garment_color":     garment_color,
        "garment_texture":   garment_texture,
        "garment_fidelity":  garment_fidelity,
        "identity_ssim":     identity_ssim,
        "identity_color":    identity_color,
        "identity_score":    identity_score,
        "overall_grem":      overall_grem,
        "grem_gap":          grem_gap,
        "garment_mask":      garment_mask,
    }

def get_score_color(score):
    if score >= 0.75:
        return "🟢"
    elif score >= 0.55:
        return "🟡"
    else:
        return "🔴"

def get_quality_label(score):
    if score >= 0.75:
        return "Good"
    elif score >= 0.55:
        return "Moderate"
    else:
        return "Poor"

# ─────────────────────────────────────────────
# UI Layout
# ─────────────────────────────────────────────

st.markdown('<div class="main-title">👗 GREM Evaluation Tool</div>', 
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Garment Region Evaluation Metric — '
    'A novel evaluation metric for Virtual Try-On systems</div>',
    unsafe_allow_html=True
)

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Single Evaluation", 
    "📊 Batch Evaluation", 
    "ℹ️ About GREM"
])

# ─────────────────────────────────────────────
# TAB 1 — Single Evaluation
# ─────────────────────────────────────────────
with tab1:
    st.subheader("Upload Your 3 Images")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**1. Person Image** (Input to IDM-VTON)")
        person_file = st.file_uploader(
            "Upload person image", 
            type=["jpg", "jpeg", "png"],
            key="person"
        )
        if person_file:
            st.image(person_file, use_column_width=True)

    with col2:
        st.markdown("**2. Garment Image** (Input to IDM-VTON)")
        garment_file = st.file_uploader(
            "Upload garment image",
            type=["jpg", "jpeg", "png"],
            key="garment"
        )
        if garment_file:
            st.image(garment_file, use_column_width=True)

    with col3:
        st.markdown("**3. Output Image** (IDM-VTON Result)")
        output_file = st.file_uploader(
            "Upload IDM-VTON output",
            type=["jpg", "jpeg", "png"],
            key="output"
        )
        if output_file:
            st.image(output_file, use_column_width=True)

    # Category selector
    st.markdown("---")
    col_cat, col_btn = st.columns([2, 1])
    with col_cat:
        category = st.selectbox(
            "Garment Category",
            ["western", "ethnic"],
            format_func=lambda x: 
                "👔 Western Garment" if x == "western" else "🥻 Indian Ethnic Garment"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        evaluate_btn = st.button("🚀 Run GREM Evaluation", key="eval_btn")

    # Run evaluation
    if evaluate_btn:
        if not all([person_file, garment_file, output_file]):
            st.error("❌ Please upload all 3 images before evaluating!")
        else:
            with st.spinner("Running GREM evaluation..."):
                # Load images
                person_np  = load_pil_image(Image.open(person_file))
                garment_np = load_pil_image(Image.open(garment_file))
                output_np  = load_pil_image(Image.open(output_file))

                # Run evaluation
                results = evaluate_grem(person_np, garment_np, output_np)
                mask = results["garment_mask"]

            st.success("✅ Evaluation complete!")
            st.markdown("---")

            # ── Scores Section ──
            st.subheader("📊 GREM Scores")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{results['standard_ssim']}</div>
                    <div class="metric-label">Standard SSIM (old metric)</div>
                </div>""", unsafe_allow_html=True)

            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{results['overall_grem']}</div>
                    <div class="metric-label">Overall GREM (your metric)</div>
                </div>""", unsafe_allow_html=True)

            with col_c:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{results['grem_gap']}</div>
                    <div class="metric-label">GREM Gap (SSIM overestimation)</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("---")

            # ── Detailed Breakdown ──
            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("#### 👕 Garment Fidelity")
                st.markdown(
                    f"*How well the garment appears in the output*"
                )
                metrics_data = {
                    "Metric": [
                        "Garment SSIM", 
                        "Color Preservation", 
                        "Texture Score",
                        "**Garment Fidelity**"
                    ],
                    "Score": [
                        results['garment_ssim'],
                        results['garment_color'],
                        results['garment_texture'],
                        results['garment_fidelity']
                    ],
                    "Quality": [
                        get_score_color(results['garment_ssim']) + " " + 
                        get_quality_label(results['garment_ssim']),
                        get_score_color(results['garment_color']) + " " + 
                        get_quality_label(results['garment_color']),
                        get_score_color(results['garment_texture']) + " " + 
                        get_quality_label(results['garment_texture']),
                        get_score_color(results['garment_fidelity']) + " " + 
                        get_quality_label(results['garment_fidelity']),
                    ]
                }
                st.dataframe(
                    pd.DataFrame(metrics_data), 
                    hide_index=True,
                    use_container_width=True
                )

            with col_right:
                st.markdown("#### 👤 Identity Preservation")
                st.markdown(
                    f"*How well the person is preserved in the output*"
                )
                identity_data = {
                    "Metric": [
                        "Identity SSIM",
                        "Identity Color",
                        "**Identity Score**"
                    ],
                    "Score": [
                        results['identity_ssim'],
                        results['identity_color'],
                        results['identity_score']
                    ],
                    "Quality": [
                        get_score_color(results['identity_ssim']) + " " + 
                        get_quality_label(results['identity_ssim']),
                        get_score_color(results['identity_color']) + " " + 
                        get_quality_label(results['identity_color']),
                        get_score_color(results['identity_score']) + " " + 
                        get_quality_label(results['identity_score']),
                    ]
                }
                st.dataframe(
                    pd.DataFrame(identity_data),
                    hide_index=True,
                    use_container_width=True
                )

            st.markdown("---")

            # ── Key Finding ──
            st.subheader("🔑 Key Research Finding")
            if results['grem_gap'] > 0.05:
                st.markdown(f"""
                <div class="finding-box">
                    <h4>⚠️ SSIM is misleading for this result!</h4>
                    <p>Standard SSIM reports <strong>{results['standard_ssim']}</strong> 
                    but GREM reveals the actual garment quality is only 
                    <strong>{results['overall_grem']}</strong></p>
                    <p>SSIM <strong>overestimates quality by 
                    {results['grem_gap']*100:.1f}%</strong> 
                    because it includes unchanged regions like 
                    face and background in its score.</p>
                    <p>Category: <strong>{category.upper()}</strong></p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="warning-box">
                    <h4>✅ SSIM is relatively accurate for this result</h4>
                    <p>GREM gap is only {results['grem_gap']*100:.1f}% — 
                    SSIM and GREM agree on quality.</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # ── Visualizations ──
            st.subheader("🖼️ Visual Analysis")

            viz_cols = st.columns(5)
            labels = [
                "1. Person Input",
                "2. Garment Input", 
                "3. VTON Output",
                "4. Garment Mask",
                "5. Garment Region"
            ]
            captions = [
                "Original person",
                "Target garment",
                f"SSIM: {results['standard_ssim']}",
                "Green = evaluated region",
                f"GREM: {results['overall_grem']}"
            ]

            mask_3ch = np.stack([mask, mask, mask], axis=2)
            garment_region = output_np * mask_3ch

            images_to_show = [
                person_np, garment_np, output_np,
                cv2.cvtColor(
                    (mask * 255).astype(np.uint8), 
                    cv2.COLOR_GRAY2RGB
                ),
                garment_region
            ]

            for i, (col, img, label, cap) in enumerate(
                zip(viz_cols, images_to_show, labels, captions)
            ):
                with col:
                    st.markdown(f"**{label}**")
                    st.image(img, caption=cap, use_column_width=True)

            # ── Bar Chart ──
            st.markdown("---")
            st.subheader("📈 Score Comparison Chart")

            chart_data = pd.DataFrame({
                "Metric": [
                    "Standard SSIM", 
                    "Garment SSIM",
                    "Color Score", 
                    "Texture Score",
                    "Garment Fidelity",
                    "Identity Score",
                    "Overall GREM"
                ],
                "Score": [
                    results['standard_ssim'],
                    results['garment_ssim'],
                    results['garment_color'],
                    results['garment_texture'],
                    results['garment_fidelity'],
                    results['identity_score'],
                    results['overall_grem']
                ]
            })
            st.bar_chart(chart_data.set_index("Metric"))

            # ── Save Results ──
            st.markdown("---")
            st.subheader("💾 Save Results")

            result_text = f"""GREM Evaluation Results
========================
Category: {category}
Standard SSIM:    {results['standard_ssim']}
Garment SSIM:     {results['garment_ssim']}
Color Score:      {results['garment_color']}
Texture Score:    {results['garment_texture']}
Garment Fidelity: {results['garment_fidelity']}
Identity SSIM:    {results['identity_ssim']}
Identity Color:   {results['identity_color']}
Identity Score:   {results['identity_score']}
Overall GREM:     {results['overall_grem']}
GREM Gap:         {results['grem_gap']}
"""
            st.download_button(
                label="📥 Download Results as TXT",
                data=result_text,
                file_name=f"grem_results_{category}.txt",
                mime="text/plain"
            )

# ─────────────────────────────────────────────
# TAB 2 — Batch Evaluation
# ─────────────────────────────────────────────
with tab2:
    st.subheader("Batch Evaluation")
    st.info(
        "Upload multiple sets of images to compare "
        "Western vs Indian Ethnic garment performance"
    )

    num_samples = st.slider(
        "Number of samples to evaluate", 
        min_value=2, max_value=10, value=4
    )

    batch_data = []
    all_valid = True

    for i in range(num_samples):
        st.markdown(f"---")
        st.markdown(f"#### Sample {i+1}")
        
        b_col1, b_col2, b_col3, b_col4 = st.columns([2, 2, 2, 1])
        
        with b_col1:
            p = st.file_uploader(
                f"Person {i+1}", 
                type=["jpg","jpeg","png"],
                key=f"bp_{i}"
            )
        with b_col2:
            g = st.file_uploader(
                f"Garment {i+1}",
                type=["jpg","jpeg","png"],
                key=f"bg_{i}"
            )
        with b_col3:
            o = st.file_uploader(
                f"Output {i+1}",
                type=["jpg","jpeg","png"],
                key=f"bo_{i}"
            )
        with b_col4:
            cat = st.selectbox(
                f"Category {i+1}",
                ["western", "ethnic"],
                key=f"bc_{i}"
            )
        
        if p and g and o:
            batch_data.append({
                "person": p, "garment": g, 
                "output": o, "category": cat
            })
        else:
            all_valid = False

    if st.button("🚀 Run Batch Evaluation", key="batch_btn"):
        if len(batch_data) < 2:
            st.error("Please upload at least 2 complete sets of images!")
        else:
            all_results = []
            progress = st.progress(0)
            
            for idx, case in enumerate(batch_data):
                with st.spinner(f"Evaluating sample {idx+1}..."):
                    p_np = load_pil_image(Image.open(case["person"]))
                    g_np = load_pil_image(Image.open(case["garment"]))
                    o_np = load_pil_image(Image.open(case["output"]))
                    
                    r = evaluate_grem(p_np, g_np, o_np)
                    r["category"] = case["category"]
                    r["sample"] = f"Sample {idx+1}"
                    all_results.append(r)
                    progress.progress((idx+1) / len(batch_data))

            st.success(f"✅ Evaluated {len(all_results)} samples!")

            # Results table
            st.subheader("📊 Results Table")
            df = pd.DataFrame(all_results)[[
                "sample", "category", "standard_ssim", 
                "garment_fidelity", "identity_score",
                "overall_grem", "grem_gap"
            ]]
            df.columns = [
                "Sample", "Category", "SSIM", 
                "Garment Fidelity", "Identity",
                "Overall GREM", "GREM Gap"
            ]
            st.dataframe(df, hide_index=True, use_container_width=True)

            # Summary by category
            st.subheader("📈 Western vs Ethnic Comparison")
            
            western = [r for r in all_results if r["category"] == "western"]
            ethnic  = [r for r in all_results if r["category"] == "ethnic"]

            s_col1, s_col2 = st.columns(2)

            def show_category_stats(col, label, group, emoji):
                with col:
                    if group:
                        st.markdown(f"#### {emoji} {label} (n={len(group)})")
                        avg_ssim = np.mean([r['standard_ssim'] for r in group])
                        avg_grem = np.mean([r['overall_grem'] for r in group])
                        avg_gap  = np.mean([r['grem_gap'] for r in group])
                        avg_gf   = np.mean([r['garment_fidelity'] for r in group])
                        
                        st.metric("Avg SSIM", f"{avg_ssim:.4f}")
                        st.metric("Avg GREM", f"{avg_grem:.4f}")
                        st.metric("Avg Garment Fidelity", f"{avg_gf:.4f}")
                        st.metric("Avg GREM Gap", f"{avg_gap:.4f}")
                    else:
                        st.info(f"No {label} samples uploaded")

            show_category_stats(s_col1, "Western Garments", western, "👔")
            show_category_stats(s_col2, "Indian Ethnic Garments", ethnic, "🥻")

            # Key finding
            if western and ethnic:
                w_grem = np.mean([r['overall_grem'] for r in western])
                e_grem = np.mean([r['overall_grem'] for r in ethnic])
                diff = w_grem - e_grem

                st.markdown("---")
                st.subheader("🔑 Key Research Finding")
                st.markdown(f"""
                <div class="finding-box">
                    <h4>IDM-VTON performs {diff*100:.1f}% worse on Indian ethnic wear!</h4>
                    <p>Western GREM: <strong>{w_grem:.4f}</strong></p>
                    <p>Ethnic GREM: <strong>{e_grem:.4f}</strong></p>
                    <p>This performance gap is your core research contribution —
                    demonstrating that IDM-VTON has a significant bias towards 
                    Western garments and performs poorly on Indian ethnic wear.</p>
                </div>
                """, unsafe_allow_html=True)

            # Download CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="grem_batch_results.csv",
                mime="text/csv"
            )

# ─────────────────────────────────────────────
# TAB 3 — About GREM
# ─────────────────────────────────────────────
with tab3:
    st.subheader("About GREM")
    
    st.markdown("""
    ### What is GREM?
    
    **Garment Region Evaluation Metric (GREM)** is a novel evaluation metric 
    proposed for Virtual Try-On (VTON) systems. Unlike standard metrics like 
    SSIM and FID that evaluate the entire image, GREM focuses specifically on 
    the garment region — the part that actually matters for try-on quality.
    
    ---
    
    ### Why is GREM needed?
    
    Standard SSIM is broken for VTON evaluation:
    - It measures the **whole image** including background, face, and legs
    - These regions don't change during try-on, so they inflate the score
    - A plain white image can score **86/100 on SSIM** (TryOffDiff, 2024)
    - SSIM cannot detect if the garment is blurry, miscolored, or misaligned
    
    ---
    
    ### How GREM works
```
    GREM = Garment Fidelity (60%) + Identity Preservation (40%)
    
    Garment Fidelity  = Garment SSIM (40%) 
                      + Color Score (30%) 
                      + Texture Score (30%)
    
    Identity Score    = Identity SSIM (60%) 
                      + Identity Color (40%)
```
    
    ---
    
    ### GREM vs Standard SSIM
    
    | Aspect | SSIM | GREM |
    |--------|------|------|
    | Evaluation region | Whole image | Garment only |
    | Color accuracy | ❌ Not measured | ✅ Measured |
    | Texture fidelity | ❌ Not measured | ✅ Measured |
    | Identity check | ❌ Mixed with garment | ✅ Separate score |
    | Garment bias | ❌ Inflated by background | ✅ Focused |
    
    ---
    
    ### Research Application
    
    GREM was validated by evaluating IDM-VTON on:
    - **Western garments** (t-shirts, shirts, dresses)
    - **Indian ethnic garments** (kurtas, sarees, salwar suits)
    
    Results show GREM reveals a larger performance gap between 
    Western and ethnic garments compared to standard SSIM — 
    demonstrating SSIM's inability to accurately assess 
    VTON quality on diverse garment types.
    
    ---
    
    ### Citation
```
    GREM: Garment Region Evaluation Metric for Virtual Try-On
    Systems with Application to Indian Ethnic Wear
    MCA Research Project, 2024-25
```
    """)