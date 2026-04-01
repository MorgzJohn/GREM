# GREM — Garment Region Evaluation Metric

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Research](https://img.shields.io/badge/Research-MCA%20Final%20Year-purple)

> A novel evaluation metric for Virtual Try-On (VTON) systems that 
> focuses on garment region quality rather than whole-image similarity.

---

## 🔍 What is GREM?

Standard metrics like **SSIM** and **FID** evaluate the entire image 
when scoring VTON outputs — including the background, face, and legs 
which don't change during try-on. This inflates scores and hides poor 
garment quality.

**GREM** fixes this by isolating and scoring only the garment region.
```
Standard SSIM → scores whole image → misleading
GREM          → scores garment only → honest
```

### Key finding
A completely wrong garment can still score **0.90 on SSIM** but only 
**0.52 on GREM** — proving SSIM is unreliable for VTON evaluation.

---

## 📊 How GREM Works
```
GREM = Garment Fidelity (80%) + Identity Preservation (20%)

Garment Fidelity = Garment SSIM (40%)
                 + Color Preservation (30%)
                 + Texture Score (30%)

Identity Score   = Identity SSIM (60%)
                 + Identity Color (40%)
```

### Inputs Required
| Input | Description |
|-------|-------------|
| Person Image | Original person photo (input to IDM-VTON) |
| Garment Image | Target garment photo (input to IDM-VTON) |
| Output Image | Generated try-on result (IDM-VTON output) |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/GREM-VTON.git
cd GREM-VTON
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run grem_app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 🧪 Research Application

This metric was developed as part of an MCA final year research project 
evaluating **IDM-VTON** on Indian ethnic wear.

### Research Question
*Does IDM-VTON perform differently on Indian ethnic garments compared 
to Western garments, and can GREM reveal this difference better than 
standard SSIM?*

### Garment Categories Tested
- 👔 Western garments (t-shirts, shirts, dresses)
- 🥻 Indian ethnic garments (kurtas, sarees, salwar suits, lehengas)

---

## 📁 Project Structure
```
GREM-VTON/
├── grem_app.py       ← Streamlit web interface
├── grem_core.py      ← Core metric functions (reusable)
├── vton.py           ← Command-line evaluation script
├── requirements.txt  ← Python dependencies
├── README.md         ← This file
├── .gitignore        ← Git ignore rules
└── samples/          ← Sample images for testing
```

---

## 📖 Usage

### Streamlit App (Recommended)
Upload images through the web interface and get instant scores 
with visualizations.

### Python Script
```python
from grem_core import load_image, evaluate_grem
from PIL import Image

person  = load_image(Image.open("person.jpg"))
garment = load_image(Image.open("garment.jpg"))
output  = load_image(Image.open("output.jpg"))

results = evaluate_grem(person, garment, output)

print(f"Standard SSIM:  {results['standard_ssim']}")
print(f"Overall GREM:   {results['overall_grem']}")
print(f"GREM Gap:       {results['grem_gap']}")
```

### Command Line
```bash
python vton.py
```

---

## 📈 Score Interpretation

| Score Range | Quality |
|-------------|---------|
| 0.75 — 1.00 | 🟢 Good |
| 0.55 — 0.74 | 🟡 Moderate |
| 0.00 — 0.54 | 🔴 Poor |

---

## 🔬 Related Work

- IDM-VTON (Choi et al., ECCV 2024) — Base VTON model evaluated
- TryOffDiff (2024) — Identifies SSIM limitations for garment evaluation
- VTONQA (2026) — Human opinion scoring for VTON
- Mask-SSIM (Çoğalan et al., 2023) — Masked SSIM in medical imaging

---

## 📄 Citation
```bibtex
@misc{grem2025,
  title   = {GREM: Garment Region Evaluation Metric for Virtual 
             Try-On with Application to Indian Ethnic Wear},
  author  = {Morgan John},
  year    = {2025},
  note    = {MCA Final Year Research Project}
}
```

---

## 🙏 Acknowledgements

- [IDM-VTON](https://github.com/yisol/IDM-VTON) by Yisol Choi et al.
- [Streamlit](https://streamlit.io) for the web framework
- VITON-HD dataset for evaluation reference