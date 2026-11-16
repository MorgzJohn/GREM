# GREM: Garment Region Evaluation Metric for Virtual Try-On

GREM is a lightweight, training-free score for virtual try-on (VTON) outputs. It looks at an *estimated garment region* and compares
its colour and texture with the reference garment. It needs three images: the original person photo, the reference (product) garment
photo, and the generated try-on output. It needs **no ground-truth try-on image, no training data and no GPU**.

> Status: research code accompanying a manuscript. GREM has **not** been validated against human judgments, and the garment region is
> a heuristic (a fixed torso window filled with the silhouette of the product photo), not a segmentation of the worn garment.

## Score
* **CP** (colour): K-Means (k = 3) on the reference garment pixels (pixels with R, G and B all above 220 dropped); for every output pixel in the region the
  smallest CIEDE2000 distance to those three colours; `CP = exp(-mean dE / 40)`.
* **TF** (texture): ratio `r` of mean Sobel gradient magnitude (output region / reference at the same pixel positions); `TF = exp(-|r-1| / (0.3 (r+1)))`.
* **NP** (non-garment colour): `NP = exp(-mean dE / 30)` between person photo and output outside the region.
* `GF_raw = 0.55 CP + 0.45 TF`; penalty `p = 0.12` if CP < 0.3 and TF < 0.3, else `0.06` if CP < 0.4, else 0; `GF = max(0, GF_raw - p)`.
* `GREM = 0.8 GF + 0.2 NP` if `GF >= 0.55`, otherwise `0.8 GF + min(0.2 NP, 0.10)`.

SSIM (whole image, person photo vs output) is reported only as a baseline and is **not** part of GREM.

## Use
**Web app:** https://grem-vton.streamlit.app (source: `grm_app.py`).

**Python:**
```python
import sys; sys.path.insert(0, "code")          # note: a folder called "code" clashes with a standard-library module name
from PIL import Image
from grem_core import load_pil_image, evaluate_grem
person  = load_pil_image(Image.open("person.png"))
garment = load_pil_image(Image.open("garment.jpg"))
output  = load_pil_image(Image.open("output.png"))
r = evaluate_grem(person, garment, output)
print(r["overall_grem"], r["garment_color"], r["garment_texture"], r["identity_score"], r["ssim_whole"])
```
All images are resized to 768 x 1024.

## Repository layout
```
grm_app.py                  Streamlit app (self-contained)
requirements.txt            what the Streamlit app needs
requirements-experiments.txt, requirements-pinned.txt
code/grem_core.py           the scoring functions (same code as the app)
expanded_experiments/       the sensitivity study reported in the paper (see its README)
```
The input photographs are **not** included (some may not be redistributable). Configuration, per-sample scores and analysis scripts are.

## Reproducing the reported numbers
See `expanded_experiments/README.md`. The analysis and figures can be re-run from the files in the repository without any images:
`python3 expanded_experiments/scripts/04_analyze.py` then `05_figures.py` and `06_paper_figure.py`.
