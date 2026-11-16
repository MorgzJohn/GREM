"""check_rows.py -- draws person | garment | output for every row of the CSV so you can confirm the mapping by eye.
Usage:  python3 check_rows.py ~/Downloads/outfits_prefilled.csv "/home/morgz/Documents/GREM/SAMPLE IMAGES"
Writes check_rows_1.jpg, check_rows_2.jpg, ... in the current folder (5 rows per sheet)."""
import csv, sys
from pathlib import Path
from PIL import Image, ImageDraw

csv_path, base = sys.argv[1], Path(sys.argv[2])
rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
w, h, per = 210, 280, 5
for s in range(0, len(rows), per):
    chunk = rows[s:s + per]
    sheet = Image.new("RGB", (3 * (w + 10), len(chunk) * (h + 24)), "white"); d = ImageDraw.Draw(sheet)
    for i, r in enumerate(chunk):
        y = i * (h + 24)
        d.text((4, y + 4), f"SPREADSHEET ROW {s + i + 2}:  person={r['person_file']}   garment={r['garment_file']}   output={r['output_file']}", fill="black")
        for j, (sub, name) in enumerate((("Model", r["person_file"]), ("garment", r["garment_file"]), ("generated Img", r["output_file"]))):
            if not name.strip():
                d.text((j * (w + 10) + 4, y + 40), f"NOT FILLED YET\n({sub} column is empty)", fill="red"); continue
            try:
                im = Image.open(base / sub / name).convert("RGB"); im.thumbnail((w, h)); sheet.paste(im, (j * (w + 10), y + 20))
            except (FileNotFoundError, IsADirectoryError, OSError):
                d.text((j * (w + 10) + 4, y + 40), f"MISSING FILE\n{sub}/{name}", fill="red")
    sheet.save(f"check_rows_{s // per + 1}.jpg")
print("wrote", -(-len(rows) // per), "sheet(s)")
