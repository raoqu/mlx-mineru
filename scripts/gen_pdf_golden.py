#!/usr/bin/env python3
"""Generate PDF rasterization golden by mirroring MinerU's page_to_image exactly.

Contract (mineru/utils/pdf_reader.py::page_to_image):
  scale = dpi / 72; if max(page_size_pt) * scale > 3500: scale = 3500 / max(side)
  bitmap = ceil(width_pt * scale) x ceil(height_pt * scale)   # pypdfium2 uses ceil

Dev-time tool (needs pypdfium2). Freezes per-page dimensions into
tests/golden/pdf_raster.json so the C++ pdfium renderer can be diffed.
"""
import hashlib
import json
import math
import os
import sys

import pypdfium2 as pdfium

DPI = 200
MAX_EDGE = 3500
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.expanduser("~/research/MinerU/demo/pdfs")
OUT = os.path.join(HERE, "tests", "golden", "pdf_raster.json")
REF_PNG_DIR = os.path.join(HERE, "tests", "golden", "pdf_ref")
os.makedirs(REF_PNG_DIR, exist_ok=True)

PDFS = ["demo1.pdf", "small_ocr.pdf"]
MAX_PAGES = 3  # keep golden small/fast

result = {"dpi": DPI, "max_edge": MAX_EDGE, "libpdfium": str(pdfium.version.PDFIUM_INFO),
          "files": {}}

for name in PDFS:
    path = os.path.join(PDF_DIR, name)
    doc = pdfium.PdfDocument(path)
    pages = []
    n = min(len(doc), MAX_PAGES)
    for i in range(n):
        page = doc[i]
        w_pt, h_pt = page.get_size()
        scale = DPI / 72
        long_side = max(w_pt, h_pt)
        if long_side * scale > MAX_EDGE:
            scale = MAX_EDGE / long_side
        img_w = math.ceil(w_pt * scale)
        img_h = math.ceil(h_pt * scale)
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil().convert("RGB")
        assert pil.size == (img_w, img_h), f"{name} p{i}: {pil.size} != {(img_w,img_h)}"
        rgb = pil.tobytes()
        if i == 0:
            pil.save(os.path.join(REF_PNG_DIR, name.replace(".pdf", "_p0.png")))
        # Content metrics robust to anti-aliasing differences across pdfium builds.
        n_px = img_w * img_h
        sum_lum = 0
        ink = 0
        for k in range(0, len(rgb), 3):
            lum = (rgb[k] * 299 + rgb[k + 1] * 587 + rgb[k + 2] * 114) // 1000
            sum_lum += lum
            if lum < 128:
                ink += 1
        pages.append({
            "page_idx": i,
            "width_pt": round(w_pt, 4),
            "height_pt": round(h_pt, 4),
            "scale": round(scale, 10),
            "img_w": img_w,
            "img_h": img_h,
            "rgb_sha256": hashlib.sha256(rgb).hexdigest(),
            "ink_ratio": round(ink / n_px, 6),
            "mean_lum": round(sum_lum / n_px, 4),
        })
        bitmap.close()
    result["files"][name] = {"page_count": len(doc), "pages": pages}
    doc.close()

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)
print("wrote", OUT)
print(json.dumps(result, indent=2)[:1200])
