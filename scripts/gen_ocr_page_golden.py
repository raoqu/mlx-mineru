#!/usr/bin/env python3
"""Golden for the C++ OcrPipeline (full page OCR). Runs MinerU's ACTUAL chain on
a.pdf p0: det (ocr_det.onnx) -> sorted_boxes -> merge_det_boxes -> rotate-crop ->
batched CTC rec (ocr_rec.onnx, predict_rec batching) -> drop_score 0.5. Feeds the
models BGR (cv2 order), exactly like mineru/backend/pipeline. Reuses the page RGB
fixture saved by gen_ocr_det_golden.py (ocr_det_input.rgb).
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium

MINERU = os.path.expanduser("~/research/MinerU")
sys.path.insert(0, MINERU)
from mineru.model.utils.pytorchocr.data import create_operators, transform  # noqa: E402
from mineru.model.utils.pytorchocr.postprocess.db_postprocess import DBPostProcess  # noqa: E402
from mineru.utils.ocr_utils import (  # noqa: E402
    sorted_boxes, merge_det_boxes, get_rotate_crop_image_for_text_rec)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR = os.path.join(HERE, "models", "pipeline", "OCR")
GOLDEN = os.path.join(HERE, "tests", "golden")
H, WB, MINW, MAXW, BATCH, DROP = 48, 320, 16, 2560, 6, 0.5

pre_ops = create_operators([
    {"DetResizeForTest": {"limit_side_len": 960, "limit_type": "max", "max_side_limit": 4000}},
    {"NormalizeImage": {"std": [0.229, 0.224, 0.225], "mean": [0.485, 0.456, 0.406],
                        "scale": "1./255.", "order": "hwc"}},
    {"ToCHWImage": None},
    {"KeepKeys": {"keep_keys": ["image", "shape"]}},
])
post = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5,
                     use_dilation=False, score_mode="fast", box_type="quad")

chars = ["blank"]
with open(os.path.join(OCR, "ppocrv6_dict.txt"), "rb") as fin:
    chars += [ln.decode("utf-8").rstrip("\r\n") for ln in fin]
chars.append(" ")

rgb = np.asarray(pdfium.PdfDocument(os.path.join(HERE, "a.pdf"))[0].render(scale=2).to_pil().convert("RGB"))
bgr = np.ascontiguousarray(rgb[:, :, ::-1])  # cv2 / pipeline channel order

# ---- detection ----
data = transform({"image": bgr.copy(), "polys": np.zeros((0, 4, 2), np.float32)}, pre_ops)
x, shape_list = data[0][None], np.array([data[1]])
det_sess = ort.InferenceSession(os.path.join(OCR, "ocr_det.onnx"), providers=["CPUExecutionProvider"])
pred = det_sess.run(None, {det_sess.get_inputs()[0].name: x.astype(np.float32)})[0]
dt_boxes = post({"maps": pred}, shape_list)[0]["points"]
dt_boxes = [np.array(b).reshape(-1, 2).astype(np.float32) for b in dt_boxes]

# ---- MinerU box ordering + merge ----
dt_boxes = sorted_boxes(dt_boxes)
dt_boxes = merge_det_boxes(dt_boxes)  # enable_merge_det_boxes=True (text path default)

# ---- crops (rotate-for-text-rec), from the BGR image ----
crops = [get_rotate_crop_image_for_text_rec(bgr, np.array(b, dtype=np.float32).copy()) for b in dt_boxes]

# ---- batched CTC rec (predict_rec batching) ----
rec_sess = ort.InferenceSession(os.path.join(OCR, "ocr_rec.onnx"), providers=["CPUExecutionProvider"])
widths = [c.shape[1] / float(c.shape[0]) for c in crops]
order = np.argsort(np.array(widths))
results = [("", 0.0)] * len(crops)
for beg in range(0, len(crops), BATCH):
    end = min(len(crops), beg + BATCH)
    max_wh = widths[order[end - 1]]
    for ino in range(beg, end):
        c = crops[order[ino]]
        ch, cw = c.shape[:2]
        mwr = max(max_wh, WB / H)
        imgW = max(min(int(H * mwr), MAXW), MINW)
        ratio = cw / float(ch)
        rw = min(imgW, int(max(math.ceil(H * ratio), MINW)))
        resized = cv2.resize(c, (rw, H)) / 127.5 - 1.0  # BGR already
        inp = np.zeros((1, 3, H, imgW), np.float32)
        inp[0, :, :, :rw] = resized.transpose(2, 0, 1)
        logits = rec_sess.run(None, {rec_sess.get_inputs()[0].name: inp})[0][0]
        idx = logits.argmax(1)
        text, score, kept, prev = "", 0.0, 0, -1
        for t in range(len(idx)):
            b = int(idx[t])
            if b != 0 and b != prev:
                row = logits[t]
                score += float(1.0 / np.sum(np.exp(row - row[b])))
                text += chars[b]
                kept += 1
            prev = b
        results[order[ino]] = (text, score / kept if kept else 0.0)

lines = []
for box, (text, score) in zip(dt_boxes, results):
    if score >= DROP:
        lines.append({"box": np.array(box).astype(float).tolist(), "text": text,
                      "score": round(float(score), 5)})

src_h, src_w = rgb.shape[:2]
out = {"input": "ocr_det_input.rgb", "src_w": int(src_w), "src_h": int(src_h),
       "drop_score": DROP, "lines": lines}
json.dump(out, open(os.path.join(GOLDEN, "ocr_page.json"), "w"), ensure_ascii=False, indent=1)
print(f"wrote ocr_page.json: {len(lines)} lines (of {len(dt_boxes)} boxes)")
for ln in lines[:6]:
    b = ln["box"]
    print(f"  [{b[0][0]:.0f},{b[0][1]:.0f}]-[{b[2][0]:.0f},{b[2][1]:.0f}] {ln['score']:.3f} {ln['text']}")
