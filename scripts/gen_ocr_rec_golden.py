#!/usr/bin/env python3
"""Golden for the C++ TextRecognizer: render a.pdf p0, crop a layout 'text' box,
run ocr_rec.onnx with a numpy half-pixel bilinear (matches C++ resize_bilinear_rgb8)
+ the exact CTC greedy decode. Saves the crop RGB + expected text/score.
"""
import json
import os

import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR = os.path.join(HERE, "models", "pipeline", "OCR")
GOLDEN = os.path.join(HERE, "tests", "golden")
H, WB, MINW, MAXW = 48, 320, 16, 2560


def resize_bilinear(img, oh, ow):  # matches C++ resize_bilinear_rgb8
    ih, iw = img.shape[:2]
    sy, sx = ih / oh, iw / ow
    ys = (np.arange(oh) + 0.5) * sy - 0.5
    xs = (np.arange(ow) + 0.5) * sx - 0.5
    y0 = np.floor(ys).astype(int); wy = ys - y0
    x0 = np.floor(xs).astype(int); wx = xs - x0
    y0c, y1c = np.clip(y0, 0, ih - 1), np.clip(y0 + 1, 0, ih - 1)
    x0c, x1c = np.clip(x0, 0, iw - 1), np.clip(x0 + 1, 0, iw - 1)
    f = img.astype(np.float64)
    top = f[np.ix_(y0c, x0c)] * (1 - wx)[None, :, None] + f[np.ix_(y0c, x1c)] * wx[None, :, None]
    bot = f[np.ix_(y1c, x0c)] * (1 - wx)[None, :, None] + f[np.ix_(y1c, x1c)] * wx[None, :, None]
    out = top * (1 - wy)[:, None, None] + bot * wy[:, None, None]
    return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)


# character array: blank + dict + space
chars = ["blank"]
with open(os.path.join(OCR, "ppocrv6_dict.txt"), "rb") as fin:
    chars += [ln.decode("utf-8").rstrip("\r\n") for ln in fin]
chars.append(" ")

# Render a.pdf p0 and crop the first layout 'text' box (coords in 800-space).
page = pdfium.PdfDocument(os.path.join(HERE, "a.pdf"))[0]
img = np.asarray(page.render(scale=2).to_pil().convert("RGB"))
PH, PW = img.shape[:2]
det = json.load(open(os.path.join(GOLDEN, "layout_det.json")))
# pick the most line-shaped detection (highest width/height) so rec sees one line.
cand = [d["bbox"] for d in det["detections"]]
box = max(cand, key=lambda b: (b[2] - b[0]) / max(1, b[3] - b[1]))
sx, sy = PW / 800.0, PH / 800.0
x0, y0, x1, y1 = int(box[0] * sx), int(box[1] * sy), int(box[2] * sx), int(box[3] * sy)
crop = np.ascontiguousarray(img[y0:y1, x0:x1, :])
ch, cw = crop.shape[:2]
crop.tofile(os.path.join(GOLDEN, "ocr_rec_input.rgb"))

# preprocess
max_wh = max(cw / ch, WB / H)
imgW = max(MINW, min(MAXW, int(H * max_wh)))
rw = min(imgW, max(int(np.ceil(H * cw / ch)), MINW))
resized = resize_bilinear(crop, H, rw)[:, :, ::-1]  # RGB crop -> BGR (MinerU recognizes BGR)
x = np.zeros((1, 3, H, imgW), np.float32)
x[0, :, :, :rw] = (resized.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)

sess = ort.InferenceSession(os.path.join(OCR, "ocr_rec.onnx"), providers=["CPUExecutionProvider"])
logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]  # (T, C)
idx = logits.argmax(1)
text, score, kept, prev = "", 0.0, 0, -1
for t in range(len(idx)):
    b = int(idx[t])
    if b != 0 and b != prev:
        row = logits[t]
        score += float(np.exp(0.0) / np.sum(np.exp(row - row[b])))
        text += chars[b]
        kept += 1
    prev = b
score = score / kept if kept else 0.0

out = {"input": "ocr_rec_input.rgb", "w": cw, "h": ch, "text": text, "score": round(score, 5)}
json.dump(out, open(os.path.join(GOLDEN, "ocr_rec.json"), "w"), ensure_ascii=False, indent=2)
print("wrote ocr_rec.json:", json.dumps(out, ensure_ascii=False))
