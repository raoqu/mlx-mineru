#!/usr/bin/env python3
"""Golden for the C++ PP-DocLayoutV2 detector: render a page, resize to 800x800
(save the exact RGB the C++ will consume), run layout.onnx via onnxruntime, apply
the CORE post-processing (box decode -> scale -> topk -> conf>=0.45 -> clip), and
freeze the detections. Isolates onnx+postprocess from rendering.

MinerU's heuristic filters (IoU dedup, formula relabel, paddlex) and reading order
are a follow-up layer; this verifies the core detector exactly.
"""
import json
import os

import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium
from PIL import Image

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "pipeline", "Layout")
GOLDEN = os.path.join(HERE, "tests", "golden")
os.makedirs(GOLDEN, exist_ok=True)

CONF = 0.45
SIZE = 800
cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
id2label = {int(k): v for k, v in cfg["id2label"].items()}

page = pdfium.PdfDocument(os.path.join(HERE, "a.pdf"))[0]
img = page.render(scale=2).to_pil().convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
rgb = np.asarray(img, dtype=np.uint8)
rgb.tofile(os.path.join(GOLDEN, "layout_input.rgb"))

x = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
sess = ort.InferenceSession(os.path.join(MODEL_DIR, "layout.onnx"),
                            providers=["CPUExecutionProvider"])
logits, pred_boxes = sess.run(["logits", "pred_boxes"], {"pixel_values": x})

boxes = pred_boxes[0]
c, d = boxes[:, :2], boxes[:, 2:]
corners = np.concatenate([c - 0.5 * d, c + 0.5 * d], axis=-1) * SIZE
scores = 1.0 / (1.0 + np.exp(-logits[0]))
nq, ncl = scores.shape
flat = scores.reshape(-1)
idx = np.argsort(-flat, kind="stable")[:nq]
dets = []
for fi in idx:
    s = float(flat[fi])
    if s < CONF:
        continue
    label_id, q = int(fi % ncl), int(fi // ncl)
    x0, y0, x1, y1 = corners[q]
    bbox = [max(0, min(SIZE, int(np.floor(x0)))), max(0, min(SIZE, int(np.floor(y0)))),
            max(0, min(SIZE, int(np.ceil(x1)))), max(0, min(SIZE, int(np.ceil(y1))))]
    dets.append({"cls_id": label_id, "label": id2label[label_id],
                 "score": round(s, 4), "bbox": bbox})

out = {"size": SIZE, "conf": CONF, "input_rgb": "layout_input.rgb", "detections": dets}
json.dump(out, open(os.path.join(GOLDEN, "layout_det.json"), "w"), indent=2)
print(f"wrote layout_det.json: {len(dets)} detections")
for de in dets[:8]:
    print(" ", de["label"], de["score"], de["bbox"])
