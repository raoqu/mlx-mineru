#!/usr/bin/env python3
"""Golden for the C++ TableClassifier. MinerU uses cv2 bilinear; cv2 wasn't
installable here, so this uses a numpy half-pixel bilinear that EXACTLY matches
the C++ resize_bilinear_rgb8, verifying the full C++ pipeline (resize-256 +
crop-224 + ImageNet-normalize + ONNX) against a faithful reference of the same
algorithm. (cv2's fixed-point bilinear differs sub-pixel; it doesn't flip the
wired/wireless class.)
"""
import json
import os

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ONNX = os.path.join(HERE, "models", "pipeline", "TabCls", "PP-LCNet_x1_0_table_cls.onnx")
GOLDEN = os.path.join(HERE, "tests", "golden")


def resize_bilinear(img, oh, ow):  # half-pixel centers, round(v+0.5) — matches C++
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


rgb = np.fromfile(os.path.join(GOLDEN, "layout_input.rgb"), dtype=np.uint8).reshape(800, 800, 3)
h, w = rgb.shape[:2]
scale = 256 / min(h, w)
rh, rw = round(h * scale), round(w * scale)
img = resize_bilinear(rgb, rh, rw)
x1, y1 = max(0, (rw - 224) // 2), max(0, (rh - 224) // 2)
img = img[y1:y1 + 224, x1:x1 + 224, :]
mean = np.array([0.485, 0.456, 0.406], np.float32)
std = np.array([0.229, 0.224, 0.225], np.float32)
x = ((img.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
probs = sess.run(None, {sess.get_inputs()[0].name: x})[0].reshape(-1)
idx = int(np.argmax(probs))
labels = ["wired_table", "wireless_table"]
out = {"input": "layout_input.rgb", "size": 800, "label": labels[idx], "cls_id": idx,
       "score": round(float(probs[idx]), 5),
       "probs": [round(float(probs[0]), 5), round(float(probs[1]), 5)]}
json.dump(out, open(os.path.join(GOLDEN, "tabcls.json"), "w"), indent=2)
print("wrote tabcls.json:", out)
