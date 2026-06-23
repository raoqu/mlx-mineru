#!/usr/bin/env python3
"""Golden for Qwen2-VL image preprocessing using the REAL transformers
Qwen2VLImageProcessor. Uses a deterministic procedural image (saved as raw RGB)
so the C++ test consumes identical input and the comparison isolates
preprocessing (resize+normalize+patchify) from image decoding."""
import json
import os

import numpy as np
from transformers import AutoImageProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOK_DIR = os.path.join(HERE, "models", "MinerU2.5-tokenizer")
GOLDEN = os.path.join(HERE, "tests", "golden")
os.makedirs(GOLDEN, exist_ok=True)

W, H = 280, 196
img = np.zeros((H, W, 3), dtype=np.uint8)
for y in range(H):
    for x in range(W):
        for c in range(3):
            img[y, x, c] = (x * 7 + y * 13 + c * 53 + (x ^ y)) % 256
img.tofile(os.path.join(GOLDEN, "preprocess_input.rgb"))

# Slow processor: PIL bicubic resize (what the C++ port mirrors faithfully).
proc = AutoImageProcessor.from_pretrained(TOK_DIR, trust_remote_code=True, use_fast=False)
out = proc(images=[img], return_tensors="np")
pv = np.asarray(out["pixel_values"]).astype(np.float64)  # [seq, 1176]
grid = [int(v) for v in np.asarray(out["image_grid_thw"]).reshape(-1)[:3]]

flat = pv.reshape(-1)
# 256 deterministic sample indices spread across the tensor.
n = flat.size
idxs = [int(i * (n - 1) // 255) for i in range(256)]
samples = [[i, float(flat[i])] for i in idxs]

cfg = json.load(open(os.path.join(TOK_DIR, "preprocessor_config.json")))
factor = cfg["patch_size"] * cfg["merge_size"]
sr_cases = []
for (h, w) in [(196, 280), (743, 544), (2339, 1654), (100, 100), (30, 4000), (4000, 30)]:
    try:
        hb, wb = smart_resize(h, w, factor, cfg["min_pixels"], cfg["max_pixels"])
        sr_cases.append({"h": h, "w": w, "h_bar": int(hb), "w_bar": int(wb)})
    except Exception as e:
        sr_cases.append({"h": h, "w": w, "error": True})

result = {
    "input_rgb": "preprocess_input.rgb",
    "in_w": W, "in_h": H,
    "grid_thw": grid,
    "feat_dim": int(pv.shape[1]),
    "seq_len": int(pv.shape[0]),
    "mean": float(flat.mean()),
    "std": float(flat.std()),
    "min": float(flat.min()),
    "max": float(flat.max()),
    "samples": samples,
    "smart_resize": sr_cases,
    "config": {k: cfg[k] for k in ("min_pixels", "max_pixels", "patch_size",
                                    "temporal_patch_size", "merge_size")},
}
with open(os.path.join(GOLDEN, "preprocess.json"), "w") as f:
    json.dump(result, f, indent=2)
print("grid_thw", grid, "seq", pv.shape[0], "mean %.5f std %.5f" % (flat.mean(), flat.std()))
print("smart_resize:", [(c.get("h"), c.get("w"), c.get("h_bar"), c.get("w_bar")) for c in sr_cases])
