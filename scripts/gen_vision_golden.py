#!/usr/bin/env python3
"""Golden for the Qwen2-VL vision tower using the REAL transformers model.
Runs the vision encoder on the SAME deterministic image used by the preprocess
golden (preprocessing is bit-exact, so pixel_values match) and freezes merged
image-embed stats + samples. bf16 -> compared with tolerance."""
import json
import os

import numpy as np
import torch
from transformers import AutoImageProcessor, Qwen2VLForConditionalGeneration

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "MinerU2.5-tokenizer")
GOLDEN = os.path.join(HERE, "tests", "golden")

W, H = 280, 196
rgb = np.fromfile(os.path.join(GOLDEN, "preprocess_input.rgb"), dtype=np.uint8).reshape(H, W, 3)

proc = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
enc = proc(images=[rgb], return_tensors="pt")
pixel_values = enc["pixel_values"].to(torch.bfloat16)
grid_thw = enc["image_grid_thw"]
grid = [int(v) for v in grid_thw.reshape(-1)[:3]]

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).eval()
visual = getattr(model, "visual", None) or getattr(model.model, "visual")

with torch.no_grad():
    embeds = visual(pixel_values, grid_thw=grid_thw).float().numpy()

flat = embeds.reshape(-1)
n = flat.size
idxs = [int(i * (n - 1) // 255) for i in range(256)]
out = {
    "grid_thw": grid,
    "embed_shape": list(embeds.shape),
    "mean": float(flat.mean()),
    "std": float(flat.std()),
    "min": float(flat.min()),
    "max": float(flat.max()),
    "samples": [[i, float(flat[i])] for i in idxs],
}
with open(os.path.join(GOLDEN, "vision.json"), "w") as f:
    json.dump(out, f, indent=2)
print("grid", grid, "embeds", embeds.shape, "mean %.5f std %.5f" % (flat.mean(), flat.std()))
