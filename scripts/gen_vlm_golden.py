#!/usr/bin/env python3
"""Golden for the end-to-end multimodal Qwen2-VL forward (vision + merge + LLM)
using the REAL transformers model. Constructs input_ids with image placeholders,
runs the model with pixel_values, and records greedy generation (with per-step
top-2 gaps so the C++ test tolerates bf16 near-ties)."""
import json
import os

import numpy as np
import torch
from transformers import (AutoImageProcessor, AutoTokenizer,
                          Qwen2VLForConditionalGeneration)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "MinerU2.5-tokenizer")
GOLDEN = os.path.join(HERE, "tests", "golden")

W, H = 280, 196
rgb = np.fromfile(os.path.join(GOLDEN, "preprocess_input.rgb"), dtype=np.uint8).reshape(H, W, 3)

proc = AutoImageProcessor.from_pretrained(MODEL_DIR, trust_remote_code=True, use_fast=False)
tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
enc = proc(images=[rgb], return_tensors="pt")
pv = enc["pixel_values"].to(torch.bfloat16)
grid_thw = enc["image_grid_thw"]
grid = [int(v) for v in grid_thw.reshape(-1)[:3]]
n_img = (grid[0] * grid[1] * grid[2]) // (proc.merge_size ** 2)

IMAGE_TOKEN = 151655
V_START, V_END = 151652, 151653
prefix = [V_START] + [IMAGE_TOKEN] * n_img + [V_END]
suffix = tok.encode("\nText Recognition:", add_special_tokens=False)
input_ids = prefix + suffix

model = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_DIR, dtype=torch.bfloat16).eval()

EOS = [151645, 151643]
cur = list(input_ids)
greedy = []
with torch.no_grad():
    for _ in range(12):
        out = model(input_ids=torch.tensor([cur]), pixel_values=pv, image_grid_thw=grid_thw)
        lg = out.logits[0, -1].float()
        t2 = torch.topk(lg, 2).values.tolist()
        nxt = int(lg.argmax())
        greedy.append({"token": nxt, "gap": float(t2[0] - t2[1])})
        if nxt in EOS:
            break
        cur.append(nxt)

result = {
    "grid_thw": grid, "n_img": n_img, "input_ids": input_ids,
    "eos": EOS, "greedy": greedy,
}
with open(os.path.join(GOLDEN, "vlm.json"), "w") as f:
    json.dump(result, f, indent=2)
print("grid", grid, "n_img", n_img, "len(input_ids)", len(input_ids))
print("greedy:", [(g["token"], round(g["gap"], 3)) for g in greedy])
print("decoded:", tok.decode([g["token"] for g in greedy]))
