#!/usr/bin/env python3
"""Export PP-DocLayoutV2 (RT-DETR + reading-order) to ONNX for the C++ pipeline
backend. Dev-time only (needs torch + MinerU's modeling code). The C++ runtime
runs the resulting layout.onnx via onnxruntime — no Python/torch at runtime.

Strategy decision (recorded): pipeline torch models are exported to ONNX once
here; preprocessing and post-processing (thresholds, box scaling, reading order)
are ported to C++.
"""
import os
import sys

import numpy as np
import torch

MINERU = os.path.expanduser("~/research/MinerU")
sys.path.insert(0, MINERU)
from mineru.model.layout.pp_doclayoutv2 import (  # noqa: E402
    PPDocLayoutV2Config,
    PPDocLayoutV2ForObjectDetection,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "models", "pipeline", "Layout")
OUT = os.path.join(MODEL_DIR, "layout.onnx")

cfg = PPDocLayoutV2Config.from_pretrained(MODEL_DIR)
model = PPDocLayoutV2ForObjectDetection.from_pretrained(MODEL_DIR, config=cfg)
model.eval()


class Wrap(torch.nn.Module):
    """Return a fixed tuple of tensors so ONNX export has stable outputs."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        o = self.m(pixel_values=pixel_values)
        outs = []
        for name in ("logits", "pred_boxes", "order_logits"):
            v = getattr(o, name, None)
            if v is not None:
                outs.append(v)
        return tuple(outs)


wrap = Wrap(model)
dummy = torch.zeros(1, 3, 800, 800)
with torch.no_grad():
    ref = wrap(dummy)
names = ["logits", "pred_boxes", "order_logits"][: len(ref)]
print("model outputs:", [(n, tuple(t.shape)) for n, t in zip(names, ref)])

torch.onnx.export(
    wrap,
    (dummy,),
    OUT,
    input_names=["pixel_values"],
    output_names=names,
    dynamic_axes={"pixel_values": {0: "batch"}},
    opset_version=17,
    do_constant_folding=True,
)
print("exported", OUT, "(%.1f MB)" % (os.path.getsize(OUT) / 1e6))

# Parity: torch vs onnxruntime on a random input.
import onnxruntime as ort  # noqa: E402

rng = np.random.default_rng(0)
x = rng.standard_normal((1, 3, 800, 800)).astype(np.float32)
with torch.no_grad():
    tref = wrap(torch.from_numpy(x))
sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
oref = sess.run(names, {"pixel_values": x})
print("=== torch vs onnxruntime parity ===")
for n, t, o in zip(names, tref, oref):
    d = float(np.max(np.abs(t.numpy() - o)))
    print(f"  {n:12s} shape {tuple(o.shape)} max|diff| = {d:.2e}")
