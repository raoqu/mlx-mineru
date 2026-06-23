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

# torch.take_along_dim isn't in ONNX opset 17, but it's equivalent to gather here
# (indices are pre-expanded to match), and gather IS supported.
_orig_take = torch.take_along_dim
def _take_along_dim(x, idx, dim):
    if idx.shape == x.shape:
        return torch.gather(x, dim, idx)
    # broadcast idx to x's shape on the non-dim axes (matches take_along_dim semantics)
    shp = list(x.shape)
    shp[dim] = idx.shape[dim]
    return torch.gather(x, dim, idx.expand(shp))
torch.take_along_dim = _take_along_dim

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
        return o.logits, o.pred_boxes, o.order_logits


wrap = Wrap(model)
dummy = torch.zeros(1, 3, 800, 800)
with torch.no_grad():
    ref = wrap(dummy)
names = ["logits", "pred_boxes", "order_logits"]
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

# Parity on a REAL page image (the model's actual input distribution). NOTE:
# raw-tensor max|diff| on the 300 queries is large only on low-confidence padding
# rows (the in-graph argsort tie-breaks them differently torch-vs-ORT); what
# matters is that the *thresholded* detections match exactly.
import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
thr = np.array(cfg.class_thresholds, dtype=np.float32)


def thresholded(logits, boxes):
    p = 1.0 / (1.0 + np.exp(-logits[0]))
    cid, sc = p.argmax(1), p.max(1)
    keep = sc >= thr[cid]
    return sorted(
        [(int(c), round(float(s), 3), tuple(np.round(b, 3)))
         for c, s, b, k in zip(cid, sc, boxes[0], keep) if k],
        key=lambda r: -r[1])


img_path = os.path.join(HERE, "a.pdf")
if os.path.exists(img_path):
    import pypdfium2 as pdfium
    from PIL import Image

    page = pdfium.PdfDocument(img_path)[0]
    img = page.render(scale=2).to_pil().convert("RGB").resize((800, 800), Image.BICUBIC)
    x = (np.asarray(img).astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    with torch.no_grad():
        tl, tb, _ = wrap(torch.from_numpy(x))
    ol, ob, oo = sess.run(names, {"pixel_values": x})
    td, od = thresholded(tl.numpy(), tb.numpy()), thresholded(ol, ob)
    print("=== parity on real page (thresholded detections) ===")
    print(f"  torch={len(td)} onnx={len(od)} detections; order_logits max|diff|="
          f"{float(np.max(np.abs(wrap(torch.from_numpy(x))[2].detach().numpy() - oo))):.2e}")
    print("  detections match:", td == od)
else:
    print("(skip real-image parity: no a.pdf)")
