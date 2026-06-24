#!/usr/bin/env python3
"""Export PP-OCRv6 detection (DBNet) + recognition (CTC) nets to ONNX, building
them via pytorchocr's BaseOCRV20 + arch_config.yaml directly (pure torch — avoids
the cv2/shapely/pyclipper deps that the full pipeline import pulls in). Weights
loaded from the local PDF-Extract-Kit dir; nets reparameterized (RepVGG fuse)
before export, exactly as MinerU does at inference.
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

MINERU = os.path.expanduser("~/research/MinerU")
sys.path.insert(0, MINERU)
from mineru.model.utils.pytorchocr.base_ocr_v20 import BaseOCRV20  # noqa: E402

LOCAL = "/Users/raoqu/models/models/OpenDataLab-PDF-Extract-Kit-1.0/models/OCR/paddleocr_torch"
CFG = os.path.join(MINERU, "mineru/model/utils/pytorchocr/utils/resources/arch_config.yaml")
ALL_CFG = yaml.safe_load(open(CFG, encoding="utf-8"))
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "models", "pipeline", "OCR")
os.makedirs(OUT_DIR, exist_ok=True)


def build(fname):
    wpath = os.path.join(LOCAL, fname)
    cfg = ALL_CFG[Path(wpath).stem]  # complete arch (rec head has out_channels_list)
    m = BaseOCRV20(cfg)
    m.load_pytorch_weights(wpath)
    for mod in m.net.modules():
        if hasattr(mod, "rep"):
            mod.rep()
    return m.net.eval()


def export(net, dummy, path, dyn):
    torch.onnx.export(net, (dummy,), path, input_names=["x"], output_names=["y"],
                      dynamic_axes=dyn, opset_version=17, do_constant_folding=True)
    import onnxruntime as ort
    with torch.no_grad():
        t = net(dummy)
    if isinstance(t, dict):
        t = list(t.values())[0]
    t = t[0] if isinstance(t, (tuple, list)) else t
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    o = sess.run(None, {"x": dummy.numpy()})[0]
    print(f"  {os.path.basename(path)}: out {tuple(o.shape)} torch-vs-ort "
          f"max|diff|={float(np.max(np.abs(t.numpy() - o))):.2e}")


print("=== detection (DBNet) ===")
det = build("ch_PP-OCRv6_small_det_infer.safetensors")
export(det, torch.zeros(1, 3, 640, 640), os.path.join(OUT_DIR, "ocr_det.onnx"),
       {"x": {0: "b", 2: "h", 3: "w"}, "y": {0: "b", 2: "h", 3: "w"}})

print("=== recognition (CTC) ===")
class RecWrap(torch.nn.Module):  # MultiHead eval returns {"ctc_logits":.., bool}; keep logits
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        o = self.net(x)
        return o["ctc_logits"] if isinstance(o, dict) else o

rec = RecWrap(build("ch_PP-OCRv6_medium_rec_infer.safetensors")).eval()
export(rec, torch.zeros(1, 3, 48, 320), os.path.join(OUT_DIR, "ocr_rec.onnx"),
       {"x": {0: "b", 3: "w"}, "y": {0: "b", 1: "t"}})

import shutil  # noqa: E402
shutil.copy(os.path.join(MINERU, "mineru/model/utils/pytorchocr/utils/resources/dict/ppocrv6_dict.txt"),
            os.path.join(OUT_DIR, "ppocrv6_dict.txt"))
print("copied ppocrv6_dict.txt")
