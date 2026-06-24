#!/usr/bin/env python3
"""Golden for the C++ OCR det DB-postprocess. Runs MinerU's ACTUAL det preprocess
(DetResizeForTest max-960 /32 + normalize) + ocr_det.onnx, then MinerU's ACTUAL
DBPostProcess (cv2.findContours + cv2.minAreaRect + pyclipper unclip). Saves the
prob-map + shape so the C++ verifies its DB-postprocess against MinerU on the SAME
prob-map (isolating the geometric port from the resize).
"""
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import pypdfium2 as pdfium

MINERU = os.path.expanduser("~/research/MinerU")
sys.path.insert(0, MINERU)
from mineru.model.utils.pytorchocr.data import create_operators, transform  # noqa: E402
from mineru.model.utils.pytorchocr.postprocess.db_postprocess import DBPostProcess  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR = os.path.join(HERE, "models", "pipeline", "OCR")
GOLDEN = os.path.join(HERE, "tests", "golden")

pre_ops = create_operators([
    {"DetResizeForTest": {"limit_side_len": 960, "limit_type": "max", "max_side_limit": 4000}},
    {"NormalizeImage": {"std": [0.229, 0.224, 0.225], "mean": [0.485, 0.456, 0.406],
                        "scale": "1./255.", "order": "hwc"}},
    {"ToCHWImage": None},
    {"KeepKeys": {"keep_keys": ["image", "shape"]}},
])
post = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5,
                     use_dilation=False, score_mode="fast", box_type="quad")

img = np.asarray(pdfium.PdfDocument(os.path.join(HERE, "a.pdf"))[0].render(scale=2).to_pil().convert("RGB"))
src_h, src_w = img.shape[:2]
np.ascontiguousarray(img).tofile(os.path.join(GOLDEN, "ocr_det_input.rgb"))  # RGB; C++ swaps to BGR
# MinerU's pipeline feeds the model BGR (cv2.cvtColor(RGB, COLOR_RGB2BGR)); match it.
bgr = img[:, :, ::-1]
data = transform({"image": bgr.copy(), "polys": np.zeros((0, 4, 2), np.float32)}, pre_ops)
x, shape_list = data[0][None], np.array([data[1]])

sess = ort.InferenceSession(os.path.join(OCR, "ocr_det.onnx"), providers=["CPUExecutionProvider"])
pred = sess.run(None, {sess.get_inputs()[0].name: x.astype(np.float32)})[0]  # (1,1,Hd,Wd)
Hd, Wd = pred.shape[2], pred.shape[3]
pred[0, 0].astype(np.float32).tofile(os.path.join(GOLDEN, "ocr_det_probmap.f32"))

boxes = post({"maps": pred}, shape_list)[0]["points"]
boxes = [np.array(b).reshape(-1, 2).astype(int).tolist() for b in boxes]
out = {"probmap": "ocr_det_probmap.f32", "Hd": Hd, "Wd": Wd,
       "input": "ocr_det_input.rgb", "src_w": int(src_w), "src_h": int(src_h),
       "shape_list": [float(v) for v in shape_list[0]],
       "thresh": 0.3, "box_thresh": 0.6, "unclip_ratio": 1.5,
       "boxes": boxes}
json.dump(out, open(os.path.join(GOLDEN, "ocr_det.json"), "w"), indent=1)
print(f"wrote ocr_det.json: prob {Hd}x{Wd}, {len(boxes)} boxes; src "
      f"{shape_list[0][0]:.0f}x{shape_list[0][1]:.0f}")
for b in boxes[:5]:
    print("  ", b)
