#!/usr/bin/env python3
"""Golden for the C++ TableMatch. OCRs the table (onnx det+rec, same chain as
gen_ocr_page_golden) to get ocr_result, then runs MinerU's ACTUAL TableMatch on the
real structure tokens + adapted cell bboxes -> pred_html. Saves the ocr items + html;
the C++ test feeds the SAME ocr_result through its own structurer+matcher.
"""
import json
import math
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OCR = os.path.join(HERE, "models", "pipeline", "OCR")
TABROOT = os.path.join(HERE, "models", "pipeline", "TabRec", "SlanetPlus")
GOLDEN = os.path.join(HERE, "tests", "golden")
sys.path.insert(0, os.path.expanduser("~/research/MinerU"))
from mineru.utils.ocr_utils import sorted_boxes, merge_det_boxes, get_rotate_crop_image_for_text_rec  # noqa: E402
from mineru.model.utils.pytorchocr.data import create_operators, transform  # noqa: E402
from mineru.model.utils.pytorchocr.postprocess.db_postprocess import DBPostProcess  # noqa: E402
from mineru.model.table.rec.slanet_plus.table_structure import TableStructurer  # noqa: E402
from mineru.model.table.rec.slanet_plus.main import PaddleTable, PaddleTableInput  # noqa: E402
from mineru.model.table.rec.slanet_plus.matcher import TableMatch  # noqa: E402

H, WB, MINW, MAXW, BATCH = 48, 320, 16, 2560, 6


def ocr_table(bgr):
    pre = create_operators([
        {"DetResizeForTest": {"limit_side_len": 960, "limit_type": "max", "max_side_limit": 4000}},
        {"NormalizeImage": {"std": [0.229, 0.224, 0.225], "mean": [0.485, 0.456, 0.406],
                            "scale": "1./255.", "order": "hwc"}},
        {"ToCHWImage": None}, {"KeepKeys": {"keep_keys": ["image", "shape"]}}])
    post = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5,
                         use_dilation=False, score_mode="fast", box_type="quad")
    chars = ["blank"] + [l.decode("utf-8").rstrip("\r\n") for l in open(os.path.join(OCR, "ppocrv6_dict.txt"), "rb")] + [" "]
    d = transform({"image": bgr.copy(), "polys": np.zeros((0, 4, 2), np.float32)}, pre)
    ds = ort.InferenceSession(os.path.join(OCR, "ocr_det.onnx"), providers=["CPUExecutionProvider"])
    pred = ds.run(None, {ds.get_inputs()[0].name: d[0][None].astype(np.float32)})[0]
    boxes = [np.array(b).reshape(-1, 2).astype(np.float32) for b in post({"maps": pred}, np.array([d[1]]))[0]["points"]]
    boxes = merge_det_boxes(sorted_boxes(boxes))
    crops = [get_rotate_crop_image_for_text_rec(bgr, np.array(b, np.float32).copy()) for b in boxes]
    rs = ort.InferenceSession(os.path.join(OCR, "ocr_rec.onnx"), providers=["CPUExecutionProvider"])
    widths = [c.shape[1] / float(c.shape[0]) for c in crops]
    order = np.argsort(np.array(widths))
    result = [("", 0.0)] * len(crops)
    for beg in range(0, len(crops), BATCH):
        end = min(len(crops), beg + BATCH)
        mw = widths[order[end - 1]]
        for ino in range(beg, end):
            c = crops[order[ino]]; ch, cw = c.shape[:2]
            imgW = max(min(int(H * max(mw, WB / H)), MAXW), MINW)
            rw = min(imgW, int(max(math.ceil(H * cw / ch), MINW)))
            r = cv2.resize(c, (rw, H)) / 127.5 - 1.0
            inp = np.zeros((1, 3, H, imgW), np.float32); inp[0, :, :, :rw] = r.transpose(2, 0, 1)
            lg = rs.run(None, {rs.get_inputs()[0].name: inp})[0][0]
            idx = lg.argmax(1); t = ""; score = 0.0; kept = 0; prev = -1
            for k in range(len(idx)):
                b = int(idx[k])
                if b != 0 and b != prev:
                    row = lg[k]; score += float(1.0 / np.sum(np.exp(row - row[b]))); t += chars[b]; kept += 1
                prev = b
            result[order[ino]] = (t, score / kept if kept else 0.0)
    return [[b.tolist(), t, s] for b, (t, s) in zip(boxes, result)]


def main():
    bgr = cv2.imread("/tmp/table.png")
    h, w = bgr.shape[:2]
    ocr_result = ocr_table(bgr)

    onnx = os.path.join(TABROOT, "slanet-plus.onnx")
    ts = TableStructurer({"model_path": onnx, "device": "cpu", "intra_op_num_threads": 4, "inter_op_num_threads": 4})
    structure, cells, _ = ts.process(bgr.copy())
    pt = PaddleTable(PaddleTableInput(model_path=onnx))
    cells = pt.adapt_slanet_plus(bgr, np.array(cells, np.float64))
    dt_boxes, rec_res = pt.get_boxes_recs(ocr_result, h, w)
    pred_html = TableMatch()(structure, cells, dt_boxes, rec_res)

    items = [{"box": b, "text": t, "score": round(float(s), 5)} for b, t, s in ocr_result]
    out = {"input": "table_input.rgb", "src_w": int(w), "src_h": int(h),
           "ocr": items, "pred_html": pred_html}
    json.dump(out, open(os.path.join(GOLDEN, "table_match.json"), "w"), ensure_ascii=False, indent=1)
    print(f"wrote table_match.json: {len(items)} ocr items")
    print("pred_html:", pred_html)


if __name__ == "__main__":
    main()
