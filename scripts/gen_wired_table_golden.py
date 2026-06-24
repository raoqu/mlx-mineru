#!/usr/bin/env python3
"""Golden for the C++ wired-table (UNet) recognizer. Draws a bordered table, runs MinerU's
ACTUAL TSRUnet (preprocess -> unet.onnx -> seg postprocess) and the full
WiredTableRecognition, saving the RGB crop + the preprocessed input tensor + the segmentation
prediction + the final polygons/html so the C++ port can be verified stage by stage.
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNET = os.path.join(HERE, "models", "pipeline", "TabRec", "UnetStructure", "unet.onnx")
GOLDEN = os.path.join(HERE, "tests", "golden")
sys.path.insert(0, os.path.expanduser("~/research/MinerU"))
from mineru.model.table.rec.unet_table.table_structure_unet import TSRUnet  # noqa: E402
from mineru.model.table.rec.unet_table.main import WiredTableRecognition, WiredTableInput  # noqa: E402


def draw_wired_table(path):
    h, w = 240, 480
    img = np.full((h, w, 3), 255, np.uint8)
    rows = [20, 70, 120, 170, 220]
    cols = [20, 140, 260, 380, 460]
    for y in rows:
        cv2.line(img, (cols[0], y), (cols[-1], y), (0, 0, 0), 2)
    for x in cols:
        cv2.line(img, (x, rows[0]), (x, rows[-1]), (0, 0, 0), 2)
    cells = [["Name", "Age", "City", "ID"], ["Alice", "30", "NYC", "1"],
             ["Bob", "25", "LA", "2"], ["Carol", "41", "SF", "3"]]
    for r in range(4):
        for c in range(4):
            cv2.putText(img, cells[r][c], (cols[c] + 8, rows[r] + 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return img


def main():
    p = "/tmp/wired_table.png"
    bgr = draw_wired_table(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    np.ascontiguousarray(rgb).tofile(os.path.join(GOLDEN, "wired_table_input.rgb"))

    ts = TSRUnet({"model_path": UNET, "device": "cpu", "intra_op_num_threads": 4,
                  "inter_op_num_threads": 4})
    # capture preprocess + raw seg
    data = ts.preprocess(bgr.copy())
    inp = data["img"]  # [1,3,1024,1024]
    # (preprocessed tensor not persisted; C++ recomputes preprocess)
    pred = ts.infer(data)  # [1024,1024] uint8 seg map (0/1/2)
    np.ascontiguousarray(pred.astype(np.uint8)).tofile(os.path.join(GOLDEN, "wired_table_seg.u8"))

    # full wired recognition (need_ocr=False -> polygons only, no OCR dependency)
    wired = WiredTableRecognition(WiredTableInput(model_path=UNET))
    out = wired(bgr.copy(), ocr_result=None, need_ocr=False)
    polys = out.cell_bboxes  # (N, 8)
    logi = out.logic_points  # (N, 4) row_start,row_end,col_start,col_end

    # Stage 4 golden: structure HTML (empty cells) via plot_html_table. need_ocr=False stores
    # cell_bboxes as box_4_1 [xmin,ymin,xmax,ymax]; convert back to box_4_2 (N,4,2).
    from mineru.model.table.rec.unet_table.utils_table_recover import (
        plot_html_table, box_4_1_poly_to_box_4_2)
    polys42 = np.array([box_4_1_poly_to_box_4_2(b) for b in polys]) if polys is not None else None
    struct_html = plot_html_table(logi, {}, polys42) if polys is not None else ""

    # Stage 5 golden: OCR the table (onnx det+rec) + full WiredTableRecognition -> pred_html.
    import math
    OCR = os.path.join(HERE, "models", "pipeline", "OCR")
    from mineru.model.utils.pytorchocr.data import create_operators, transform
    from mineru.model.utils.pytorchocr.postprocess.db_postprocess import DBPostProcess
    from mineru.utils.ocr_utils import sorted_boxes as ocr_sort, merge_det_boxes, get_rotate_crop_image_for_text_rec
    import onnxruntime as ort2
    H_, WB, MINW, MAXW, BATCH = 48, 320, 16, 2560, 6
    pre = create_operators([
        {"DetResizeForTest": {"limit_side_len": 960, "limit_type": "max", "max_side_limit": 4000}},
        {"NormalizeImage": {"std": [0.229, 0.224, 0.225], "mean": [0.485, 0.456, 0.406], "scale": "1./255.", "order": "hwc"}},
        {"ToCHWImage": None}, {"KeepKeys": {"keep_keys": ["image", "shape"]}}])
    post = DBPostProcess(thresh=0.3, box_thresh=0.6, max_candidates=1000, unclip_ratio=1.5, use_dilation=False, score_mode="fast", box_type="quad")
    chars = ["blank"] + [l.decode("utf-8").rstrip("\r\n") for l in open(os.path.join(OCR, "ppocrv6_dict.txt"), "rb")] + [" "]
    d = transform({"image": bgr.copy(), "polys": np.zeros((0, 4, 2), np.float32)}, pre)
    ds = ort2.InferenceSession(os.path.join(OCR, "ocr_det.onnx"), providers=["CPUExecutionProvider"])
    dpred = ds.run(None, {ds.get_inputs()[0].name: d[0][None].astype(np.float32)})[0]
    boxes = [np.array(b).reshape(-1, 2).astype(np.float32) for b in post({"maps": dpred}, np.array([d[1]]))[0]["points"]]
    boxes = merge_det_boxes(ocr_sort(boxes))
    crops = [get_rotate_crop_image_for_text_rec(bgr, np.array(b, np.float32).copy()) for b in boxes]
    rs = ort2.InferenceSession(os.path.join(OCR, "ocr_rec.onnx"), providers=["CPUExecutionProvider"])
    widths = [c.shape[1] / float(c.shape[0]) for c in crops]
    order = np.argsort(np.array(widths))
    rec = [("", 0.0)] * len(crops)
    for beg in range(0, len(crops), BATCH):
        end = min(len(crops), beg + BATCH)
        mw = widths[order[end - 1]]
        for ino in range(beg, end):
            c = crops[order[ino]]; ch, cw = c.shape[:2]
            imgW = max(min(int(H_ * max(mw, WB / H_)), MAXW), MINW)
            rw = min(imgW, int(max(math.ceil(H_ * cw / ch), MINW)))
            rimg = cv2.resize(c, (rw, H_)) / 127.5 - 1.0
            inp2 = np.zeros((1, 3, H_, imgW), np.float32); inp2[0, :, :, :rw] = rimg.transpose(2, 0, 1)
            lg = rs.run(None, {rs.get_inputs()[0].name: inp2})[0][0]; idx = lg.argmax(1); t = ""; prev = -1
            for k in range(len(idx)):
                b = int(idx[k])
                if b != 0 and b != prev: t += chars[b]
                prev = b
            rec[order[ino]] = (t, 1.0)
    ocr_result = [[b.tolist(), t, s] for b, (t, s) in zip(boxes, rec)]
    wired_full = WiredTableRecognition(WiredTableInput(model_path=UNET))
    out2 = wired_full(bgr.copy(), ocr_result=ocr_result, need_ocr=True)
    pred_html = out2.pred_html

    meta = {"input": "wired_table_input.rgb", "w": int(w), "h": int(h),
            "inp_shape": list(inp.shape), "seg_shape": list(pred.shape),
            "n_cells": int(len(polys)) if polys is not None else 0,
            "structure_html": struct_html, "pred_html": pred_html,
            "ocr": [{"box": b, "text": t, "score": s} for b, t, s in ocr_result]}
    json.dump(meta, open(os.path.join(GOLDEN, "wired_table.json"), "w"), indent=1)
    if polys is not None:
        np.array(polys).astype(np.float32).tofile(os.path.join(GOLDEN, "wired_table_polys.f32"))
        np.array(logi).astype(np.int32).tofile(os.path.join(GOLDEN, "wired_table_logi.i32"))
    print(f"wired golden: img {w}x{h}, seg {pred.shape}, seg vals {sorted(set(pred.flatten().tolist()))}, "
          f"{meta['n_cells']} cells")


if __name__ == "__main__":
    main()
