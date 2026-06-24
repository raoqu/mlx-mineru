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

    meta = {"input": "wired_table_input.rgb", "w": int(w), "h": int(h),
            "inp_shape": list(inp.shape), "seg_shape": list(pred.shape),
            "n_cells": int(len(polys)) if polys is not None else 0}
    json.dump(meta, open(os.path.join(GOLDEN, "wired_table.json"), "w"), indent=1)
    if polys is not None:
        np.array(polys).astype(np.float32).tofile(os.path.join(GOLDEN, "wired_table_polys.f32"))
        np.array(logi).astype(np.int32).tofile(os.path.join(GOLDEN, "wired_table_logi.i32"))
    print(f"wired golden: img {w}x{h}, seg {pred.shape}, seg vals {sorted(set(pred.flatten().tolist()))}, "
          f"{meta['n_cells']} cells")


if __name__ == "__main__":
    main()
