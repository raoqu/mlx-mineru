#!/usr/bin/env python3
"""Golden for the C++ SLANet table structurer. Renders a table, runs MinerU's ACTUAL
TableStructurer (resize-488 + BGR normalize + pad + slanet-plus.onnx + TableLabelDecode)
+ adapt_slanet_plus. Saves the table RGB crop + expected structure tokens + adapted cell
bboxes. (The OCR->HTML matcher is verified by a separate golden.)
"""
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(HERE, "models", "pipeline", "TabRec", "SlanetPlus", "slanet-plus.onnx")
GOLDEN = os.path.join(HERE, "tests", "golden")
sys.path.insert(0, os.path.expanduser("~/research/MinerU"))
from mineru.model.table.rec.slanet_plus.table_structure import TableStructurer  # noqa: E402
from mineru.model.table.rec.slanet_plus.main import PaddleTable, PaddleTableInput  # noqa: E402


def render_table(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.axis("off")
    data = [["Name", "Age", "City"], ["Alice", "30", "NYC"], ["Bob", "25", "LA"]]
    tb = ax.table(cellText=data, loc="center", cellLoc="center")
    tb.scale(1, 2)
    tb.set_fontsize(14)
    fig.savefig(path, dpi=120, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    p = "/tmp/table.png"
    if not os.path.exists(p):
        render_table(p)
    bgr = cv2.imread(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    np.ascontiguousarray(rgb).tofile(os.path.join(GOLDEN, "table_input.rgb"))

    cfg = {"model_path": TAB, "device": "cpu", "intra_op_num_threads": 4,
           "inter_op_num_threads": 4}
    ts = TableStructurer(cfg)
    struct, cell_bboxes, _ = ts.process(bgr.copy())  # MinerU OCRs from BGR

    # adapt_slanet_plus (scale cells from original-image space to padded-488 space)
    pt = PaddleTable(PaddleTableInput(model_path=TAB))
    cells = pt.adapt_slanet_plus(bgr, np.array(cell_bboxes, dtype=np.float64))

    out = {"input": "table_input.rgb", "src_w": int(w), "src_h": int(h),
           "vocab_size": len(ts.character),
           "structure": struct,
           "cells": [[round(float(v), 3) for v in c] for c in cells]}
    json.dump(out, open(os.path.join(GOLDEN, "table_struct.json"), "w"), indent=1)
    print(f"wrote table_struct.json: {len(struct)} tokens, {len(cells)} cells, "
          f"vocab {len(ts.character)}")
    print("struct:", struct)


if __name__ == "__main__":
    main()
