#!/usr/bin/env python3
"""Golden for the layout-output parser. Replicates mineru-vl-utils
parse_layout_output exactly (regex + _convert_bbox + table-internal filter) from
source; BLOCK_TYPES imported from the dependency-free structs.py."""
import json
import os
import re
import sys

REF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "third_party", "reference", "mineru-vl-utils")
sys.path.insert(0, REF)
from mineru_vl_utils.structs import BLOCK_TYPES  # noqa: E402

_layout_re = (
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
    r"<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>"
    r"(?:(<\|rotate_(?:up|right|down|left)\|>))?"
    r"(.*?)(?=<\|box_start\|>|$)"
)
ANGLE = {"<|rotate_up|>": 0, "<|rotate_right|>": 90, "<|rotate_down|>": 180, "<|rotate_left|>": 270}


def convert_bbox(b):
    b = tuple(map(int, b))
    if any(c < 0 or c > 1000 for c in b):
        return None
    x1, y1, x2, y2 = b
    x1, x2 = (x2, x1) if x2 < x1 else (x1, x2)
    y1, y2 = (y2, y1) if y2 < y1 else (y1, y2)
    if x1 == x2 or y1 == y2:
        return None
    return [n / 1000.0 for n in (x1, y1, x2, y2)]


def parse_angle(tail):
    for tok, a in ANGLE.items():
        if tok in tail:
            return a
    return None


def cover_ratio(inner, outer):
    ix0, iy0 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix1, iy1 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    ia = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return (iw * ih) / ia if ia > 0 else 0.0


def parse(output):
    blocks = []
    for m in re.finditer(_layout_re, output, re.DOTALL):
        x1, y1, x2, y2, rt, rot, tail = m.groups()
        bb = convert_bbox((x1, y1, x2, y2))
        if bb is None:
            continue
        rt = rt.lower()
        if rt == "unknown":
            rt = "image"
        if rt == "inline_formula":
            continue
        if rt not in BLOCK_TYPES:
            continue
        angle = parse_angle(rot) if rot else None
        mp = ("txt_contd_tgt" in tail) if rt == "text" else False
        blocks.append({"type": rt, "bbox": bb, "angle": angle, "merge_prev": mp})
    # table-internal filter
    tables = [i for i, b in enumerate(blocks) if b["type"] == "table"]
    drop = set()
    for i, b in enumerate(blocks):
        if b["type"] not in {"text", "equation", "equation_block"}:
            continue
        for ci in tables:
            if ci != i and cover_ratio(b["bbox"], blocks[ci]["bbox"]) >= 0.9:
                drop.add(i)
                break
    return [b for i, b in enumerate(blocks) if i not in drop]


SAMPLES = [
    "<|box_start|>100 120 900 180<|box_end|><|ref_start|>Title<|ref_end|><|rotate_up|>",
    ("<|box_start|>50 200 950 400<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>txt_contd_tgt"
     "<|box_start|>60 420 940 800<|box_end|><|ref_start|>table<|ref_end|><|rotate_up|>"
     "<|box_start|>70 430 500 600<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>"),
    "<|box_start|>10 10 5 5<|box_end|><|ref_start|>text<|ref_end|>",
    "<|box_start|>0 0 1200 50<|box_end|><|ref_start|>text<|ref_end|>",
    "<|box_start|>30 40 300 90<|box_end|><|ref_start|>unknown<|ref_end|><|rotate_right|>",
    "<|box_start|>30 40 300 90<|box_end|><|ref_start|>inline_formula<|ref_end|>",
    "<|box_start|>10 20 800 60<|box_end|><|ref_start|>image_caption<|ref_end|><|rotate_down|>caption text here",
]

out = {"samples": [{"output": s, "blocks": parse(s)} for s in SAMPLES]}
GOLDEN = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "golden")
json.dump(out, open(os.path.join(GOLDEN, "layout.json"), "w"), indent=2)
print("wrote layout.json")
for s in out["samples"]:
    print(len(s["blocks"]), [(b["type"], b["angle"], b["merge_prev"]) for b in s["blocks"]])
