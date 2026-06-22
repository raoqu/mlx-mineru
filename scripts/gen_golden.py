#!/usr/bin/env python3
"""Generate golden mkcontent outputs by running the REAL MinerU union_make.

Dev-time tool only (needs the MinerU source tree + loguru + fast-langdetect);
NOT part of the C++ runtime. It freezes the output contract into tests/golden/
so the C++ renderer can be diffed against it.

Usage: python3 scripts/gen_golden.py [MINERU_SRC]
"""
import json
import os
import sys

MINERU_SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/research/MinerU")
sys.path.insert(0, MINERU_SRC)

# Deterministic config: no ~/mineru.json influence, formula+table on.
os.environ.setdefault("MINERU_VLM_FORMULA_ENABLE", "True")
os.environ.setdefault("MINERU_VLM_TABLE_ENABLE", "True")

from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make  # noqa: E402
from mineru.utils.enum_class import MakeMode  # noqa: E402
from mineru.utils.language import detect_lang  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLDEN = os.path.join(HERE, "tests", "golden")
os.makedirs(GOLDEN, exist_ok=True)

fixture = os.path.join(HERE, "tests", "data", "sample_middle.json")
mj = json.load(open(fixture, encoding="utf-8"))

# Report detected languages so we can confirm fixtures are linguistically
# unambiguous (C++ uses a CJK-script heuristic that must agree here).
print("detected langs per text block:")
for page in mj["pdf_info"]:
    for b in page.get("para_blocks", []):
        txt = ""
        for line in b.get("lines", []) or []:
            for sp in line.get("spans", []):
                if sp.get("type") == "text":
                    txt += sp.get("content", "")
        if txt:
            print(f"  {b['type']:20s} -> {detect_lang(txt)!r}  ({txt[:24]})")

img = "images"
# Re-load per mode because union_make mutates spans in place (full_to_half).
def fresh():
    return json.load(open(fixture, encoding="utf-8"))["pdf_info"]

mm_md = union_make(fresh(), MakeMode.MM_MD, img)
nlp_md = union_make(fresh(), MakeMode.NLP_MD, img)
content_list = union_make(fresh(), MakeMode.CONTENT_LIST, img)
content_list_v2 = union_make(fresh(), MakeMode.CONTENT_LIST_V2, img)

with open(os.path.join(GOLDEN, "sample.mm.md"), "w", encoding="utf-8") as f:
    f.write(mm_md)
with open(os.path.join(GOLDEN, "sample.nlp.md"), "w", encoding="utf-8") as f:
    f.write(nlp_md)
with open(os.path.join(GOLDEN, "sample.content_list.json"), "w", encoding="utf-8") as f:
    json.dump(content_list, f, ensure_ascii=False, indent=4)
with open(os.path.join(GOLDEN, "sample.content_list_v2.json"), "w", encoding="utf-8") as f:
    json.dump(content_list_v2, f, ensure_ascii=False, indent=4)

print("\nwrote golden to", GOLDEN)
