#!/usr/bin/env python3
"""Golden for the C++ cross_page_table_merge. Runs MinerU's own
mineru.utils.table_merge.merge_table() on synthetic two/three-page pdf_info inputs
and records {name, input, expected} so the C++ port can be checked for deep JSON
equality against the reference implementation.

Requires MinerU importable, e.g.:
    PYTHONPATH=/path/to/MinerU python3 scripts/gen_table_merge_golden.py

Cases exercise every decision branch: repeated/absent header, continuation vs plain
caption, footnote on current/previous page, column & width mismatch, rowspan header,
and a three-page chain merge.
"""
import copy
import json
import os

from mineru.utils.table_merge import merge_table
from mineru.utils.enum_class import BlockType

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "tests", "golden", "table_merge_golden.json")


def body(html, bbox, index):
    return {"type": BlockType.TABLE_BODY, "bbox": bbox, "index": index,
            "lines": [{"spans": [{"type": "table", "html": html}]}]}


def cap(text, bbox, index):
    return {"type": BlockType.TABLE_CAPTION, "bbox": bbox, "index": index,
            "lines": [{"spans": [{"type": "text", "content": text}]}]}


def foot(text, bbox, index):
    return {"type": BlockType.TABLE_FOOTNOTE, "bbox": bbox, "index": index,
            "lines": [{"spans": [{"type": "text", "content": text}]}]}


def table(blocks, bbox, index):
    return {"type": BlockType.TABLE, "bbox": bbox, "index": index, "blocks": blocks}


def page(*blocks):
    return {"para_blocks": list(blocks)}


cases = []


def add(name, pages):
    out = copy.deepcopy(pages)
    merge_table(out)
    cases.append({"name": name, "input": copy.deepcopy(pages), "expected": out})


TB = [10, 500, 300, 560]
TT = [10, 40, 300, 100]
add("A_header_repeat", [
    page(table([body("<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>1</td><td>2</td><td>3</td></tr></table>", TB, 5)], TB, 5)),
    page(table([body("<table><tr><td>A</td><td>B</td><td>C</td></tr><tr><td>4</td><td>5</td><td>6</td></tr></table>", TT, 8)], TT, 8))])
add("B_no_header", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([body("<table><tr><td>3</td><td>4</td></tr></table>", TT, 8)], TT, 8))])
add("C_caption_plain", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([cap("Table 2: New", [10, 20, 300, 38], 7), body("<table><tr><td>3</td><td>4</td></tr></table>", TT, 8)], TT, 8))])
add("D_caption_continued", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([cap("Table 1 (续)", [10, 20, 300, 38], 7), body("<table><tr><td>3</td><td>4</td></tr></table>", TT, 8)], TT, 8))])
add("E_footnote_page2", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([body("<table><tr><td>3</td><td>4</td></tr></table>", TT, 8), foot("note", [10, 110, 300, 128], 9)], TT, 8))])
add("F_col_mismatch", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([body("<table><tr><td>x</td><td>y</td><td>z</td></tr></table>", TT, 8)], TT, 8))])
add("G_footnote_prev", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5), foot("pn", [10, 565, 300, 580], 6)], TB, 5)),
    page(table([body("<table><tr><td>3</td><td>4</td></tr></table>", TT, 8)], TT, 8))])
add("H_width_mismatch", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([body("<table><tr><td>3</td><td>4</td></tr></table>", [10, 40, 150, 100], 8)], [10, 40, 150, 100], 8))])
add("I_rowspan_header", [
    page(table([body('<table><tr><td rowspan="2">H</td><td>B</td></tr><tr><td>b</td></tr><tr><td>1</td><td>2</td></tr></table>', TB, 5)], TB, 5)),
    page(table([body('<table><tr><td rowspan="2">H</td><td>B</td></tr><tr><td>b</td></tr><tr><td>9</td><td>8</td></tr></table>', TT, 8)], TT, 8))])
add("J_three_page", [
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>", TB, 5)], TB, 5)),
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>3</td><td>4</td></tr></table>", TT, 8)], TT, 8)),
    page(table([body("<table><tr><td>A</td><td>B</td></tr><tr><td>5</td><td>6</td></tr></table>", TT, 8)], TT, 8))])

json.dump(cases, open(OUT, "w"), ensure_ascii=False, indent=1)
print(f"wrote {len(cases)} cases -> {OUT}")
