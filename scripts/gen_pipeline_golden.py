#!/usr/bin/env python3
"""Golden for the C++ pipeline assembly (P5): model_list -> middle_json.

Runs MinerU's REAL pipeline backend on a.pdf and dumps, per page, the raw model
output (model_list: layout_dets + page_info) and the assembled middle_json
(pdf_info: preproc/para_blocks). The C++ assembly port (MagicModel +
model_json_to_middle_json + para_split) must transform model_list -> middle_json;
union_make (already byte-exact) then renders Markdown.

Prerequisites (one-time local MinerU setup, since models aren't bundled):
  1. pip install ftfy pypdf reportlab pdftext magika json_repair openai
  2. ~/mineru.json:
       {"models-dir": {"pipeline": "<PDF-Extract-Kit-1.0 dir>",
                       "vlm": "<PDF-Extract-Kit-1.0 dir>"},
        "model-source": "local"}
     where <...> contains models/Layout, models/OCR, models/TabCls, models/TabRec,
     models/MFR/unimernet_hf_small_2503 (symlink the project's models/pipeline/MFR).
  3. The PDF-Extract-Kit Layout/PP-DocLayoutV2 dir must contain config.json +
     preprocessor_config.json (copy from models/pipeline/Layout/) -- the raw HF
     download ships only model.safetensors, and without the config MinerU loads a
     2-class head and the weights fail to load.

Run from the MinerU checkout:
  MINERU_MODEL_SOURCE=local PYTHONPATH=~/research/MinerU \
    python3 ~/mylab/mlx-mineru/scripts/gen_pipeline_golden.py
"""
import json
import os

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "golden")
PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "a.pdf")


def main():
    os.environ.setdefault("MINERU_MODEL_SOURCE", "local")
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze_streaming as analyze

    data = open(PDF, "rb").read()
    os.makedirs("/tmp/mineru_imgs", exist_ok=True)
    iw = FileBasedDataWriter("/tmp/mineru_imgs")
    cap = {}

    def on_ready(doc_index, model_list, middle_json, ocr_enable):
        cap["ml"] = model_list
        cap["mj"] = middle_json

    analyze([data], [iw], ["ch"], on_ready, parse_method="auto",
            formula_enable=True, table_enable=True)

    ml, mj = cap["ml"], cap["mj"]
    json.dump(ml, open(os.path.join(OUT, "pipeline_model.json"), "w"), ensure_ascii=False)
    json.dump(mj, open(os.path.join(OUT, "pipeline_middle.json"), "w"), ensure_ascii=False)
    # compact single-page golden for the first assembly slice
    json.dump(ml[0], open(os.path.join(OUT, "pipeline_p0_model.json"), "w"), ensure_ascii=False)
    json.dump(mj["pdf_info"][0], open(os.path.join(OUT, "pipeline_p0_middle.json"), "w"),
              ensure_ascii=False)
    p0 = mj["pdf_info"][0]
    print(f"wrote pipeline goldens: {len(ml)} pages; p0 {len(ml[0]['layout_dets'])} dets -> "
          f"{len(p0['para_blocks'])} para_blocks; page_size {p0['page_size']}, "
          f"model width {ml[0]['page_info']['width']}")


if __name__ == "__main__":
    main()
