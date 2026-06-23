#!/usr/bin/env bash
# Fetch the pipeline-backend model files into models/pipeline/ (gitignored).
# Uses huggingface_hub (resumable/retried) — the large files drop on plain curl.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="opendatalab/PDF-Extract-Kit-1.0"

python3 - "$REPO" <<'PY'
import os, shutil, sys
from huggingface_hub import hf_hub_download
repo = sys.argv[1]
dest_root = os.path.join("models", "pipeline")
# (subpath_in_repo, local_dir) — extend as phases land.
FILES = {
    "models/Layout/PP-DocLayoutV2/config.json": "Layout",
    "models/Layout/PP-DocLayoutV2/preprocessor_config.json": "Layout",
    "models/Layout/PP-DocLayoutV2/model.safetensors": "Layout",
    "models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx": "TabCls",
    "models/TabRec/SlanetPlus/slanet-plus.onnx": "TabRec",
    "models/TabRec/UnetStructure/unet.onnx": "TabRec",
}
for sub, sub_dir in FILES.items():
    out_dir = os.path.join(dest_root, sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, os.path.basename(sub))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print("have", dst); continue
    print("fetching", sub)
    p = hf_hub_download(repo, sub, resume_download=True)
    shutil.copy(p, dst)
    print("  ->", dst, os.path.getsize(dst), "bytes")
print("pipeline models ready in", dest_root)
PY
