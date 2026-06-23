#!/usr/bin/env bash
# Fetch the MinerU2.5 VLM model weights (~2.2GB safetensors) into
# models/MinerU2.5-tokenizer/ (gitignored). Resumable.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="models/MinerU2.5-tokenizer"
REPO="opendatalab/MinerU2.5-Pro-2605-1.2B"
mkdir -p "$DEST"

# Tokenizer/config first (small).
./scripts/fetch_tokenizer.sh

f="$DEST/model.safetensors"
if [ -s "$f" ] && python3 - "$f" <<'PY'
import sys, struct, json
p=sys.argv[1]
with open(p,'rb') as fh:
    n=struct.unpack('<Q', fh.read(8))[0]
    hdr=json.loads(fh.read(n))
need=8+n+max(v['data_offsets'][1] for k,v in hdr.items() if k!='__metadata__')
import os
sys.exit(0 if os.path.getsize(p)>=need else 1)
PY
then
  echo "weights already complete"; exit 0
fi

echo "Downloading model.safetensors (~2.2GB, resumable)"
curl -L -C - --retry 5 --retry-all-errors -o "$f" \
  "https://huggingface.co/$REPO/resolve/main/model.safetensors"
echo "weights ready: $f"
