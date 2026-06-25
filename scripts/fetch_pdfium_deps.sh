#!/usr/bin/env bash
# Fetch the external source dependencies pdfium needs (those NOT vendored in its own repo),
# pinned to the revisions in pdfium's DEPS. Used by the from-source trimmed static build
# (scripts/build_pdfium_static_src.sh) — no depot_tools / gclient required.
#
# In-tree already (no fetch): lcms, libopenjpeg, agg23, libtiff.
# System lib (no fetch): zlib  -> macOS /usr/lib/libz  (USE_SYSTEM_ZLIB).
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="third_party/.pdfium-src"
GIT=https://chromium.googlesource.com

# name|repo-path|revision
DEPS="
freetype|chromium/src/third_party/freetype2|b08a2eb0dd37f4a6c886fa5b0ecf5b3e1d27aac7
fast_float|external/github.com/fastfloat/fast_float|05087a303dad9c98768b33c829d398223a649bc6
libjpeg_turbo|chromium/deps/libjpeg_turbo|640f254ad0fa03f6b1f29f89b7dd9366f2f6e533
abseil-cpp|chromium/src/third_party/abseil-cpp|5e42a36a85a252d8cdee6c39661d2bfd9883fd5c
partition_allocator|chromium/src/base/allocator/partition_allocator|390c4b91f769aad03ea13d2a479d7430406302f2
build|chromium/src/build|613f5c13bccbc15bd7ce8da9acb13ac06459f8cb
"

fetch() {  # dest repo rev
  local dest="$1" repo="$2" rev="$3"
  if [ -d "$dest/.git" ]; then echo "  have $(basename "$dest")"; return 0; fi
  echo "  fetching $(basename "$dest") @ ${rev:0:10} ..."
  rm -rf "$dest"; mkdir -p "$dest"
  ( cd "$dest"
    git init -q
    git remote add origin "$GIT/$repo.git"
    # Fetch just the pinned commit (fast, shallow); retry for flaky googlesource.
    for i in 1 2 3 4 5; do
      git fetch -q --depth 1 origin "$rev" && break || { echo "    retry $i"; sleep 3; }
    done
    git checkout -q FETCH_HEAD )
}

echo "[pdfium-deps] fetching external sources into $SRC/third_party/ ..."
while IFS='|' read -r name repo rev; do
  [ -z "$name" ] && continue
  case "$name" in
    build) fetch "$SRC/build_cfg" "$repo" "$rev" ;;  # only need build_config.h + buildflag.h
    freetype)   fetch "$SRC/third_party/freetype/src" "$repo" "$rev" ;;
    fast_float) fetch "$SRC/third_party/fast_float/src" "$repo" "$rev" ;;
    *)        fetch "$SRC/third_party/$name" "$repo" "$rev" ;;
  esac
done <<< "$DEPS"

# pdfium expects build/{build_config.h,buildflag.h} on the include path; expose them.
mkdir -p "$SRC/build"
cp -f "$SRC/build_cfg/build_config.h" "$SRC/build/" 2>/dev/null || true
cp -f "$SRC/build_cfg/buildflag.h"    "$SRC/build/" 2>/dev/null || true

echo "[pdfium-deps] done."
