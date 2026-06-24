#!/usr/bin/env bash
# Fetch prebuilt pdfium for macOS arm64 into third_party/pdfium/ (gitignored: 7.7MB
# binary). Retags the dylib install name to @rpath so the build can locate it.
#
# Pinned to the SAME pdfium build that MinerU's pypdfium2 wraps so rendering + text
# extraction byte-match the source (a render-version mismatch was traced to glyph/codepoint
# differences). pypdfium2 4.30.0 -> pdfium build 6462 (bblanchon tag chromium/6462).
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="third_party/pdfium"
PDFIUM_BUILD="${PDFIUM_BUILD:-6462}"  # match pypdfium2 4.30.0 (MinerU)

if [ -f "$DEST/lib/libpdfium.dylib" ] && [ -f "$DEST/include/fpdfview.h" ] \
   && grep -q "BUILD=${PDFIUM_BUILD}" "$DEST/VERSION" 2>/dev/null; then
  echo "pdfium already present ($(cat "$DEST/VERSION" 2>/dev/null | tr '\n' ' '))"
  exit 0
fi
rm -rf "$DEST"  # re-fetch if version differs

mkdir -p "$DEST"
URL="https://github.com/bblanchon/pdfium-binaries/releases/download/chromium%2F${PDFIUM_BUILD}/pdfium-mac-arm64.tgz"
echo "Downloading pdfium build ${PDFIUM_BUILD}: $URL"
curl -sSL -o /tmp/pdfium-mac-arm64.tgz "$URL"
tar -xzf /tmp/pdfium-mac-arm64.tgz -C "$DEST"
install_name_tool -id @rpath/libpdfium.dylib "$DEST/lib/libpdfium.dylib" 2>/dev/null || true
echo "pdfium ready: $(cat "$DEST/VERSION" 2>/dev/null | tr '\n' ' ')"
