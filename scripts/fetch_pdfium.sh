#!/usr/bin/env bash
# Fetch prebuilt pdfium for macOS arm64 into third_party/pdfium/ (gitignored: 7.7MB
# binary). Retags the dylib install name to @rpath so the build can locate it.
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="third_party/pdfium"

if [ -f "$DEST/lib/libpdfium.dylib" ] && [ -f "$DEST/include/fpdfview.h" ]; then
  echo "pdfium already present ($(cat "$DEST/VERSION" 2>/dev/null | tr '\n' ' '))"
  exit 0
fi

mkdir -p "$DEST"
URL=$(curl -sSL "https://api.github.com/repos/bblanchon/pdfium-binaries/releases/latest" \
  | grep -oE '"browser_download_url": "[^"]+pdfium-mac-arm64.tgz"' | head -1 | sed 's/.*: "//;s/"$//')
echo "Downloading $URL"
curl -sSL -o /tmp/pdfium-mac-arm64.tgz "$URL"
tar -xzf /tmp/pdfium-mac-arm64.tgz -C "$DEST"
install_name_tool -id @rpath/libpdfium.dylib "$DEST/lib/libpdfium.dylib" 2>/dev/null || true
echo "pdfium ready: $(cat "$DEST/VERSION" 2>/dev/null | tr '\n' ' ')"
