// Copyright (c) mlx-mineru.
// Embedded web-UI assets (built from web/dist by scripts/embed_web.py) for `--web`.
#pragma once

#include <cstddef>
#include <string>

namespace mineru {

struct WebAsset {
  const unsigned char* data = nullptr;
  size_t size = 0;
  const char* mime = "application/octet-stream";
};

// Look up an embedded asset by request path ("/" -> "/index.html"). nullptr if not found.
const WebAsset* web_asset(const std::string& path);

// True if a built UI was bundled into the binary.
bool web_assets_bundled();

}  // namespace mineru
