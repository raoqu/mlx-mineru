// Copyright (c) mlx-mineru.
// Minimal RGB8 -> JPEG writer (stb_image_write) for saving cropped image/chart
// regions referenced from the Markdown.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mineru {

// Write width*height*3 RGB8 to `path` as JPEG. Returns false on failure.
bool write_jpeg(const std::string& path, const std::vector<uint8_t>& rgb, int width, int height,
                int quality = 90);

// Encode width*height*3 RGB8 to an in-memory JPEG byte buffer (empty on failure). Used to
// inline image/chart crops as base64 data URIs in the web response (no file serving needed).
std::vector<uint8_t> encode_jpeg(const std::vector<uint8_t>& rgb, int width, int height,
                                 int quality = 90);

// Stable hex content hash (for image filenames).
std::string content_hash_hex(const std::vector<uint8_t>& data);

}  // namespace mineru
