// Copyright (c) mlx-mineru.
#include "mineru/image_write.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_WRITE_NO_STDIO_DEPRECATED
#include "stb/stb_image_write.h"

namespace mineru {

bool write_jpeg(const std::string& path, const std::vector<uint8_t>& rgb, int width, int height,
                int quality) {
  if ((int)rgb.size() < width * height * 3) return false;
  return stbi_write_jpg(path.c_str(), width, height, 3, rgb.data(), quality) != 0;
}

std::string content_hash_hex(const std::vector<uint8_t>& data) {
  // FNV-1a 64-bit.
  uint64_t h = 1469598103934665603ULL;
  for (uint8_t b : data) {
    h ^= b;
    h *= 1099511628211ULL;
  }
  static const char* hex = "0123456789abcdef";
  std::string out(16, '0');
  for (int i = 15; i >= 0; --i) {
    out[i] = hex[h & 0xF];
    h >>= 4;
  }
  return out;
}

}  // namespace mineru
