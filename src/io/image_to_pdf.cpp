// Copyright (c) mlx-mineru.
// Image -> single-page PDF (see header). Mirrors MinerU images_bytes_to_pdf_bytes: the page is
// the image at DEFAULT_PDF_IMAGE_DPI = 200, so MediaBox = pixels * 72 / 200 points.
#include "mineru/image_to_pdf.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include "mineru/image_write.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#include "stb/stb_image.h"

namespace mineru {
namespace {

constexpr double kImageDpi = 200.0;  // MinerU DEFAULT_PDF_IMAGE_DPI

bool starts_with(const std::vector<uint8_t>& b, std::initializer_list<int> sig, size_t off = 0) {
  if (b.size() < off + sig.size()) return false;
  size_t i = off;
  for (int s : sig) {
    if (s >= 0 && b[i] != (uint8_t)s) return false;
    ++i;
  }
  return true;
}

struct Buf {
  std::vector<uint8_t> d;
  void str(const std::string& s) { d.insert(d.end(), s.begin(), s.end()); }
  void raw(const uint8_t* p, size_t n) { d.insert(d.end(), p, p + n); }
  size_t size() const { return d.size(); }
};

std::string fmt(double v) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.4f", v);
  return buf;
}

// Build a minimal single-page PDF embedding a JPEG image XObject (DCTDecode).
std::vector<uint8_t> build_pdf(const std::vector<uint8_t>& jpeg, int w, int h) {
  double W = w * 72.0 / kImageDpi, H = h * 72.0 / kImageDpi;
  Buf b;
  size_t off[6] = {0};
  b.str("%PDF-1.7\n%\xE2\xE3\xCF\xD3\n");
  auto begin = [&](int n) { off[n] = b.size(); b.str(std::to_string(n) + " 0 obj\n"); };

  begin(1);
  b.str("<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");
  begin(2);
  b.str("<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");
  begin(3);
  b.str("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 " + fmt(W) + " " + fmt(H) +
        "] /Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n");
  begin(4);
  b.str("<< /Type /XObject /Subtype /Image /Width " + std::to_string(w) + " /Height " +
        std::to_string(h) +
        " /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length " +
        std::to_string(jpeg.size()) + " >>\nstream\n");
  b.raw(jpeg.data(), jpeg.size());
  b.str("\nendstream\nendobj\n");
  begin(5);
  std::string content = "q " + fmt(W) + " 0 0 " + fmt(H) + " 0 0 cm /Im0 Do Q\n";
  b.str("<< /Length " + std::to_string(content.size()) + " >>\nstream\n");
  b.str(content);
  b.str("endstream\nendobj\n");

  size_t xref = b.size();
  b.str("xref\n0 6\n0000000000 65535 f \n");
  for (int i = 1; i <= 5; ++i) {
    char e[32];
    std::snprintf(e, sizeof(e), "%010zu 00000 n \n", off[i]);
    b.str(e);
  }
  b.str("trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n" + std::to_string(xref) + "\n%%EOF\n");
  return b.d;
}

}  // namespace

bool looks_like_image(const std::vector<uint8_t>& b) {
  return starts_with(b, {0x89, 0x50, 0x4E, 0x47}) ||           // PNG
         starts_with(b, {0xFF, 0xD8, 0xFF}) ||                 // JPEG
         starts_with(b, {0x42, 0x4D}) ||                       // BMP
         starts_with(b, {0x47, 0x49, 0x46, 0x38}) ||           // GIF8
         starts_with(b, {0x49, 0x49, 0x2A, 0x00}) ||           // TIFF (LE)
         starts_with(b, {0x4D, 0x4D, 0x00, 0x2A}) ||           // TIFF (BE)
         (starts_with(b, {0x52, 0x49, 0x46, 0x46}) &&          // RIFF....WEBP
          starts_with(b, {0x57, 0x45, 0x42, 0x50}, 8));
}

std::vector<uint8_t> image_bytes_to_pdf_bytes(const std::vector<uint8_t>& image_bytes) {
  bool is_jpeg = starts_with(image_bytes, {0xFF, 0xD8, 0xFF});
  if (is_jpeg) {
    int w = 0, h = 0, comp = 0;
    if (!stbi_info_from_memory(image_bytes.data(), (int)image_bytes.size(), &w, &h, &comp) ||
        w <= 0 || h <= 0)
      throw std::runtime_error("image_to_pdf: cannot read JPEG dimensions");
    return build_pdf(image_bytes, w, h);  // embed original JPEG verbatim
  }
  int w = 0, h = 0, comp = 0;
  // Decode as RGBA so we can flatten transparency onto white (PDF/JPEG have no alpha). For
  // opaque images this is identity; for transparent PNGs it avoids a black background.
  unsigned char* rgba = stbi_load_from_memory(image_bytes.data(), (int)image_bytes.size(), &w, &h,
                                              &comp, 4);
  if (!rgba)
    throw std::runtime_error(std::string("image_to_pdf: undecodable image (") +
                             stbi_failure_reason() + ")");
  std::vector<uint8_t> px((size_t)w * h * 3);
  for (size_t i = 0, n = (size_t)w * h; i < n; ++i) {
    int a = rgba[i * 4 + 3];
    for (int k = 0; k < 3; ++k)
      px[i * 3 + k] = (uint8_t)((rgba[i * 4 + k] * a + 255 * (255 - a)) / 255);
  }
  stbi_image_free(rgba);
  std::vector<uint8_t> jpeg = encode_jpeg(px, w, h, /*quality=*/95);
  return build_pdf(jpeg, w, h);
}

}  // namespace mineru
