// Copyright (c) mlx-mineru.
// Image input support: wrap a raster image (png/jpeg/bmp/gif/...) into a single-page PDF so
// the rest of the pipeline (PdfDocument render/extract -> layout/OCR/VLM) is identical to PDF
// input. Faithful to MinerU read_fn -> images_bytes_to_pdf_bytes (single page at 200 DPI).
#pragma once

#include <cstdint>
#include <vector>

namespace mineru {

// True if the bytes start with a supported image magic (PNG/JPEG/BMP/GIF/TIFF/WEBP).
bool looks_like_image(const std::vector<uint8_t>& bytes);

// Convert image bytes to a one-page PDF embedding the image at MinerU's 200 DPI. JPEG input is
// embedded verbatim (DCTDecode); other formats are decoded (stb_image) and re-encoded as JPEG.
// Throws std::runtime_error if the image cannot be decoded.
std::vector<uint8_t> image_bytes_to_pdf_bytes(const std::vector<uint8_t>& image_bytes);

}  // namespace mineru
