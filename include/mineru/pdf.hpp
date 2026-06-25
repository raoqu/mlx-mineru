// Copyright (c) mlx-mineru.
// PDF rasterization via pdfium — faithful port of MinerU's page_to_image
// (mineru/utils/pdf_reader.py): scale = dpi/72 capped so the long side <= max_edge,
// bitmap = ceil(width_pt*scale) x ceil(height_pt*scale). RGB8, top-down rows.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

inline constexpr int kDefaultPdfDpi = 200;       // DEFAULT_PDF_IMAGE_DPI
inline constexpr int kDefaultMaxEdge = 3500;     // max_width_or_height

struct PageImage {
  int width = 0;          // pixels
  int height = 0;         // pixels
  double scale = 0.0;     // applied render scale
  double width_pt = 0.0;  // page width in points
  double height_pt = 0.0;
  std::vector<uint8_t> rgb;  // width*height*3, row-major, top-down
};

// One PDF text character (digital-PDF text path). bbox is in page points with a
// top-left origin (y down), matching middle_json span coordinates.
struct PdfChar {
  unsigned int cp = 0;  // unicode codepoint
  int idx = 0;          // original char index on the page (PDF order)
  double x0 = 0, y0 = 0, x1 = 0, y1 = 0;
};

// Owns a pdfium document. One global library init is handled internally
// (ref-counted). Throws std::runtime_error on load/render failure.
class PdfDocument {
 public:
  static PdfDocument open_file(const std::string& path, const std::string& password = "");
  static PdfDocument open_bytes(const std::vector<uint8_t>& bytes, const std::string& password = "");
  ~PdfDocument();
  PdfDocument(PdfDocument&&) noexcept;
  PdfDocument& operator=(PdfDocument&&) noexcept;
  PdfDocument(const PdfDocument&) = delete;
  PdfDocument& operator=(const PdfDocument&) = delete;

  int page_count() const;
  PageImage render_page(int index, int dpi = kDefaultPdfDpi, int max_edge = kDefaultMaxEdge) const;

  // Extract embedded text characters (unicode + bbox in top-left page points) for the
  // digital-PDF text path. Empty for scanned/image pages.
  std::vector<PdfChar> extract_chars(int index) const;

 private:
  PdfDocument();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Acquire/release the global pdfium library (ref-counted, thread-safe). PdfDocument brackets
// this automatically; standalone FPDF_* users (e.g. the draw_bbox overlay) call these directly.
void pdfium_acquire();
void pdfium_release();

}  // namespace mineru
