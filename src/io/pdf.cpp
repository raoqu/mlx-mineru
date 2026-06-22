// Copyright (c) mlx-mineru.
#include "mineru/pdf.hpp"

#include <atomic>
#include <cmath>
#include <fstream>
#include <mutex>
#include <stdexcept>

#include "fpdf_formfill.h"
#include "fpdfview.h"

namespace mineru {
namespace {

// Ref-counted global pdfium library init/teardown (FPDF_InitLibrary is global).
std::mutex g_lib_mutex;
int g_lib_refs = 0;

void library_acquire() {
  std::lock_guard<std::mutex> lock(g_lib_mutex);
  if (g_lib_refs++ == 0) {
    FPDF_LIBRARY_CONFIG cfg{};
    cfg.version = 2;
    cfg.m_pUserFontPaths = nullptr;
    cfg.m_pIsolate = nullptr;
    cfg.m_v8EmbedderSlot = 0;
    FPDF_InitLibraryWithConfig(&cfg);
  }
}

void library_release() {
  std::lock_guard<std::mutex> lock(g_lib_mutex);
  if (--g_lib_refs == 0) FPDF_DestroyLibrary();
}

std::string pdfium_error() {
  switch (FPDF_GetLastError()) {
    case FPDF_ERR_SUCCESS: return "success";
    case FPDF_ERR_FILE: return "file not found or could not be opened";
    case FPDF_ERR_FORMAT: return "not a PDF or corrupted";
    case FPDF_ERR_PASSWORD: return "password required or incorrect";
    case FPDF_ERR_SECURITY: return "unsupported security scheme";
    case FPDF_ERR_PAGE: return "page not found or content error";
    default: return "unknown error";
  }
}

}  // namespace

struct PdfDocument::Impl {
  FPDF_DOCUMENT doc = nullptr;
  FPDF_FORMHANDLE form = nullptr;
  std::vector<uint8_t> owned_bytes;  // kept alive for FPDF_LoadMemDocument

  // Lazily create the form-fill environment (pypdfium2 does this and calls
  // FPDF_FFLDraw, so form-field widgets render identically).
  void ensure_form() {
    if (form || !doc) return;
    static FPDF_FORMFILLINFO ffi{};  // zero callbacks; version 2 is enough to draw
    ffi.version = 2;
    form = FPDFDOC_InitFormFillEnvironment(doc, &ffi);
  }
  void close() {
    if (form) { FPDFDOC_ExitFormFillEnvironment(form); form = nullptr; }
    if (doc) { FPDF_CloseDocument(doc); doc = nullptr; }
  }
};

PdfDocument::PdfDocument() : impl_(std::make_unique<Impl>()) { library_acquire(); }

PdfDocument::~PdfDocument() {
  if (impl_) {
    impl_->close();
    library_release();
  }
}

PdfDocument::PdfDocument(PdfDocument&& o) noexcept : impl_(std::move(o.impl_)) {}

PdfDocument& PdfDocument::operator=(PdfDocument&& o) noexcept {
  if (this != &o) {
    if (impl_) {
      impl_->close();
      library_release();
    }
    impl_ = std::move(o.impl_);
  }
  return *this;
}

PdfDocument PdfDocument::open_file(const std::string& path, const std::string& password) {
  PdfDocument d;
  d.impl_->doc = FPDF_LoadDocument(path.c_str(), password.empty() ? nullptr : password.c_str());
  if (!d.impl_->doc) throw std::runtime_error("PdfDocument::open_file failed: " + pdfium_error() + " (" + path + ")");
  return d;
}

PdfDocument PdfDocument::open_bytes(const std::vector<uint8_t>& bytes, const std::string& password) {
  PdfDocument d;
  d.impl_->owned_bytes = bytes;
  d.impl_->doc = FPDF_LoadMemDocument(
      d.impl_->owned_bytes.data(), static_cast<int>(d.impl_->owned_bytes.size()),
      password.empty() ? nullptr : password.c_str());
  if (!d.impl_->doc) throw std::runtime_error("PdfDocument::open_bytes failed: " + pdfium_error());
  return d;
}

int PdfDocument::page_count() const { return FPDF_GetPageCount(impl_->doc); }

PageImage PdfDocument::render_page(int index, int dpi, int max_edge) const {
  FPDF_PAGE page = FPDF_LoadPage(impl_->doc, index);
  if (!page) throw std::runtime_error("render_page: load failed: " + pdfium_error());

  // get_size() in points (FPDF_GetPageWidthF/HeightF return float, as pypdfium2 uses).
  double w_pt = FPDF_GetPageWidthF(page);
  double h_pt = FPDF_GetPageHeightF(page);

  double scale = static_cast<double>(dpi) / 72.0;
  double long_side = std::max(w_pt, h_pt);
  if (long_side * scale > max_edge) scale = static_cast<double>(max_edge) / long_side;

  int img_w = static_cast<int>(std::ceil(w_pt * scale));
  int img_h = static_cast<int>(std::ceil(h_pt * scale));

  // BGRA bitmap, white background, render with annotations (pypdfium2 draw_annots=True).
  FPDF_BITMAP bmp = FPDFBitmap_Create(img_w, img_h, /*alpha=*/0);  // BGRx, 4 bytes/px
  if (!bmp) { FPDF_ClosePage(page); throw std::runtime_error("render_page: bitmap alloc failed"); }
  FPDFBitmap_FillRect(bmp, 0, 0, img_w, img_h, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bmp, page, 0, 0, img_w, img_h, /*rotate=*/0, FPDF_ANNOT);
  // Draw form-field widgets (pypdfium2 may_draw_forms=True default).
  impl_->ensure_form();
  if (impl_->form) {
    FPDF_FFLDraw(impl_->form, bmp, page, 0, 0, img_w, img_h, /*rotate=*/0, FPDF_ANNOT);
  }

  const uint8_t* buf = static_cast<const uint8_t*>(FPDFBitmap_GetBuffer(bmp));
  int stride = FPDFBitmap_GetStride(bmp);

  PageImage out;
  out.width = img_w;
  out.height = img_h;
  out.scale = scale;
  out.width_pt = w_pt;
  out.height_pt = h_pt;
  out.rgb.resize(static_cast<size_t>(img_w) * img_h * 3);
  // BGRx -> RGB.
  for (int y = 0; y < img_h; ++y) {
    const uint8_t* src = buf + static_cast<size_t>(y) * stride;
    uint8_t* dst = out.rgb.data() + static_cast<size_t>(y) * img_w * 3;
    for (int x = 0; x < img_w; ++x) {
      dst[x * 3 + 0] = src[x * 4 + 2];  // R
      dst[x * 3 + 1] = src[x * 4 + 1];  // G
      dst[x * 3 + 2] = src[x * 4 + 0];  // B
    }
  }

  FPDFBitmap_Destroy(bmp);
  FPDF_ClosePage(page);
  return out;
}

}  // namespace mineru
