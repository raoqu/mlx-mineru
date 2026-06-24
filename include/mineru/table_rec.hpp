// Copyright (c) mlx-mineru.
// Table structure recognition (SLANet+, pipeline backend): table crop -> HTML structure
// tokens + cell bboxes, then match OCR text into cells -> HTML. Faithful port of MinerU
// slanet_plus (TableStructurer + adapt_slanet_plus + TableMatch).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mineru {

// One OCR text box for table matching: axis-aligned-ish quad + text.
struct TableOcrItem {
  std::array<std::array<float, 2>, 4> box;  // 4 corners in table-crop pixel coords
  std::string text;
  float score;
};

struct TableStructure {
  std::vector<std::string> tokens;            // structure tokens (incl. <html>..</html>)
  std::vector<std::array<float, 8>> cells;    // cell bboxes (8 coords), adapted to 488-space
};

class TableRecognizer {
 public:
  // onnx = slanet-plus.onnx, dict = table_structure_dict.txt (50 tokens incl sos/eos).
  static TableRecognizer load(const std::string& onnx, const std::string& dict);
  ~TableRecognizer();
  TableRecognizer(TableRecognizer&&) noexcept;
  TableRecognizer& operator=(TableRecognizer&&) noexcept;

  // rgb: w*h*3 RGB8 table crop. Returns structure tokens + adapted cell bboxes.
  TableStructure recognize_structure(const std::vector<uint8_t>& rgb, int w, int h) const;

  // Full path: structure + match OCR items into cells -> HTML string.
  std::string recognize_html(const std::vector<uint8_t>& rgb, int w, int h,
                             const std::vector<TableOcrItem>& ocr) const;

 private:
  TableRecognizer();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Exposed for tests: faithful TableMatch (OCR boxes + structure -> HTML).
std::string table_match(const std::vector<std::string>& structure,
                        const std::vector<std::array<float, 8>>& cells,
                        const std::vector<TableOcrItem>& ocr);

}  // namespace mineru
