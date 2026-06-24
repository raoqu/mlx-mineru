// Copyright (c) mlx-mineru.
// Full-page OCR chain, faithful to MinerU PytorchPaddleOCR.__call__:
// det -> sorted_boxes -> merge_det_boxes -> rotate-crop -> batched rec -> drop_score.
#include "mineru/ocr.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "mineru/image_preprocess.hpp"  // resize_bilinear_rgb8

namespace mineru {
namespace {

constexpr float kLineWHRatio = 4.0f;  // LINE_WIDTH_TO_HEIGHT_RATIO_THRESHOLD
constexpr float kRotateRatio = 1.5f;  // TEXT_REC_ROTATE_RATIO
constexpr int kRecBatch = 6;          // rec_batch_num

struct Bbox {
  float x0, y0, x1, y1;
};

// points_to_bbox: [pts0.x, pts0.y, pts1.x, pts2.y].
Bbox points_to_bbox(const Quad& q) { return {q[0][0], q[0][1], q[1][0], q[2][1]}; }
Quad bbox_to_points(const Bbox& b) {
  return {{{b.x0, b.y0}, {b.x1, b.y0}, {b.x1, b.y1}, {b.x0, b.y1}}};
}

// calculate_is_angle: True if the box is skewed (vertical extent deviates >20% from
// the mean side height).
bool calculate_is_angle(const Quad& q) {
  float height = ((q[3][1] - q[0][1]) + (q[2][1] - q[1][1])) / 2.0f;
  float d = q[2][1] - q[0][1];
  return !(0.8f * height <= d && d <= 1.2f * height);
}

bool overlaps_y_exceeds(const Bbox& a, const Bbox& b, float thr) {
  float overlap = std::max(0.0f, std::min(a.y1, b.y1) - std::max(a.y0, b.y0));
  float min_h = std::min(a.y1 - a.y0, b.y1 - b.y0);
  return min_h > 0 ? (overlap / min_h) > thr : false;
}

// merge_spans_to_line: group bboxes into lines by vertical overlap (threshold 0.6).
std::vector<std::vector<Bbox>> merge_spans_to_line(std::vector<Bbox> spans) {
  std::vector<std::vector<Bbox>> lines;
  if (spans.empty()) return lines;
  std::stable_sort(spans.begin(), spans.end(), [](const Bbox& a, const Bbox& b) {
    return a.y0 < b.y0;
  });
  std::vector<Bbox> cur{spans[0]};
  for (size_t i = 1; i < spans.size(); ++i) {
    if (overlaps_y_exceeds(spans[i], cur.back(), 0.6f)) {
      cur.push_back(spans[i]);
    } else {
      lines.push_back(cur);
      cur = {spans[i]};
    }
  }
  lines.push_back(cur);
  return lines;
}

// merge_overlapping_spans: merge horizontally overlapping bboxes on a line.
std::vector<Bbox> merge_overlapping_spans(std::vector<Bbox> spans) {
  std::vector<Bbox> merged;
  if (spans.empty()) return merged;
  std::stable_sort(spans.begin(), spans.end(), [](const Bbox& a, const Bbox& b) {
    return a.x0 < b.x0;
  });
  for (const Bbox& s : spans) {
    if (merged.empty() || merged.back().x1 < s.x0) {
      merged.push_back(s);
    } else {
      Bbox last = merged.back();
      merged.pop_back();
      merged.push_back({std::min(last.x0, s.x0), std::min(last.y0, s.y0),
                        std::max(last.x1, s.x1), std::max(last.y1, s.y1)});
    }
  }
  return merged;
}

// np.rot90 (CCW): out[i][j] = src[j][W-1-i]; result is (W x H).
std::vector<uint8_t> rot90(const std::vector<uint8_t>& src, int w, int h, int& ow, int& oh) {
  ow = h;
  oh = w;
  std::vector<uint8_t> out((size_t)ow * oh * 3);
  for (int i = 0; i < oh; ++i)        // i over new rows (= old cols)
    for (int j = 0; j < ow; ++j)      // j over new cols (= old rows reversed)
      for (int c = 0; c < 3; ++c)
        out[((size_t)i * ow + j) * 3 + c] = src[((size_t)j * w + (w - 1 - i)) * 3 + c];
  return out;
}

}  // namespace

std::vector<Quad> sorted_boxes(std::vector<Quad> boxes) {
  std::stable_sort(boxes.begin(), boxes.end(), [](const Quad& a, const Quad& b) {
    if (a[0][1] != b[0][1]) return a[0][1] < b[0][1];
    return a[0][0] < b[0][0];
  });
  int n = (int)boxes.size();
  for (int i = 0; i < n - 1; ++i) {
    for (int j = i; j >= 0; --j) {
      if (std::abs(boxes[j + 1][0][1] - boxes[j][0][1]) < 10.0f &&
          boxes[j + 1][0][0] < boxes[j][0][0]) {
        std::swap(boxes[j], boxes[j + 1]);
      } else {
        break;
      }
    }
  }
  return boxes;
}

std::vector<Quad> merge_det_boxes(const std::vector<Quad>& boxes) {
  std::vector<Bbox> dict_list;
  std::vector<Quad> angle_list;
  for (const Quad& q : boxes) {
    if (calculate_is_angle(q)) {
      angle_list.push_back(q);
      continue;
    }
    dict_list.push_back(points_to_bbox(q));
  }
  std::vector<Quad> out;
  for (const auto& line : merge_spans_to_line(dict_list)) {
    float min_x = line[0].x0, max_x = line[0].x1, min_y = line[0].y0, max_y = line[0].y1;
    for (const Bbox& b : line) {
      min_x = std::min(min_x, b.x0); max_x = std::max(max_x, b.x1);
      min_y = std::min(min_y, b.y0); max_y = std::max(max_y, b.y1);
    }
    if ((max_x - min_x) > (max_y - min_y) * kLineWHRatio) {
      for (const Bbox& s : merge_overlapping_spans(line)) out.push_back(bbox_to_points(s));
    } else {
      for (const Bbox& b : line) out.push_back(bbox_to_points(b));
    }
  }
  for (const Quad& q : angle_list) out.push_back(q);
  return out;
}

std::vector<uint8_t> get_rotate_crop_image(const std::vector<uint8_t>& rgb, int w, int h,
                                           const Quad& box, int& out_w, int& out_h) {
  // is_bbox_aligned_rect: exactly 2 unique x and 2 unique y.
  float xs[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  float ys[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  auto uniq = [](float* v) {
    float s[4] = {v[0], v[1], v[2], v[3]};
    std::sort(s, s + 4);
    int u = 1;
    for (int i = 1; i < 4; ++i)
      if (s[i] != s[i - 1]) ++u;
    return u;
  };
  auto crop_aligned = [&](int xmin, int ymin, int xmax, int ymax) {
    out_w = xmax - xmin;
    out_h = ymax - ymin;
    std::vector<uint8_t> c((size_t)out_w * out_h * 3);
    for (int y = 0; y < out_h; ++y)
      for (int x = 0; x < out_w; ++x)
        for (int ch = 0; ch < 3; ++ch)
          c[((size_t)y * out_w + x) * 3 + ch] =
              rgb[((size_t)(ymin + y) * w + (xmin + x)) * 3 + ch];
    return c;
  };

  std::vector<uint8_t> crop;
  int cw = 0, chh = 0;
  bool done = false;
  if (uniq(xs) == 2 && uniq(ys) == 2) {
    int xmin = (int)*std::min_element(xs, xs + 4), xmax = (int)*std::max_element(xs, xs + 4);
    int ymin = (int)*std::min_element(ys, ys + 4), ymax = (int)*std::max_element(ys, ys + 4);
    if (ymax - ymin > 0 && xmax - xmin > 0) {
      crop = crop_aligned(xmin, ymin, xmax, ymax);
      cw = out_w;
      chh = out_h;
      done = true;
    }
  }
  if (!done) {
    // Perspective unwarp (approximate: bilinear sampling instead of cv2 INTER_CUBIC;
    // hit only by skewed quads, which a.pdf never produces).
    auto norm = [](const std::array<float, 2>& a, const std::array<float, 2>& b) {
      return std::hypot(a[0] - b[0], a[1] - b[1]);
    };
    int cwi = (int)std::max(norm(box[0], box[1]), norm(box[2], box[3]));
    int chi = (int)std::max(norm(box[0], box[3]), norm(box[1], box[2]));
    cwi = std::max(cwi, 1);
    chi = std::max(chi, 1);
    crop.assign((size_t)cwi * chi * 3, 0);
    for (int y = 0; y < chi; ++y) {
      float ty = chi > 1 ? (float)y / (chi - 1) : 0.0f;
      for (int x = 0; x < cwi; ++x) {
        float tx = cwi > 1 ? (float)x / (cwi - 1) : 0.0f;
        // bilinear over the quad corners (p0 tl, p1 tr, p2 br, p3 bl)
        float sx = (1 - ty) * ((1 - tx) * box[0][0] + tx * box[1][0]) +
                   ty * ((1 - tx) * box[3][0] + tx * box[2][0]);
        float sy = (1 - ty) * ((1 - tx) * box[0][1] + tx * box[1][1]) +
                   ty * ((1 - tx) * box[3][1] + tx * box[2][1]);
        int ix = std::max(0, std::min(w - 1, (int)std::lround(sx)));
        int iy = std::max(0, std::min(h - 1, (int)std::lround(sy)));
        for (int ch = 0; ch < 3; ++ch)
          crop[((size_t)y * cwi + x) * 3 + ch] = rgb[((size_t)iy * w + ix) * 3 + ch];
      }
    }
    cw = cwi;
    chh = chi;
  }

  // rotate_vertical_crop_if_needed: tall crop -> np.rot90.
  if (chh > 0 && cw > 0 && (float)chh / cw >= kRotateRatio) {
    return rot90(crop, cw, chh, out_w, out_h);
  }
  out_w = cw;
  out_h = chh;
  return crop;
}

OcrPipeline OcrPipeline::load(const std::string& det_onnx, const std::string& rec_onnx,
                              const std::string& dict) {
  return OcrPipeline(TextDetector::load(det_onnx), TextRecognizer::load(rec_onnx, dict));
}

std::vector<OcrLine> OcrPipeline::run(const std::vector<uint8_t>& rgb, int w, int h,
                                      float drop_score) const {
  auto det_boxes = det_.detect(rgb, w, h);
  std::vector<Quad> quads;
  quads.reserve(det_boxes.size());
  for (const auto& db : det_boxes) {
    Quad q;
    for (int i = 0; i < 4; ++i) q[i] = {(float)db.pts[i][0], (float)db.pts[i][1]};
    quads.push_back(q);
  }
  quads = sorted_boxes(std::move(quads));
  quads = merge_det_boxes(quads);

  // Crop each box (rotate-for-text-rec).
  struct Crop {
    std::vector<uint8_t> rgb;
    int w, h;
  };
  std::vector<Crop> crops(quads.size());
  std::vector<double> aspect(quads.size());
  for (size_t i = 0; i < quads.size(); ++i) {
    int cw, ch;
    crops[i].rgb = get_rotate_crop_image(rgb, w, h, quads[i], cw, ch);
    crops[i].w = cw;
    crops[i].h = ch;
    aspect[i] = ch > 0 ? (double)cw / ch : 0.0;
  }

  // Batched rec: sort by aspect, batches of kRecBatch share the widest aspect.
  std::vector<int> idx(crops.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return aspect[a] < aspect[b]; });
  std::vector<RecResult> recs(crops.size());
  for (size_t beg = 0; beg < idx.size(); beg += kRecBatch) {
    size_t end = std::min(idx.size(), beg + kRecBatch);
    double max_wh = aspect[idx[end - 1]];
    for (size_t ino = beg; ino < end; ++ino) {
      int j = idx[ino];
      if (crops[j].w <= 0 || crops[j].h <= 0) continue;
      recs[j] = rec_.recognize(crops[j].rgb, crops[j].w, crops[j].h, max_wh);
    }
  }

  std::vector<OcrLine> out;
  for (size_t i = 0; i < quads.size(); ++i) {
    if (recs[i].score >= drop_score) out.push_back({quads[i], recs[i].text, recs[i].score});
  }
  return out;
}

}  // namespace mineru
