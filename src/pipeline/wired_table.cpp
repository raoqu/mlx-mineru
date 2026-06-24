// Copyright (c) mlx-mineru.
// Wired-table UNet — Stage 1: preprocess (keep-ratio bicubic resize to 1024 + normalize) +
// onnx inference -> line segmentation map. Faithful to MinerU TSRUnet.preprocess/infer.
#include "mineru/wired_table.hpp"

#include <algorithm>
#include <cmath>

#include "mineru/image_preprocess.hpp"  // resize_bicubic_rgb8
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {
constexpr int kInp = 1024;
// UNet normalization (ImageNet-style, RGB).
const std::array<float, 3> kMean{123.675f, 116.28f, 103.53f};
const std::array<float, 3> kStd{58.395f, 57.12f, 57.375f};
}  // namespace

struct WiredTableRecognizer::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-wired"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::string in_name, out_name;
};

WiredTableRecognizer::WiredTableRecognizer() : impl_(std::make_unique<Impl>()) {}
WiredTableRecognizer::~WiredTableRecognizer() = default;
WiredTableRecognizer::WiredTableRecognizer(WiredTableRecognizer&&) noexcept = default;
WiredTableRecognizer& WiredTableRecognizer::operator=(WiredTableRecognizer&&) noexcept = default;

WiredTableRecognizer WiredTableRecognizer::load(const std::string& onnx) {
  WiredTableRecognizer r;
  Impl& m = *r.impl_;
  m.session = std::make_unique<Ort::Session>(m.env, onnx.c_str(), m.opts);
  Ort::AllocatorWithDefaultOptions alloc;
  m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
  m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  return r;
}

std::vector<uint8_t> WiredTableRecognizer::segment(const std::vector<uint8_t>& rgb, int w, int h,
                                                   int& nh, int& nw) const {
  const Impl& m = *impl_;
  // imrescale: keep aspect, longest side -> 1024 (rounded).
  double scale = (double)kInp / std::max(w, h);
  nw = (int)(w * scale + 0.5);
  nh = (int)(h * scale + 0.5);
  std::vector<uint8_t> resized = resize_bicubic_rgb8(rgb, w, h, nw, nh);

  // NormalizeImage (RGB), CHW.
  std::vector<float> input(static_cast<size_t>(3) * nh * nw);
  for (int y = 0; y < nh; ++y)
    for (int x = 0; x < nw; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * nh + y) * nw + x] =
            (resized[(static_cast<size_t>(y) * nw + x) * 3 + c] - kMean[c]) / kStd[c];

  std::array<int64_t, 4> ishape{1, 3, nh, nw};
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {m.in_name.c_str()};
  const char* out_names[] = {m.out_name.c_str()};
  auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                        out_names, 1);
  // Output [1,1,nh,nw] int64 -> uint8 seg map.
  auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int oh = (int)oshape[oshape.size() - 2], ow = (int)oshape[oshape.size() - 1];
  const int64_t* seg = outs[0].GetTensorData<int64_t>();
  std::vector<uint8_t> out((size_t)oh * ow);
  for (size_t i = 0; i < out.size(); ++i) out[i] = (uint8_t)seg[i];
  nh = oh;
  nw = ow;
  return out;
}

// ---- Stage 2: segmentation -> cell polygons --------------------------------
namespace {
using Mask = std::vector<uint8_t>;  // H*W, row-major

// Nearest-neighbour resize of a label mask to (ow,oh) — matches cv2.resize on label maps
// closely enough for line topology (the seg is then thresholded >0).
Mask resize_mask(const Mask& m, int iw, int ih, int ow, int oh) {
  Mask o((size_t)ow * oh);
  for (int y = 0; y < oh; ++y) {
    int sy = std::min(ih - 1, (int)((y + 0.5) * ih / oh));
    for (int x = 0; x < ow; ++x) {
      int sx = std::min(iw - 1, (int)((x + 0.5) * iw / ow));
      o[(size_t)y * ow + x] = m[(size_t)sy * iw + sx];
    }
  }
  return o;
}

// Morphological close (dilate then erode) with a rect kernel (kw x kh), binary mask.
Mask morph_close(const Mask& m, int W, int H, int kw, int kh) {
  auto dil = [&](const Mask& s) {
    Mask r((size_t)W * H, 0);
    int ax = kw / 2, ay = kh / 2;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        if (!s[(size_t)y * W + x]) continue;
        for (int dy = -ay; dy <= kh - 1 - ay; ++dy)
          for (int dx = -ax; dx <= kw - 1 - ax; ++dx) {
            int ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < H && nx >= 0 && nx < W) r[(size_t)ny * W + nx] = 1;
          }
      }
    return r;
  };
  auto ero = [&](const Mask& s) {
    Mask r((size_t)W * H, 0);
    int ax = kw / 2, ay = kh / 2;
    for (int y = 0; y < H; ++y)
      for (int x = 0; x < W; ++x) {
        bool all = true;
        for (int dy = -ay; dy <= kh - 1 - ay && all; ++dy)
          for (int dx = -ax; dx <= kw - 1 - ax && all; ++dx) {
            int ny = y + dy, nx = x + dx;
            if (ny < 0 || ny >= H || nx < 0 || nx >= W || !s[(size_t)ny * W + nx]) all = false;
          }
        r[(size_t)y * W + x] = all ? 1 : 0;
      }
    return r;
  };
  return ero(dil(m));
}

struct Pt { double x, y; };
// 8-connected components -> per-component pixel coords + bbox.
struct Comp { std::vector<Pt> coords; int x0, y0, x1, y1; };
std::vector<Comp> connected(const Mask& m, int W, int H) {
  std::vector<int> lab((size_t)W * H, 0);
  std::vector<Comp> out;
  std::vector<std::pair<int, int>> st;
  const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1}, dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  int next = 0;
  for (int y = 0; y < H; ++y)
    for (int x = 0; x < W; ++x) {
      if (!m[(size_t)y * W + x] || lab[(size_t)y * W + x]) continue;
      ++next;
      st.clear();
      st.push_back({x, y});
      lab[(size_t)y * W + x] = next;
      Comp c{{}, x, y, x, y};
      while (!st.empty()) {
        auto [cx, cy] = st.back();
        st.pop_back();
        c.coords.push_back({(double)cx, (double)cy});
        c.x0 = std::min(c.x0, cx); c.y0 = std::min(c.y0, cy);
        c.x1 = std::max(c.x1, cx); c.y1 = std::max(c.y1, cy);
        for (int k = 0; k < 8; ++k) {
          int nx = cx + dx[k], ny = cy + dy[k];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          size_t ni = (size_t)ny * W + nx;
          if (m[ni] && !lab[ni]) { lab[ni] = next; st.push_back({nx, ny}); }
        }
      }
      out.push_back(std::move(c));
    }
  return out;
}

// minAreaRect (rotating calipers) -> 4 corner points (cv2.boxPoints order-agnostic).
std::array<Pt, 4> min_area_rect(const std::vector<Pt>& pts) {
  // convex hull (Andrew monotone chain)
  std::vector<Pt> p = pts;
  std::sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  p.erase(std::unique(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
            return a.x == b.x && a.y == b.y;
          }), p.end());
  int n = (int)p.size(), k = 0;
  std::vector<Pt> hull(2 * n);
  auto cross = [](const Pt& O, const Pt& A, const Pt& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
  };
  if (n < 3) { return {p[0], p[0], n > 1 ? p[1] : p[0], n > 1 ? p[1] : p[0]}; }
  for (int i = 0; i < n; ++i) { while (k >= 2 && cross(hull[k - 2], hull[k - 1], p[i]) <= 0) k--; hull[k++] = p[i]; }
  for (int i = n - 2, t = k + 1; i >= 0; --i) { while (k >= t && cross(hull[k - 2], hull[k - 1], p[i]) <= 0) k--; hull[k++] = p[i]; }
  hull.resize(k - 1);
  double best = 1e300;
  std::array<Pt, 4> box{};
  int m = (int)hull.size();
  for (int i = 0; i < m; ++i) {
    Pt a = hull[i], b = hull[(i + 1) % m];
    double ex = b.x - a.x, ey = b.y - a.y, len = std::hypot(ex, ey);
    if (len < 1e-9) continue;
    double ux = ex / len, uy = ey / len, vx = -uy, vy = ux;
    double minu = 1e300, maxu = -1e300, minv = 1e300, maxv = -1e300;
    for (const Pt& q : hull) {
      double du = (q.x - a.x) * ux + (q.y - a.y) * uy, dv = (q.x - a.x) * vx + (q.y - a.y) * vy;
      minu = std::min(minu, du); maxu = std::max(maxu, du);
      minv = std::min(minv, dv); maxv = std::max(maxv, dv);
    }
    double area = (maxu - minu) * (maxv - minv);
    if (area < best) {
      best = area;
      auto corner = [&](double cu, double cv) {
        return Pt{a.x + cu * ux + cv * vx, a.y + cu * uy + cv * vy};
      };
      box = {corner(minu, minv), corner(maxu, minv), corner(maxu, maxv), corner(minu, maxv)};
    }
  }
  return box;
}

// _order_points: tl (min sum), br (max sum), tr (min diff y-x), bl (max diff).
std::array<Pt, 4> order_points(std::array<Pt, 4> p) {
  std::array<Pt, 4> o;
  int tl = 0, br = 0, tr = 0, bl = 0;
  double smin = 1e300, smax = -1e300, dmin = 1e300, dmax = -1e300;
  for (int i = 0; i < 4; ++i) {
    double s = p[i].x + p[i].y, d = p[i].y - p[i].x;
    if (s < smin) { smin = s; tl = i; }
    if (s > smax) { smax = s; br = i; }
    if (d < dmin) { dmin = d; tr = i; }
    if (d > dmax) { dmax = d; bl = i; }
  }
  o = {p[tl], p[tr], p[br], p[bl]};
  return o;
}

// min_area_rect for a line component -> [xmin,ymin,xmax,ymax] center line (get_table_line).
std::array<double, 4> line_rect(const std::vector<Pt>& coords) {
  auto box = order_points(min_area_rect(coords));
  double x1 = box[0].x, y1 = box[0].y, x2 = box[1].x, y2 = box[1].y;
  double x3 = box[2].x, y3 = box[2].y, x4 = box[3].x, y4 = box[3].y;
  double w = (std::hypot(x2 - x1, y2 - y1) + std::hypot(x3 - x4, y3 - y4)) / 2;
  double h = (std::hypot(x2 - x3, y2 - y3) + std::hypot(x1 - x4, y1 - y4)) / 2;
  if (w < h)
    return {(x1 + x2) / 2, (y1 + y2) / 2, (x3 + x4) / 2, (y3 + y4) / 2};
  return {(x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2, (y2 + y3) / 2};
}

// get_table_line: line components whose bbox span on the given axis exceeds lineW.
std::vector<std::array<double, 4>> table_lines(const Mask& bin, int W, int H, int axis, int lineW) {
  std::vector<std::array<double, 4>> lines;
  for (auto& c : connected(bin, W, H)) {
    int span = (axis == 1) ? (c.x1 - c.x0) : (c.y1 - c.y0);  // axis1: vertical (x span? per src)
    // per source: axis==1 uses bbox[2]-bbox[0] (rows), axis==0 uses bbox[3]-bbox[1] (cols).
    span = (axis == 1) ? (c.y1 - c.y0) : (c.x1 - c.x0);
    if (span > lineW) lines.push_back(line_rect(c.coords));
  }
  return lines;
}

// draw a thick line (no anti-alias) of value 255 into mask.
void draw_line(Mask& im, int W, int H, double x1, double y1, double x2, double y2, int lw) {
  int steps = (int)std::max(std::abs(x2 - x1), std::abs(y2 - y1)) + 1;
  int r = lw / 2;
  for (int s = 0; s <= steps; ++s) {
    double t = (double)s / steps;
    int px = (int)std::lround(x1 + t * (x2 - x1)), py = (int)std::lround(y1 + t * (y2 - y1));
    for (int dy = -r; dy <= r; ++dy)
      for (int dx = -r; dx <= r; ++dx) {
        int nx = px + dx, ny = py + dy;
        if (nx >= 0 && ny >= 0 && nx < W && ny < H) im[(size_t)ny * W + nx] = 255;
      }
  }
}
}  // namespace

std::vector<std::array<float, 8>> WiredTableRecognizer::cell_polygons(
    const std::vector<uint8_t>& rgb, int w, int h) const {
  int nh, nw;
  std::vector<uint8_t> seg = segment(rgb, w, h, nh, nw);
  // Split horizontal (==1) / vertical (==2), resize to original crop size.
  Mask hp((size_t)nh * nw), vp((size_t)nh * nw);
  for (size_t i = 0; i < hp.size(); ++i) { hp[i] = seg[i] == 1 ? 1 : 0; vp[i] = seg[i] == 2 ? 1 : 0; }
  hp = resize_mask(hp, nw, nh, w, h);
  vp = resize_mask(vp, nw, nh, w, h);
  // Morphology close: kernels sized like MinerU (sqrt(dim)*1.2).
  int hk = (int)(std::sqrt((double)nw) * 1.2), vk = (int)(std::sqrt((double)nh) * 1.2);
  vp = morph_close(vp, w, h, 1, std::max(1, vk));
  hp = morph_close(hp, w, h, std::max(1, hk), 1);
  // Extract lines.
  auto cols = table_lines(vp, w, h, 1, 30);
  auto rows = table_lines(hp, w, h, 0, 50);
  // Draw all lines into a fresh canvas, then cells = connected components of (img < 255).
  Mask line_img((size_t)w * h, 0);
  for (auto& l : rows) draw_line(line_img, w, h, l[0], l[1], l[2], l[3], 2);
  for (auto& l : cols) draw_line(line_img, w, h, l[0], l[1], l[2], l[3], 2);
  Mask cellmask((size_t)w * h);
  for (size_t i = 0; i < cellmask.size(); ++i) cellmask[i] = line_img[i] < 255 ? 1 : 0;
  // cal_region_boxes: components, filter big (>3/4 area) + small (<15px), min_area_rect.
  std::vector<std::array<float, 8>> out;
  double area = (double)w * h;
  for (auto& c : connected(cellmask, w, h)) {
    double barea = (double)(c.x1 - c.x0) * (c.y1 - c.y0);
    if (barea > area * 3 / 4) continue;
    auto box = order_points(min_area_rect(c.coords));
    double bw = (std::hypot(box[1].x - box[0].x, box[1].y - box[0].y) +
                 std::hypot(box[2].x - box[3].x, box[2].y - box[3].y)) / 2;
    double bh = (std::hypot(box[1].x - box[2].x, box[1].y - box[2].y) +
                 std::hypot(box[0].x - box[3].x, box[0].y - box[3].y)) / 2;
    if (bw * bh >= 0.5 * area) continue;
    if (bw < 15 || bh < 15) continue;
    out.push_back({(float)box[0].x, (float)box[0].y, (float)box[1].x, (float)box[1].y,
                   (float)box[2].x, (float)box[2].y, (float)box[3].x, (float)box[3].y});
  }
  return out;
}

}  // namespace mineru

