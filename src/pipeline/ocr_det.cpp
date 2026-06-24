// Copyright (c) mlx-mineru.
// Faithful port of mineru pytorchocr DBPostProcess (box_type="quad").
#include "mineru/ocr_det.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace mineru {
namespace {

struct Pt {
  double x, y;
};

// numpy np.round: round half to even (banker's rounding).
inline long round_half_even(double v) {
  double r = std::nearbyint(v);  // honors FE_TONEAREST (round-half-to-even) by default
  return (long)r;
}

// Andrew's monotone-chain convex hull (CCW, no collinear duplicates).
std::vector<Pt> convex_hull(std::vector<Pt> p) {
  std::sort(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
    return a.x < b.x || (a.x == b.x && a.y < b.y);
  });
  p.erase(std::unique(p.begin(), p.end(), [](const Pt& a, const Pt& b) {
            return a.x == b.x && a.y == b.y;
          }),
          p.end());
  int n = (int)p.size(), k = 0;
  if (n < 3) return p;
  std::vector<Pt> h(2 * n);
  auto cross = [](const Pt& O, const Pt& A, const Pt& B) {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
  };
  for (int i = 0; i < n; ++i) {
    while (k >= 2 && cross(h[k - 2], h[k - 1], p[i]) <= 0) k--;
    h[k++] = p[i];
  }
  for (int i = n - 2, t = k + 1; i >= 0; --i) {
    while (k >= t && cross(h[k - 2], h[k - 1], p[i]) <= 0) k--;
    h[k++] = p[i];
  }
  h.resize(k - 1);
  return h;
}

// Min-area rectangle (rotating calipers over hull edges). Returns center, the two
// half-axis vectors (already scaled to half-extent), and the min side length.
struct MinRect {
  Pt c;
  Pt ax, ay;  // axis unit vectors
  double hw, hh;  // half extents along ax, ay
};

bool min_area_rect(const std::vector<Pt>& pts, MinRect& out, double& min_side) {
  std::vector<Pt> h = convex_hull(pts);
  if (h.size() < 2) return false;
  double best_area = 1e300;
  int m = (int)h.size();
  for (int i = 0; i < m; ++i) {
    Pt a = h[i], b = h[(i + 1) % m];
    double ex = b.x - a.x, ey = b.y - a.y;
    double len = std::hypot(ex, ey);
    if (len < 1e-9) continue;
    double ux = ex / len, uy = ey / len;     // edge direction
    double vx = -uy, vy = ux;                 // perpendicular
    double minu = 1e300, maxu = -1e300, minv = 1e300, maxv = -1e300;
    for (const Pt& q : h) {
      double du = (q.x - a.x) * ux + (q.y - a.y) * uy;
      double dv = (q.x - a.x) * vx + (q.y - a.y) * vy;
      minu = std::min(minu, du); maxu = std::max(maxu, du);
      minv = std::min(minv, dv); maxv = std::max(maxv, dv);
    }
    double w = maxu - minu, ht = maxv - minv, area = w * ht;
    if (area < best_area) {
      best_area = area;
      double cu = (minu + maxu) / 2, cv = (minv + maxv) / 2;
      out.c = {a.x + cu * ux + cv * vx, a.y + cu * uy + cv * vy};
      out.ax = {ux, uy};
      out.ay = {vx, vy};
      out.hw = w / 2;
      out.hh = ht / 2;
    }
  }
  if (best_area > 1e299) return false;
  min_side = 2 * std::min(out.hw, out.hh);
  return true;
}

// 4 corners of a MinRect (the same 4 points cv2.boxPoints yields, set-wise).
std::array<Pt, 4> box_points(const MinRect& r) {
  Pt ux{r.ax.x * r.hw, r.ax.y * r.hw}, uy{r.ay.x * r.hh, r.ay.y * r.hh};
  return {Pt{r.c.x - ux.x - uy.x, r.c.y - ux.y - uy.y},
          Pt{r.c.x + ux.x - uy.x, r.c.y + ux.y - uy.y},
          Pt{r.c.x + ux.x + uy.x, r.c.y + ux.y + uy.y},
          Pt{r.c.x - ux.x + uy.x, r.c.y - ux.y + uy.y}};
}

// MinerU get_mini_boxes ordering: sort 4 pts by x, then resolve top/bottom by y.
std::array<Pt, 4> order_box(std::array<Pt, 4> pts) {
  std::sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) { return a.x < b.x; });
  int i1, i2, i3, i4;
  if (pts[1].y > pts[0].y) { i1 = 0; i4 = 1; } else { i1 = 1; i4 = 0; }
  if (pts[3].y > pts[2].y) { i2 = 2; i3 = 3; } else { i2 = 3; i3 = 2; }
  return {pts[i1], pts[i2], pts[i3], pts[i4]};
}

// box_score_fast: mean of pred inside the quad (scanline fill over its int bbox).
float box_score(const std::vector<float>& pred, int W, int H, const std::array<Pt, 4>& box) {
  int xmin = std::max(0, std::min(W - 1, (int)std::floor(std::min({box[0].x, box[1].x, box[2].x, box[3].x}))));
  int xmax = std::max(0, std::min(W - 1, (int)std::ceil(std::max({box[0].x, box[1].x, box[2].x, box[3].x}))));
  int ymin = std::max(0, std::min(H - 1, (int)std::floor(std::min({box[0].y, box[1].y, box[2].y, box[3].y}))));
  int ymax = std::max(0, std::min(H - 1, (int)std::ceil(std::max({box[0].y, box[1].y, box[2].y, box[3].y}))));
  int bw = xmax - xmin + 1, bh = ymax - ymin + 1;
  if (bw <= 0 || bh <= 0) return 0.f;
  // Polygon in local coords. cv2.fillPoly truncates vertices to int32, so do the same.
  std::array<Pt, 4> P;
  for (int i = 0; i < 4; ++i)
    P[i] = {(double)((int)(box[i].x) - xmin), (double)((int)(box[i].y) - ymin)};
  double sum = 0;
  long cnt = 0;
  for (int y = 0; y < bh; ++y) {
    std::vector<double> xs;
    for (int i = 0; i < 4; ++i) {
      Pt a = P[i], b = P[(i + 1) % 4];
      double y0 = a.y, y1 = b.y;
      if ((y0 <= y && y1 > y) || (y1 <= y && y0 > y)) {
        double t = (y - y0) / (y1 - y0);
        xs.push_back(a.x + t * (b.x - a.x));
      }
    }
    std::sort(xs.begin(), xs.end());
    for (size_t i = 0; i + 1 < xs.size(); i += 2) {
      int x0 = std::max(0, (int)std::ceil(xs[i] - 0.5));
      int x1 = std::min(bw - 1, (int)std::floor(xs[i + 1] - 0.5));
      for (int x = x0; x <= x1; ++x) {
        sum += pred[(size_t)(ymin + y) * W + (xmin + x)];
        ++cnt;
      }
    }
  }
  return cnt ? (float)(sum / cnt) : 0.f;
}

}  // namespace

std::vector<DetBox> db_postprocess(const std::vector<float>& pred, int H, int W,
                                   const std::array<double, 4>& shape, float thresh,
                                   float box_thresh, float unclip_ratio) {
  double src_h = shape[0], src_w = shape[1];
  // Binarize.
  std::vector<uint8_t> bin((size_t)H * W, 0);
  for (size_t i = 0; i < bin.size(); ++i) bin[i] = pred[i] > thresh ? 1 : 0;

  // 8-connected components (iterative flood fill).
  std::vector<int> label((size_t)H * W, 0);
  std::vector<DetBox> out;
  int next = 0;
  std::vector<std::pair<int, int>> stack;
  const int dx[8] = {-1, 0, 1, -1, 1, -1, 0, 1}, dy[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
  for (int y = 0; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      if (!bin[(size_t)y * W + x] || label[(size_t)y * W + x]) continue;
      ++next;
      stack.clear();
      stack.push_back({x, y});
      label[(size_t)y * W + x] = next;
      std::vector<Pt> comp;
      while (!stack.empty()) {
        auto [cx, cy] = stack.back();
        stack.pop_back();
        comp.push_back({(double)cx, (double)cy});
        for (int k = 0; k < 8; ++k) {
          int nx = cx + dx[k], ny = cy + dy[k];
          if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
          size_t ni = (size_t)ny * W + nx;
          if (bin[ni] && !label[ni]) { label[ni] = next; stack.push_back({nx, ny}); }
        }
      }
      // minAreaRect of the component, then the DB filter chain.
      MinRect r;
      double sside;
      if (!min_area_rect(comp, r, sside) || sside < 3.0) continue;
      auto box = order_box(box_points(r));
      float sc = box_score(pred, W, H, box);
      if (sc < box_thresh) continue;
      // unclip: distance = area*ratio/perimeter; expand the rect by distance/side.
      double area = (2 * r.hw) * (2 * r.hh);
      double perim = 2 * (2 * r.hw + 2 * r.hh);
      double dist = area * unclip_ratio / perim;
      MinRect e = r;
      e.hw += dist;
      e.hh += dist;
      double sside2 = 2 * std::min(e.hw, e.hh);
      if (sside2 < 5.0) continue;
      auto ebox = order_box(box_points(e));
      DetBox d;
      d.score = sc;
      for (int i = 0; i < 4; ++i) {
        int X = (int)round_half_even(ebox[i].x / W * src_w);
        int Y = (int)round_half_even(ebox[i].y / H * src_h);
        d.pts[i] = {std::max(0, std::min((int)src_w, X)), std::max(0, std::min((int)src_h, Y))};
      }
      out.push_back(d);
    }
  }
  return out;
}

}  // namespace mineru
