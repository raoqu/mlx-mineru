// Copyright (c) mlx-mineru.
// Wired-table UNet — faithful port of MinerU TSRUnet (preprocess -> unet.onnx -> line
// postprocess -> cell polygons), using the SAME OpenCV (cv2 4.13.0) as MinerU so the cv
// ops (resize / morphology / connectedComponents / minAreaRect / line) are bit-exact.
#include "mineru/wired_table.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>

#include "mnn_runner.hpp"  // hybrid MNN path (faster; exact parity)
#include "onnxruntime_cxx_api.h"

#ifdef MINERU_HAVE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace mineru {
namespace {
constexpr int kInp = 1024;
const float kMean[3] = {123.675f, 116.28f, 103.53f};
const float kStd[3] = {58.395f, 57.12f, 57.375f};
}  // namespace

struct WiredTableRecognizer::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-wired"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;  // ORT path (fallback)
  std::unique_ptr<MnnRunner> mnn;         // MNN path (preferred when a .mnn sibling exists)
  std::string in_name, out_name;
};

WiredTableRecognizer::WiredTableRecognizer() : impl_(std::make_unique<Impl>()) {}
WiredTableRecognizer::~WiredTableRecognizer() = default;
WiredTableRecognizer::WiredTableRecognizer(WiredTableRecognizer&&) noexcept = default;
WiredTableRecognizer& WiredTableRecognizer::operator=(WiredTableRecognizer&&) noexcept = default;

WiredTableRecognizer WiredTableRecognizer::load(const std::string& onnx) {
  WiredTableRecognizer r;
  Impl& m = *r.impl_;
  // Prefer a `<model>.mnn` sibling (UNet runs faster with exact parity under MNN); else ORT.
  if (onnx.size() > 5 && onnx.substr(onnx.size() - 5) == ".onnx") {
    std::string mp = onnx.substr(0, onnx.size() - 5) + ".mnn";
    std::error_code ec;
    if (std::filesystem::exists(mp, ec)) m.mnn = MnnRunner::load(mp);
  }
  if (!m.mnn) {
    m.session = std::make_unique<Ort::Session>(m.env, onnx.c_str(), m.opts);
    Ort::AllocatorWithDefaultOptions alloc;
    m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
    m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  }
  return r;
}

#ifdef MINERU_HAVE_OPENCV
namespace {
using cv::Mat;
using cv::Point2f;

// imrescale(img, (1024,1024)): keep aspect, longest side -> 1024, area for shrink else cubic.
Mat imrescale(const Mat& img) {
  int h = img.rows, w = img.cols;
  double sf = std::min((double)kInp / std::max(h, w), (double)kInp / std::min(h, w));
  // _scale_size: int(x*sf + 0.5)
  int nw = (int)(w * sf + 0.5), nh = (int)(h * sf + 0.5);
  int interp = (std::min(h, w) > kInp) ? cv::INTER_AREA : cv::INTER_CUBIC;
  Mat out;
  cv::resize(img, out, cv::Size(nw, nh), 0, 0, interp);
  return out;
}

// TSRUnet.preprocess on a BGR image -> CHW float [3,nh,nw].
std::vector<float> preprocess(const Mat& bgr, int& nh, int& nw) {
  Mat img = imrescale(bgr);
  img.convertTo(img, CV_32F);
  cv::cvtColor(img, img, cv::COLOR_BGR2RGB);  // in place
  Mat mean(1, 3, CV_32F, (void*)kMean), stdv(1, 3, CV_32F, (void*)kStd);
  cv::subtract(img, cv::Scalar(kMean[0], kMean[1], kMean[2]), img);
  cv::multiply(img, cv::Scalar(1.0f / kStd[0], 1.0f / kStd[1], 1.0f / kStd[2]), img);
  nh = img.rows;
  nw = img.cols;
  std::vector<float> chw((size_t)3 * nh * nw);
  std::vector<Mat> ch(3);
  for (int c = 0; c < 3; ++c) ch[c] = Mat(nh, nw, CV_32F, chw.data() + (size_t)c * nh * nw);
  cv::split(img, ch);
  return chw;
}

double euclid(Point2f a, Point2f b) { return std::hypot(a.x - b.x, a.y - b.y); }

// _order_points -> [tl, tr, br, bl].
std::array<Point2f, 4> order_points(std::vector<Point2f> p) {
  std::sort(p.begin(), p.end(), [](Point2f a, Point2f b) { return a.x < b.x; });
  std::array<Point2f, 2> lm{p[0], p[1]}, rm{p[2], p[3]};
  if (lm[0].y > lm[1].y) std::swap(lm[0], lm[1]);
  Point2f tl = lm[0], bl = lm[1];
  // br,tr = right_most sorted by distance to tl, descending
  if (euclid(rm[0], tl) < euclid(rm[1], tl)) std::swap(rm[0], rm[1]);
  Point2f br = rm[0], tr = rm[1];
  return {tl, tr, br, bl};
}

// min_area_rect on a line component -> [xmin,ymin,xmax,ymax] centre line (get_table_line).
std::array<double, 4> line_rect(const std::vector<cv::Point>& coords) {
  cv::RotatedRect rr = cv::minAreaRect(coords);
  Point2f bp[4];
  rr.points(bp);
  auto o = order_points({bp[0], bp[1], bp[2], bp[3]});
  double x1 = o[0].x, y1 = o[0].y, x2 = o[1].x, y2 = o[1].y;
  double x3 = o[2].x, y3 = o[2].y, x4 = o[3].x, y4 = o[3].y;
  double w = (std::hypot(x2 - x1, y2 - y1) + std::hypot(x3 - x4, y3 - y4)) / 2;
  double hh = (std::hypot(x2 - x3, y2 - y3) + std::hypot(x1 - x4, y1 - y4)) / 2;
  if (w < hh) return {(x1 + x2) / 2, (y1 + y2) / 2, (x3 + x4) / 2, (y3 + y4) / 2};
  return {(x1 + x4) / 2, (y1 + y4) / 2, (x2 + x3) / 2, (y2 + y3) / 2};
}

// get_table_line: components whose span on the axis exceeds lineW.
std::vector<std::array<double, 4>> table_lines(const Mat& bin, int axis, int lineW) {
  Mat labels, stats, cent;
  Mat mask = bin > 0;
  int n = cv::connectedComponentsWithStats(mask, labels, stats, cent, 8);
  std::vector<std::array<double, 4>> lines;
  for (int id = 1; id < n; ++id) {
    int W = stats.at<int>(id, cv::CC_STAT_WIDTH), H = stats.at<int>(id, cv::CC_STAT_HEIGHT);
    int span = (axis == 1) ? H : W;  // axis1 (cols): bbox height; axis0 (rows): bbox width
    if (span <= lineW) continue;
    std::vector<cv::Point> pts;
    cv::findNonZero(labels == id, pts);
    lines.push_back(line_rect(pts));
  }
  return lines;
}

double dpt(double x1, double y1, double x2, double y2) { return std::hypot(x1 - x2, y1 - y2); }

// adjust_lines: connect nearby collinear endpoints (more_h/v_lines).
std::vector<std::array<double, 4>> adjust_lines(const std::vector<std::array<double, 4>>& lines,
                                                double alph, double angle) {
  std::vector<std::array<double, 4>> out;
  int n = (int)lines.size();
  for (int i = 0; i < n; ++i) {
    double x1 = lines[i][0], y1 = lines[i][1], x2 = lines[i][2], y2 = lines[i][3];
    double cx1 = (x1 + x2) / 2, cy1 = (y1 + y2) / 2;
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      double x3 = lines[j][0], y3 = lines[j][1], x4 = lines[j][2], y4 = lines[j][3];
      double cx2 = (x3 + x4) / 2, cy2 = (y3 + y4) / 2;
      if ((x3 < cx1 && cx1 < x4) || (y3 < cy1 && cy1 < y4) || (x1 < cx2 && cx2 < x2) ||
          (y1 < cy2 && cy2 < y2))
        continue;
      auto chk = [&](double ax, double ay, double bx, double by) {
        double r = dpt(ax, ay, bx, by);
        double k = std::abs((by - ay) / (bx - ax + 1e-10));
        double a = std::atan(k) * 180 / M_PI;
        if (r < alph && a < angle) out.push_back({ax, ay, bx, by});
      };
      chk(x1, y1, x3, y3); chk(x1, y1, x4, y4); chk(x2, y2, x3, y3); chk(x2, y2, x4, y4);
    }
  }
  return out;
}

void fit_line(double x1, double y1, double x2, double y2, double& A, double& B, double& C) {
  A = y2 - y1; B = x1 - x2; C = x2 * y1 - x1 * y2;
}
// line_to_line: extend points1 to its intersection with points2 if an endpoint is within alpha.
std::array<double, 4> line_to_line(std::array<double, 4> p1, const std::array<double, 4>& p2,
                                   double alpha, double angle) {
  double x1 = p1[0], y1 = p1[1], x2 = p1[2], y2 = p1[3];
  double A1, B1, C1, A2, B2, C2;
  fit_line(x1, y1, x2, y2, A1, B1, C1);
  fit_line(p2[0], p2[1], p2[2], p2[3], A2, B2, C2);
  double f1 = A2 * x1 + B2 * y1 + C2, f2 = A2 * x2 + B2 * y2 + C2;
  if ((f1 > 0 && f2 > 0) || (f1 < 0 && f2 < 0)) {
    double den = A1 * B2 - A2 * B1;
    if (den != 0) {
      double x = (B1 * C2 - B2 * C1) / den, y = (A2 * C1 - A1 * C2) / den;
      double r0 = dpt(x, y, x1, y1), r1 = dpt(x, y, x2, y2);
      if (std::min(r0, r1) < alpha) {
        if (r0 < r1) {
          double k = std::abs((y2 - y) / (x2 - x + 1e-10)), a = std::atan(k) * 180 / M_PI;
          if (a < angle || std::abs(90 - a) < angle) return {x, y, x2, y2};
        } else {
          double k = std::abs((y1 - y) / (x1 - x + 1e-10)), a = std::atan(k) * 180 / M_PI;
          if (a < angle || std::abs(90 - a) < angle) return {x1, y1, x, y};
        }
      }
    }
  }
  return p1;
}
void final_adjust_lines(std::vector<std::array<double, 4>>& rows,
                        std::vector<std::array<double, 4>>& cols) {
  for (size_t i = 0; i < rows.size(); ++i)
    for (size_t j = 0; j < cols.size(); ++j) {
      rows[i] = line_to_line(rows[i], cols[j], 20, 30);
      cols[j] = line_to_line(cols[j], rows[i], 20, 30);
    }
}

// cal_region_boxes: connected components of (line_img < 255), filtered, min_area_rect.
std::vector<std::array<float, 8>> region_boxes(const Mat& line_img) {
  Mat mask = line_img < 255;
  Mat labels, stats, cent;
  int n = cv::connectedComponentsWithStats(mask, labels, stats, cent, 8);
  double W = line_img.cols, H = line_img.rows;
  std::vector<std::array<float, 8>> out;
  for (int id = 1; id < n; ++id) {
    double barea = (double)stats.at<int>(id, cv::CC_STAT_WIDTH) * stats.at<int>(id, cv::CC_STAT_HEIGHT);
    if (barea > H * W * 3 / 4) continue;
    std::vector<cv::Point> pts;
    cv::findNonZero(labels == id, pts);
    cv::RotatedRect rr = cv::minAreaRect(pts);
    Point2f bp[4];
    rr.points(bp);
    auto o = order_points({bp[0], bp[1], bp[2], bp[3]});
    double x1 = o[0].x, y1 = o[0].y, x2 = o[1].x, y2 = o[1].y, x3 = o[2].x, y3 = o[2].y,
           x4 = o[3].x, y4 = o[3].y;
    double w = (std::hypot(x2 - x1, y2 - y1) + std::hypot(x3 - x4, y3 - y4)) / 2;
    double h = (std::hypot(x2 - x3, y2 - y3) + std::hypot(x1 - x4, y1 - y4)) / 2;
    if (w * h >= 0.5 * W * H) continue;
    if (w < 15 || h < 15) continue;
    out.push_back({(float)x1, (float)y1, (float)x2, (float)y2, (float)x3, (float)y3, (float)x4, (float)y4});
  }
  return out;
}
}  // namespace
#endif  // MINERU_HAVE_OPENCV

std::vector<uint8_t> WiredTableRecognizer::segment(const std::vector<uint8_t>& rgb, int w, int h,
                                                   int& nh, int& nw) const {
  const Impl& m = *impl_;
#ifdef MINERU_HAVE_OPENCV
  cv::Mat rgbm(h, w, CV_8UC3, (void*)rgb.data()), bgr;
  cv::cvtColor(rgbm, bgr, cv::COLOR_RGB2BGR);
  std::vector<float> input = preprocess(bgr, nh, nw);
#else
  (void)rgb; (void)w; (void)h; (void)nh; (void)nw;
  std::vector<float> input;
  return {};
#endif
  if (m.mnn) {
    std::vector<int> osh;
    std::vector<float> seg = m.mnn->run(input.data(), {1, 3, nh, nw}, osh);  // int labels -> float
    if (seg.empty() || osh.size() < 2) return {};
    int oh = osh[osh.size() - 2], ow = osh[osh.size() - 1];
    std::vector<uint8_t> out((size_t)oh * ow);
    for (size_t i = 0; i < out.size() && i < seg.size(); ++i) out[i] = (uint8_t)(seg[i] + 0.5f);
    nh = oh;
    nw = ow;
    return out;
  }
  std::array<int64_t, 4> ishape{1, 3, nh, nw};
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {m.in_name.c_str()};
  const char* out_names[] = {m.out_name.c_str()};
  auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                        out_names, 1);
  auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int oh = (int)oshape[oshape.size() - 2], ow = (int)oshape[oshape.size() - 1];
  const int64_t* seg = outs[0].GetTensorData<int64_t>();
  std::vector<uint8_t> out((size_t)oh * ow);
  for (size_t i = 0; i < out.size(); ++i) out[i] = (uint8_t)seg[i];
  nh = oh;
  nw = ow;
  return out;
}

std::vector<std::array<float, 8>> WiredTableRecognizer::cell_polygons(
    const std::vector<uint8_t>& rgb, int w, int h) const {
#ifdef MINERU_HAVE_OPENCV
  int nh, nw;
  std::vector<uint8_t> seg = segment(rgb, w, h, nh, nw);
  // hpred (==1), vpred (==2); zero the other; resize to original crop size.
  cv::Mat hp(nh, nw, CV_8U), vp(nh, nw, CV_8U);
  for (int i = 0; i < nh * nw; ++i) { hp.data[i] = seg[i] == 1 ? 1 : 0; vp.data[i] = seg[i] == 2 ? 1 : 0; }
  cv::resize(hp, hp, cv::Size(w, h));
  cv::resize(vp, vp, cv::Size(w, h));
  int hors_k = (int)(std::sqrt((double)nw) * 1.2), vert_k = (int)(std::sqrt((double)nh) * 1.2);
  cv::Mat hk = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, hors_k), 1));
  cv::Mat vk = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, std::max(1, vert_k)));
  cv::morphologyEx(vp, vp, cv::MORPH_CLOSE, vk, cv::Point(-1, -1), 1);
  cv::morphologyEx(hp, hp, cv::MORPH_CLOSE, hk, cv::Point(-1, -1), 1);
  auto cols = table_lines(vp, 1, 30);
  auto rows = table_lines(hp, 0, 50);
  auto rrow = adjust_lines(rows, 100, 50);
  auto rcol = adjust_lines(cols, 15, 50);
  rows.insert(rows.end(), rrow.begin(), rrow.end());
  cols.insert(cols.end(), rcol.begin(), rcol.end());
  final_adjust_lines(rows, cols);
  cv::Mat line_img = cv::Mat::zeros(h, w, CV_8U);
  for (auto& l : rows)
    cv::line(line_img, {(int)l[0], (int)l[1]}, {(int)l[2], (int)l[3]}, 255, 2, cv::LINE_AA);
  for (auto& l : cols)
    cv::line(line_img, {(int)l[0], (int)l[1]}, {(int)l[2], (int)l[3]}, 255, 2, cv::LINE_AA);
  return region_boxes(line_img);
#else
  (void)rgb; (void)w; (void)h;
  return {};
#endif
}

// ---- Stage 3: cells -> logical grid (TableRecover) --------------------------
namespace {
// is_single_axis_contained on y axis (sorted_ocr_boxes).
bool y_contained(const std::array<double, 4>& a, const std::array<double, 4>& b, double thr) {
  double a_area = a[3] - a[1], b_area = b[3] - b[1];
  double i = std::min(a[3], b[3]) - std::max(a[1], b[1]);
  double r1 = a_area > 0 ? (a_area - i) / a_area : 0, r2 = b_area > 0 ? (b_area - i) / b_area : 0;
  return r1 < thr || r2 < thr;
}
double L2(double ax, double ay, double bx, double by) { return std::hypot(ax - bx, ay - by); }
}  // namespace

WiredTableRecognizer::Structure WiredTableRecognizer::recognize_structure(
    const std::vector<uint8_t>& rgb, int w, int h) const {
  Structure S;
  auto cells = cell_polygons(rgb, w, h);  // 8-coord [tl,tr,br,bl]
  int N = (int)cells.size();
  if (N == 0) return S;
  // TSRUnet.__call__: reshape (N,4,2) then swap corner [3]<->[1] -> [tl,bl,br,tr].
  std::vector<std::array<std::array<float, 2>, 4>> P(N);
  for (int i = 0; i < N; ++i) {
    auto& c = cells[i];
    P[i] = {{{c[0], c[1]}, {c[6], c[7]}, {c[4], c[5]}, {c[2], c[3]}}};  // tl,bl,br,tr
  }
  // sorted_ocr_boxes on box_4_2_poly_to_box_4_1 = [tl.x, tl.y, br.x, br.y].
  std::vector<std::array<double, 4>> ab(N);
  std::vector<int> idx(N);
  for (int i = 0; i < N; ++i) {
    ab[i] = {P[i][0][0], P[i][0][1], P[i][2][0], P[i][2][1]};
    idx[i] = i;
  }
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
    return ab[a][1] < ab[b][1] || (ab[a][1] == ab[b][1] && ab[a][0] < ab[b][0]);
  });
  // bubble pass (sorted_ocr_boxes): same-row, swap if out of x order.
  std::vector<std::array<double, 4>> sb(N);
  for (int i = 0; i < N; ++i) sb[i] = ab[idx[i]];
  for (int i = 0; i < N - 1; ++i)
    for (int j = i; j >= 0; --j) {
      if (y_contained(sb[j], sb[j + 1], 0.2) && sb[j + 1][0] < sb[j][0] &&
          std::abs(sb[j][1] - sb[j + 1][1]) < 20) {
        std::swap(sb[j], sb[j + 1]);
        std::swap(idx[j], idx[j + 1]);
      } else break;
    }
  std::vector<std::array<std::array<float, 2>, 4>> poly(N);
  for (int i = 0; i < N; ++i) poly[i] = P[idx[i]];

  // --- TableRecover ---
  // get_rows: cluster by tl.y diff > rows_thresh(10).
  std::vector<std::vector<int>> rows;
  {
    std::vector<int> cur{0};
    for (int i = 1; i < N; ++i) {
      if (std::abs(poly[i][0][1] - poly[i - 1][0][1]) > 10) { rows.push_back(cur); cur.clear(); }
      cur.push_back(i);
    }
    rows.push_back(cur);
  }
  // get_benchmark_cols.
  int longest = 0;
  for (int r = 1; r < (int)rows.size(); ++r) if (rows[r].size() > rows[longest].size()) longest = r;
  std::vector<double> colx;
  for (int i : rows[longest]) colx.push_back(poly[i][0][0]);
  double min_x = colx.front(), max_x = poly[rows[longest].back()][2][0];
  const double cthr = 15;
  auto update_col = [&](double cur, bool insert_last) {
    for (size_t i = 0; i < colx.size(); ++i) {
      if (cur - cthr <= colx[i] && colx[i] <= cur + cthr) return;
      if (cur < min_x) { colx.insert(colx.begin(), cur); min_x = cur; return; }
      if (cur > max_x) { if (insert_last) colx.push_back(cur); max_x = cur; return; }
      if (cur < colx[i]) { colx.insert(colx.begin() + i, cur); return; }
    }
  };
  for (auto& row : rows)
    for (int i : row) { update_col(poly[i][0][0], true); update_col(poly[i][2][0], false); }
  int col_nums = (int)colx.size();
  std::vector<double> col_w;
  for (int i = 1; i < col_nums; ++i) col_w.push_back(colx[i] - colx[i - 1]);
  col_w.push_back(max_x - colx.back());
  // get_benchmark_rows.
  std::vector<double> bench_y;
  for (auto& row : rows) bench_y.push_back(poly[row[0]][0][1]);
  std::vector<double> row_h;
  for (size_t i = 1; i < bench_y.size(); ++i) row_h.push_back(bench_y[i] - bench_y[i - 1]);
  double max_h = 0;
  for (int i : rows.back()) max_h = std::max(max_h, L2(poly[i][1][0], poly[i][1][1], poly[i][0][0], poly[i][0][1]));
  row_h.push_back(max_h);
  int row_nums = (int)bench_y.size();
  // get_merge_cells -> logic_points.
  std::vector<std::array<int, 4>> logic(N);
  const double mthr = 10;
  for (int cur_row = 0; cur_row < (int)rows.size(); ++cur_row) {
    std::vector<int> col_acc;  // accumulated col span in this row
    int running = 0;
    for (int one : rows[cur_row]) {
      auto& box = poly[one];
      double bw = L2(box[3][0], box[3][1], box[0][0], box[0][1]);
      int loc = 0;
      double bd = 1e300;
      for (int i = 0; i < col_nums; ++i) { double d = std::abs(colx[i] - box[0][0]); if (d < bd) { bd = d; loc = i; } }
      int col_start = std::max(running, loc);
      int span_c = col_nums - col_start;
      for (int i = col_start; i < col_nums; ++i) {
        double cum = 0; for (int k = col_start; k <= i; ++k) cum += col_w[k];
        if (i == col_start && cum > bw) { span_c = 1; break; }
        if (std::abs(cum - bw) <= mthr) { span_c = i + 1 - col_start; break; }
        if (cum > bw) { int id = (std::abs(cum - bw) < std::abs(cum - col_w[i] - bw)) ? i : i - 1; span_c = id + 1 - col_start; break; }
      }
      int col_end = span_c + col_start - 1;
      running += span_c;
      double bh = L2(box[1][0], box[1][1], box[0][0], box[0][1]);
      int row_start = cur_row, span_r = row_nums - row_start;
      for (int j = row_start; j < row_nums; ++j) {
        double cum = 0; for (int k = row_start; k <= j; ++k) cum += row_h[k];
        if (j == row_start && cum > bh) { span_r = 1; break; }
        if (std::abs(bh - cum) <= mthr) { span_r = j + 1 - row_start; break; }
        if (cum > bh) { int id = (std::abs(cum - bh) < std::abs(cum - row_h[j] - bh)) ? j : j - 1; span_r = id + 1 - row_start; break; }
      }
      int row_end = span_r + row_start - 1;
      logic[one] = {row_start, row_end, col_start, col_end};
    }
  }
  for (int i = 0; i < N; ++i) {
    S.polygons.push_back({poly[i][0][0], poly[i][0][1], poly[i][1][0], poly[i][1][1],
                          poly[i][2][0], poly[i][2][1], poly[i][3][0], poly[i][3][1]});
    S.logic.push_back(logic[i]);
  }
  return S;
}

// ---- Stage 4: logical grid -> HTML (plot_html_table) ------------------------
namespace {
double median(std::vector<double> v) {
  if (v.empty()) return -1;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2;
}
}  // namespace

std::string WiredTableRecognizer::plot_html(const Structure& s,
                                            const std::vector<std::string>& cell_text) {
  int N = (int)s.logic.size();
  if (N == 0) return "<html><body><table></table></body></html>";
  auto txt = [&](int i) -> std::string { return i < (int)cell_text.size() ? cell_text[i] : ""; };
  auto has_text = [&](int i) {
    std::string t = txt(i);
    return t.find_first_not_of(" \t\r\n") != std::string::npos;
  };
  int max_row = 0, max_col = 0;
  for (auto& l : s.logic) { max_row = std::max(max_row, l[1] + 1); max_col = std::max(max_col, l[3] + 1); }
  // grid[r][c] = cell index (or -1).
  std::vector<std::vector<int>> grid(max_row, std::vector<int>(max_col, -1));
  for (int i = 0; i < N; ++i) {
    auto& l = s.logic[i];
    for (int r = l[0]; r <= l[1]; ++r)
      for (int c = l[2]; c <= l[3]; ++c) grid[r][c] = i;
  }
  // estimate_axis_sizes: per row/col median cell size (from cell AABB / span).
  std::vector<std::vector<double>> rs(max_row), cs(max_col);
  for (int i = 0; i < N; ++i) {
    auto& p = s.polygons[i];
    double x0 = std::min({p[0], p[2], p[4], p[6]}), y0 = std::min({p[1], p[3], p[5], p[7]});
    double x1 = std::max({p[0], p[2], p[4], p[6]}), y1 = std::max({p[1], p[3], p[5], p[7]});
    auto& l = s.logic[i];
    double cspan = std::max(l[3] - l[2] + 1, 1), rspan = std::max(l[1] - l[0] + 1, 1);
    double csz = std::max((x1 - x0) / cspan, 0.0), rsz = std::max((y1 - y0) / rspan, 0.0);
    if (csz > 0) for (int c = l[2]; c <= l[3]; ++c) if (c >= 0 && c < max_col) cs[c].push_back(csz);
    if (rsz > 0) for (int r = l[0]; r <= l[1]; ++r) if (r >= 0 && r < max_row) rs[r].push_back(rsz);
  }
  std::vector<double> row_sz(max_row), col_sz(max_col);
  for (int r = 0; r < max_row; ++r) row_sz[r] = median(rs[r]);
  for (int c = 0; c < max_col; ++c) col_sz[c] = median(cs[c]);
  // trim_noise_edges.
  int r0 = 0, r1 = max_row - 1, c0 = 0, c1 = max_col - 1;
  auto ref_size = [](const std::vector<double>& sz, int idx) {
    std::vector<double> o;
    for (int i = 0; i < (int)sz.size(); ++i) if (i != idx && sz[i] > 0) o.push_back(sz[i]);
    return median(o);
  };
  auto abnormal = [&](const std::vector<double>& sz, int idx) {
    double a = sz[idx], ref = ref_size(sz, idx);
    if (a <= 0 || ref <= 0) return false;
    double ratio = a / ref;
    return ratio < 0.35 || ratio > 2.5;
  };
  auto is_noise = [&](bool is_col, int idx) {
    // edge_axis_has_text
    if (is_col) { for (int r = r0; r <= r1; ++r) if (grid[r][idx] >= 0 && has_text(grid[r][idx])) return false; }
    else { for (int c = c0; c <= c1; ++c) if (grid[idx][c] >= 0 && has_text(grid[idx][c])) return false; }
    int covered = 0, total;
    if (is_col) { for (int r = r0; r <= r1; ++r) covered += grid[r][idx] >= 0; total = r1 - r0 + 1; }
    else { for (int c = c0; c <= c1; ++c) covered += grid[idx][c] >= 0; total = c1 - c0 + 1; }
    if (covered == 0 || covered < total) return true;
    return abnormal(is_col ? col_sz : row_sz, idx);
  };
  while (r0 <= r1 && is_noise(false, r0)) ++r0;
  while (r1 >= r0 && is_noise(false, r1)) --r1;
  while (c0 <= c1 && is_noise(true, c0)) ++c0;
  while (c1 >= c0 && is_noise(true, c1)) --c1;

  std::string html = "<html><body><table>";
  if (r0 > r1 || c0 > c1) return html + "</table></body></html>";
  for (int row = r0; row <= r1; ++row) {
    std::string tr = "<tr>";
    for (int col = c0; col <= c1; ++col) {
      int ci = grid[row][col];
      if (ci < 0) { tr += "<td></td>"; continue; }
      auto& l = s.logic[ci];
      int crs = std::max(l[0], r0), ccs = std::max(l[2], c0);
      if (row == crs && col == ccs) {
        int rspan = std::min(l[1], r1) - crs + 1, cspan = std::min(l[3], c1) - ccs + 1;
        tr += "<td rowspan=" + std::to_string(rspan) + " colspan=" + std::to_string(cspan) + ">" +
              txt(ci) + "</td>";
      }
    }
    html += tr + "</tr>";
  }
  return html + "</table></body></html>";
}

// ---- Stage 5: OCR-cell matching -> text-bearing HTML ------------------------
namespace {
struct OcrBox { double x0, y0, x1, y1; std::string text; };
}  // namespace

std::string WiredTableRecognizer::recognize_html(const std::vector<uint8_t>& rgb, int w, int h,
                                                 const std::vector<TableOcrItem>& ocr) const {
  Structure st = recognize_structure(rgb, w, h);
  int N = (int)st.polygons.size();
  if (N == 0) return "<html><body><table></table></body></html>";
  // pred cell AABBs.
  std::vector<std::array<double, 4>> pred(N);
  for (int j = 0; j < N; ++j) {
    auto& p = st.polygons[j];
    pred[j] = {std::min({p[0], p[2], p[4], p[6]}), std::min({p[1], p[3], p[5], p[7]}),
               std::max({p[0], p[2], p[4], p[6]}), std::max({p[1], p[3], p[5], p[7]})};
  }
  // match_ocr_cell (common case): each OCR box -> best cell by contained / iou.
  std::vector<std::vector<OcrBox>> cellbox(N);
  for (const auto& o : ocr) {
    double ox0 = o.box[0][0], oy0 = o.box[0][1], ox1 = o.box[2][0], oy1 = o.box[2][1];
    double oa = (ox1 - ox0) * (oy1 - oy0);
    std::vector<int> cand;
    std::vector<double> cov(N, 0), iouv(N, 0);
    for (int j = 0; j < N; ++j) {
      auto& p = pred[j];
      double ix = std::max(0.0, std::min(ox1, p[2]) - std::max(ox0, p[0]));
      double iy = std::max(0.0, std::min(oy1, p[3]) - std::max(oy0, p[1]));
      double ia = ix * iy;
      bool inter = !(ox1 < p[0] || ox0 > p[2] || oy1 < p[1] || oy0 > p[3]);
      double pa = (p[2] - p[0]) * (p[3] - p[1]);
      double outside = oa > 0 ? (oa - ia) / oa : 0;
      double uni = oa + pa - ia, iou = (uni != 0) ? ia / uni : 1.0;
      if (!inter) iou = 0;
      cov[j] = oa > 0 ? ia / oa : 0;
      iouv[j] = iou;
      if (inter && (outside < 0.6 || iou > 0.8)) cand.push_back(j);
    }
    int best = -1;
    if (cand.size() == 1) best = cand[0];
    else if (cand.size() > 1) {
      std::sort(cand.begin(), cand.end(),
                [&](int a, int b) { return cov[a] != cov[b] ? cov[a] > cov[b] : iouv[a] > iouv[b]; });
      int b0 = cand[0], b1 = cand[1];
      double cx = (ox0 + ox1) / 2, cy = (oy0 + oy1) / 2;
      std::vector<int> hits;
      for (int j : cand)
        if (pred[j][0] <= cx && cx < pred[j][2] && pred[j][1] <= cy && cy <= pred[j][3]) hits.push_back(j);
      if (hits.size() == 1 && hits[0] == b0 && cov[b0] >= 0.55 && cov[b0] - cov[b1] >= 0.15) best = b0;
      else if (cov[b0] >= 0.65 && cov[b0] - cov[b1] >= 0.2) best = b0;
    }
    if (best >= 0) cellbox[best].push_back({ox0, oy0, ox1, oy1, o.text});
  }
  // sort_and_gather_ocr_res: per cell, sort top-to-bottom/left-to-right, gather same-row.
  std::vector<std::string> cell_text(N);
  for (int j = 0; j < N; ++j) {
    auto& boxes = cellbox[j];
    if (boxes.empty()) continue;
    std::stable_sort(boxes.begin(), boxes.end(), [](const OcrBox& a, const OcrBox& b) {
      return a.y0 < b.y0 || (a.y0 == b.y0 && a.x0 < b.x0);
    });
    // gather_ocr_list_by_row: merge same-row (y-contained, thr=10) with gap spaces.
    std::vector<int> alive(boxes.size());
    for (size_t k = 0; k < boxes.size(); ++k) alive[k] = 1;
    for (size_t a = 0; a < boxes.size(); ++a) {
      if (!alive[a]) continue;
      for (size_t b = a + 1; b < boxes.size(); ++b) {
        if (!alive[b]) continue;
        double ha = boxes[a].y1 - boxes[a].y0, hb = boxes[b].y1 - boxes[b].y0;
        double iy = std::min(boxes[a].y1, boxes[b].y1) - std::max(boxes[a].y0, boxes[b].y0);
        double ra = ha > 0 ? (ha - iy) / ha : 0, rb = hb > 0 ? (hb - iy) / hb : 0;
        if (ra < 10 || rb < 10) {  // is_single_axis_contained y, thr=10 (ratio<thr)
          double dis = std::max(boxes[b].x0 - boxes[a].x1, 0.0);
          boxes[a].text += std::string((int)(dis / 10), ' ') + boxes[b].text;
          boxes[a].x0 = std::min(boxes[a].x0, boxes[b].x0);
          boxes[a].x1 = std::max(boxes[a].x1, boxes[b].x1);
          boxes[a].y0 = std::min(boxes[a].y0, boxes[b].y0);
          boxes[a].y1 = std::max(boxes[a].y1, boxes[b].y1);
          alive[b] = 0;
        }
      }
    }
    std::string t;
    for (size_t k = 0; k < boxes.size(); ++k) if (alive[k]) t += boxes[k].text;
    cell_text[j] = t;
  }
  return plot_html(st, cell_text);
}

}  // namespace mineru
