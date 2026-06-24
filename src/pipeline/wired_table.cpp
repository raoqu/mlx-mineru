// Copyright (c) mlx-mineru.
// Wired-table UNet — faithful port of MinerU TSRUnet (preprocess -> unet.onnx -> line
// postprocess -> cell polygons), using the SAME OpenCV (cv2 4.13.0) as MinerU so the cv
// ops (resize / morphology / connectedComponents / minAreaRect / line) are bit-exact.
#include "mineru/wired_table.hpp"

#include <algorithm>
#include <cmath>

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

}  // namespace mineru
