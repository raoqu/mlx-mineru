// Copyright (c) mlx-mineru.
// Faithful port of mineru pytorchocr DBPostProcess (box_type="quad") + the
// DetResizeForTest/Normalize preprocess and the ocr_det.onnx wrapper.
#include "mineru/ocr_det.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "mineru/cv_resize.hpp"  // resize_rgb8_cv (real cv2.resize)
#include "onnxruntime_cxx_api.h"
#ifdef MINERU_HAVE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace mineru {
namespace {

struct Pt { double x, y; };

// numpy np.round: round half to even (banker's rounding).
inline long round_half_even(double v) { return (long)std::nearbyint(v); }

#ifdef MINERU_HAVE_OPENCV
// get_mini_boxes: cv2.minAreaRect -> boxPoints -> sort by x, reorder by y.
bool get_mini_boxes(const std::vector<cv::Point>& contour, std::array<cv::Point2f, 4>& box,
                    double& sside) {
  cv::RotatedRect rr = cv::minAreaRect(contour);
  cv::Point2f bp[4];
  rr.points(bp);
  std::vector<cv::Point2f> p(bp, bp + 4);
  std::sort(p.begin(), p.end(), [](cv::Point2f a, cv::Point2f b) { return a.x < b.x; });
  int i1, i2, i3, i4;
  if (p[1].y > p[0].y) { i1 = 0; i4 = 1; } else { i1 = 1; i4 = 0; }
  if (p[3].y > p[2].y) { i2 = 2; i3 = 3; } else { i2 = 3; i3 = 2; }
  box = {p[i1], p[i2], p[i3], p[i4]};
  sside = std::min(rr.size.width, rr.size.height);
  return true;
}

// box_score_fast: cv2.fillPoly mask + cv2.mean over the box bbox.
float box_score(const cv::Mat& pred, const std::array<cv::Point2f, 4>& box) {
  int W = pred.cols, H = pred.rows;
  float xs[4] = {box[0].x, box[1].x, box[2].x, box[3].x};
  float ys[4] = {box[0].y, box[1].y, box[2].y, box[3].y};
  int xmin = std::max(0, std::min(W - 1, (int)std::floor(*std::min_element(xs, xs + 4))));
  int xmax = std::max(0, std::min(W - 1, (int)std::ceil(*std::max_element(xs, xs + 4))));
  int ymin = std::max(0, std::min(H - 1, (int)std::floor(*std::min_element(ys, ys + 4))));
  int ymax = std::max(0, std::min(H - 1, (int)std::ceil(*std::max_element(ys, ys + 4))));
  int bw = xmax - xmin + 1, bh = ymax - ymin + 1;
  if (bw <= 0 || bh <= 0) return 0.f;
  cv::Mat mask = cv::Mat::zeros(bh, bw, CV_8U);
  std::vector<cv::Point> poly;
  for (int i = 0; i < 4; ++i) poly.push_back({(int)box[i].x - xmin, (int)box[i].y - ymin});
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(1));
  cv::Mat roi = pred(cv::Rect(xmin, ymin, bw, bh));
  return (float)cv::mean(roi, mask)[0];
}
#endif
}  // namespace

std::vector<DetBox> db_postprocess(const std::vector<float>& pred, int H, int W,
                                   const std::array<double, 4>& shape, float thresh,
                                   float box_thresh, float unclip_ratio) {
  double src_h = shape[0], src_w = shape[1];
  std::vector<DetBox> out;
#ifdef MINERU_HAVE_OPENCV
  // Faithful DBPostProcess via real OpenCV (cv2.findContours/minAreaRect/boxPoints/fillPoly).
  cv::Mat predm(H, W, CV_32F, const_cast<float*>(pred.data()));
  cv::Mat bitmap(H, W, CV_8U);
  for (int i = 0; i < H * W; ++i) bitmap.data[i] = pred[i] > thresh ? 255 : 0;
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(bitmap, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
  for (auto& c : contours) {
    std::array<cv::Point2f, 4> box;
    double sside;
    get_mini_boxes(c, box, sside);
    if (sside < 3.0) continue;
    float sc = box_score(predm, box);
    if (sc < box_thresh) continue;
    // unclip: distance = poly.area*ratio/poly.length; expand the rect (pyclipper round-offset
    // of a rectangle == minAreaRect grown by distance on each side).
    double area = 0, length = 0;
    for (int i = 0; i < 4; ++i) {
      int j = (i + 1) % 4;
      area += (double)box[i].x * box[j].y - (double)box[j].x * box[i].y;
      length += std::hypot(box[i].x - box[j].x, box[i].y - box[j].y);
    }
    area = std::abs(area) / 2;
    if (length <= 0) continue;
    double dist = area * unclip_ratio / length;
    std::vector<cv::Point2f> bv(box.begin(), box.end());
    cv::RotatedRect rr = cv::minAreaRect(bv);
    rr.size.width += (float)(2 * dist);
    rr.size.height += (float)(2 * dist);
    cv::Point2f ep[4];
    rr.points(ep);
    std::vector<cv::Point> ec;
    for (auto& p : ep) ec.push_back({(int)std::lround(p.x), (int)std::lround(p.y)});
    std::array<cv::Point2f, 4> ebox;
    double sside2;
    get_mini_boxes(ec, ebox, sside2);
    if (sside2 < 5.0) continue;
    DetBox d;
    d.score = sc;
    for (int i = 0; i < 4; ++i) {
      int X = (int)round_half_even(ebox[i].x / W * src_w);
      int Y = (int)round_half_even(ebox[i].y / H * src_h);
      d.pts[i] = {std::max(0, std::min((int)src_w, X)), std::max(0, std::min((int)src_h, Y))};
    }
    out.push_back(d);
  }
#else
  (void)pred; (void)H; (void)W; (void)thresh; (void)box_thresh; (void)unclip_ratio;
#endif
  return out;
}

// ---- end-to-end detector ----------------------------------------------------

struct TextDetector::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-det"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::string in_name, out_name;
};

TextDetector::TextDetector() : impl_(std::make_unique<Impl>()) {}
TextDetector::~TextDetector() = default;
TextDetector::TextDetector(TextDetector&&) noexcept = default;
TextDetector& TextDetector::operator=(TextDetector&&) noexcept = default;

TextDetector TextDetector::load(const std::string& onnx_path) {
  TextDetector d;
  Impl& m = *d.impl_;
  m.session = std::make_unique<Ort::Session>(m.env, onnx_path.c_str(), m.opts);
  Ort::AllocatorWithDefaultOptions alloc;
  m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
  m.out_name = m.session->GetOutputNameAllocated(0, alloc).get();
  return d;
}

std::vector<DetBox> TextDetector::detect(const std::vector<uint8_t>& rgb, int w, int h) const {
  const Impl& m = *impl_;
  // DetResizeForTest (limit_type="max", limit_side_len=960, max_side_limit=4000).
  constexpr int kLimit = 960, kMaxSide = 4000;
  double ratio = std::max(h, w) > kLimit ? (double)kLimit / std::max(h, w) : 1.0;
  int rh = (int)(h * ratio), rw = (int)(w * ratio);
  if (std::max(rh, rw) > kMaxSide) {
    double r2 = (double)kMaxSide / std::max(rh, rw);
    rh = (int)(rh * r2);
    rw = (int)(rw * r2);
  }
  rh = std::max((int)(round_half_even(rh / 32.0) * 32), 32);
  rw = std::max((int)(round_half_even(rw / 32.0) * 32), 32);

  std::vector<uint8_t> resized = resize_rgb8_cv(rgb, w, h, rw, rh, kInterLinear);
  // MinerU feeds the model BGR (cv2.cvtColor(RGB, COLOR_RGB2BGR)); the ImageNet means
  // are applied in that channel order. NormalizeImage (order hwc): (px/255 - mean)/std,
  // then ToCHWImage. Source is RGB, so model channel c reads source channel (2 - c).
  static const float kMean[3] = {0.485f, 0.456f, 0.406f};
  static const float kStd[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> input(static_cast<size_t>(3) * rh * rw);
  for (int y = 0; y < rh; ++y)
    for (int x = 0; x < rw; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * rh + y) * rw + x] =
            (resized[(static_cast<size_t>(y) * rw + x) * 3 + (2 - c)] / 255.0f - kMean[c]) /
            kStd[c];

  std::array<int64_t, 4> ishape{1, 3, rh, rw};
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {m.in_name.c_str()};
  const char* out_names[] = {m.out_name.c_str()};
  auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                        out_names, 1);
  auto oshape = outs[0].GetTensorTypeAndShapeInfo().GetShape();
  int Hd = (int)oshape[2], Wd = (int)oshape[3];
  const float* p = outs[0].GetTensorData<float>();
  std::vector<float> prob(p, p + static_cast<size_t>(Hd) * Wd);

  std::array<double, 4> shape{(double)h, (double)w, (double)rh / h, (double)rw / w};
  return db_postprocess(prob, Hd, Wd, shape);
}

}  // namespace mineru
