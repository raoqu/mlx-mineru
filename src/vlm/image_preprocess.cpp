// Copyright (c) mlx-mineru.
#include "mineru/image_preprocess.hpp"

#include <cmath>
#include <stdexcept>

namespace mineru {
namespace {

// Python round(): banker's rounding (half to even).
long py_round(double x) {
  double r = std::nearbyint(x);  // honors current rounding mode (default: to-even)
  return static_cast<long>(r);
}

// --- PIL bicubic resampling (faithful to Resample.c, 8-bit path) -------------
constexpr int kPrecisionBits = 32 - 8 - 2;  // PIL PRECISION_BITS

double bicubic_filter(double x) {
  const double a = -0.5;
  if (x < 0) x = -x;
  if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
  if (x < 2.0) return (((x - 5.0) * x + 8.0) * x - 4.0) * a;
  return 0.0;
}

int clip8(long long in) {
  long long v = in >> kPrecisionBits;
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<int>(v);
}

// Precompute integer coefficients for one axis. Returns ksize; fills bounds
// (xmin,xmax per output) and kk (outSize*ksize ints).
int precompute_coeffs(int in_size, int out_size, std::vector<int>& bounds,
                      std::vector<int>& kk) {
  const double support = 2.0;
  double scale = static_cast<double>(in_size) / out_size;
  double filterscale = scale < 1.0 ? 1.0 : scale;
  double sup = support * filterscale;
  int ksize = static_cast<int>(std::ceil(sup)) * 2 + 1;

  bounds.assign(static_cast<size_t>(out_size) * 2, 0);
  std::vector<double> k(ksize);
  kk.assign(static_cast<size_t>(out_size) * ksize, 0);

  for (int xx = 0; xx < out_size; ++xx) {
    double center = (xx + 0.5) * scale;
    double ww = 0.0;
    double ss = 1.0 / filterscale;
    int xmin = static_cast<int>(center - sup + 0.5);
    if (xmin < 0) xmin = 0;
    int xmax = static_cast<int>(center + sup + 0.5);
    if (xmax > in_size) xmax = in_size;
    xmax -= xmin;
    for (int x = 0; x < xmax; ++x) {
      double w = bicubic_filter((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (int x = 0; x < xmax; ++x)
      if (ww != 0.0) k[x] /= ww;
    // to fixed point
    for (int x = 0; x < xmax; ++x) {
      double v = k[x] * (1 << kPrecisionBits);
      kk[static_cast<size_t>(xx) * ksize + x] =
          static_cast<int>(v < 0 ? v - 0.5 : v + 0.5);  // round half away from zero
    }
    for (int x = xmax; x < ksize; ++x) kk[static_cast<size_t>(xx) * ksize + x] = 0;
    bounds[static_cast<size_t>(xx) * 2 + 0] = xmin;
    bounds[static_cast<size_t>(xx) * 2 + 1] = xmax;
  }
  return ksize;
}

}  // namespace

std::vector<uint8_t> resize_bicubic_rgb8(const std::vector<uint8_t>& rgb, int in_w, int in_h,
                                         int out_w, int out_h) {
  // Horizontal pass: (in_h x in_w) -> (in_h x out_w), then vertical -> (out_h x out_w).
  std::vector<int> hbounds, hk;
  int hksize = precompute_coeffs(in_w, out_w, hbounds, hk);
  std::vector<uint8_t> tmp(static_cast<size_t>(in_h) * out_w * 3);
  const long long half = 1LL << (kPrecisionBits - 1);

  for (int y = 0; y < in_h; ++y) {
    const uint8_t* row = rgb.data() + static_cast<size_t>(y) * in_w * 3;
    uint8_t* orow = tmp.data() + static_cast<size_t>(y) * out_w * 3;
    for (int xx = 0; xx < out_w; ++xx) {
      int xmin = hbounds[xx * 2 + 0];
      int xmax = hbounds[xx * 2 + 1];
      const int* k = &hk[static_cast<size_t>(xx) * hksize];
      long long r = half, g = half, b = half;
      for (int x = 0; x < xmax; ++x) {
        const uint8_t* px = row + static_cast<size_t>(xmin + x) * 3;
        r += static_cast<long long>(px[0]) * k[x];
        g += static_cast<long long>(px[1]) * k[x];
        b += static_cast<long long>(px[2]) * k[x];
      }
      uint8_t* op = orow + static_cast<size_t>(xx) * 3;
      op[0] = static_cast<uint8_t>(clip8(r));
      op[1] = static_cast<uint8_t>(clip8(g));
      op[2] = static_cast<uint8_t>(clip8(b));
    }
  }

  std::vector<int> vbounds, vk;
  int vksize = precompute_coeffs(in_h, out_h, vbounds, vk);
  std::vector<uint8_t> out(static_cast<size_t>(out_h) * out_w * 3);
  for (int yy = 0; yy < out_h; ++yy) {
    int ymin = vbounds[yy * 2 + 0];
    int ymax = vbounds[yy * 2 + 1];
    const int* k = &vk[static_cast<size_t>(yy) * vksize];
    uint8_t* orow = out.data() + static_cast<size_t>(yy) * out_w * 3;
    for (int xx = 0; xx < out_w; ++xx) {
      long long r = half, g = half, b = half;
      for (int y = 0; y < ymax; ++y) {
        const uint8_t* px = tmp.data() + (static_cast<size_t>(ymin + y) * out_w + xx) * 3;
        r += static_cast<long long>(px[0]) * k[y];
        g += static_cast<long long>(px[1]) * k[y];
        b += static_cast<long long>(px[2]) * k[y];
      }
      uint8_t* op = orow + static_cast<size_t>(xx) * 3;
      op[0] = static_cast<uint8_t>(clip8(r));
      op[1] = static_cast<uint8_t>(clip8(g));
      op[2] = static_cast<uint8_t>(clip8(b));
    }
  }
  return out;
}

std::array<int, 2> smart_resize(int height, int width, int factor, int min_pixels, int max_pixels) {
  if (std::max(height, width) / static_cast<double>(std::min(height, width)) > 200.0)
    throw std::runtime_error("smart_resize: aspect ratio must be < 200");
  long h_bar = py_round(static_cast<double>(height) / factor) * factor;
  long w_bar = py_round(static_cast<double>(width) / factor) * factor;
  if (h_bar * w_bar > max_pixels) {
    double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
    h_bar = std::max<long>(factor, static_cast<long>(std::floor(height / beta / factor)) * factor);
    w_bar = std::max<long>(factor, static_cast<long>(std::floor(width / beta / factor)) * factor);
  } else if (h_bar * w_bar < min_pixels) {
    double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
    h_bar = static_cast<long>(std::ceil(height * beta / factor)) * factor;
    w_bar = static_cast<long>(std::ceil(width * beta / factor)) * factor;
  }
  return {static_cast<int>(h_bar), static_cast<int>(w_bar)};
}

VisionInput preprocess_image(const std::vector<uint8_t>& rgb, int width, int height,
                             const Qwen2VLImageConfig& cfg) {
  int factor = cfg.patch_size * cfg.merge_size;  // 28
  auto hw = smart_resize(height, width, factor, cfg.min_pixels, cfg.max_pixels);
  int rh = hw[0], rw = hw[1];

  std::vector<uint8_t> resized = resize_bicubic_rgb8(rgb, width, height, rw, rh);

  // rescale (1/255) + normalize -> CHW float [3, rh, rw]
  const int C = 3;
  std::vector<float> chw(static_cast<size_t>(C) * rh * rw);
  for (int y = 0; y < rh; ++y) {
    for (int x = 0; x < rw; ++x) {
      const uint8_t* px = resized.data() + (static_cast<size_t>(y) * rw + x) * 3;
      for (int c = 0; c < C; ++c) {
        float v = (px[c] / 255.0f - cfg.image_mean[c]) / cfg.image_std[c];
        chw[(static_cast<size_t>(c) * rh + y) * rw + x] = v;
      }
    }
  }

  // Patchify per transformers _preprocess. Single image: temporal repeat x2.
  const int ps = cfg.patch_size, ts = cfg.temporal_patch_size, ms = cfg.merge_size;
  int grid_t = 1;
  int grid_h = rh / ps, grid_w = rw / ps;
  VisionInput out;
  out.grid_thw = {grid_t, grid_h, grid_w};
  int feat = C * ts * ps * ps;  // 1176
  out.pixel_values.assign(static_cast<size_t>(grid_t) * grid_h * grid_w * feat, 0.0f);

  // Index identity of the transpose (0,3,6,4,7,2,1,5,8) over dims
  //   [grid_t, ts, C, grid_h/ms, ms, ps, grid_w/ms, ms, ps].
  // Output row index r = ((gt*grid_h + hh)*grid_w + ww) where the merge layout is
  // folded so consecutive rows are merge-blocks; output col index over (C,ts,ps,ps).
  auto chw_at = [&](int c, int yy, int xx) -> float {
    return chw[(static_cast<size_t>(c) * rh + yy) * rw + xx];
  };

  int Hm = grid_h / ms, Wm = grid_w / ms;
  size_t row = 0;
  for (int gt = 0; gt < grid_t; ++gt) {
    for (int hm = 0; hm < Hm; ++hm) {
      for (int wm = 0; wm < Wm; ++wm) {
        for (int mh = 0; mh < ms; ++mh) {
          for (int mw = 0; mw < ms; ++mw) {
            // one output patch-row: dims (C, ts, ps, ps)
            int patch_h = hm * ms + mh;  // grid row
            int patch_w = wm * ms + mw;  // grid col
            float* dst = out.pixel_values.data() + row * feat;
            size_t col = 0;
            for (int c = 0; c < C; ++c) {
              for (int t = 0; t < ts; ++t) {  // temporal copies identical for image
                for (int py = 0; py < ps; ++py) {
                  for (int px = 0; px < ps; ++px) {
                    int yy = patch_h * ps + py;
                    int xx = patch_w * ps + px;
                    dst[col++] = chw_at(c, yy, xx);
                  }
                }
              }
            }
            ++row;
          }
        }
      }
    }
  }
  return out;
}

}  // namespace mineru
