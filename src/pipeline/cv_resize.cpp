// Copyright (c) mlx-mineru.
#include "mineru/cv_resize.hpp"

#include "mineru/image_preprocess.hpp"  // resize_bilinear_rgb8 / resize_bicubic_rgb8 fallback

#ifdef MINERU_HAVE_OPENCV
#include <opencv2/imgproc.hpp>
#endif

namespace mineru {

bool have_opencv_resize() {
#ifdef MINERU_HAVE_OPENCV
  return true;
#else
  return false;
#endif
}

std::vector<uint8_t> resize_rgb8_cv(const std::vector<uint8_t>& rgb, int in_w, int in_h, int out_w,
                                    int out_h, int interp) {
#ifdef MINERU_HAVE_OPENCV
  cv::Mat src(in_h, in_w, CV_8UC3, const_cast<uint8_t*>(rgb.data()));
  cv::Mat dst;
  cv::resize(src, dst, cv::Size(out_w, out_h), 0, 0, interp);
  if (!dst.isContinuous()) dst = dst.clone();
  return std::vector<uint8_t>(dst.data, dst.data + static_cast<size_t>(out_w) * out_h * 3);
#else
  // No OpenCV: high-quality float fallback (≈cv2 within ~1 LSB).
  return (interp == kInterCubic) ? resize_bicubic_rgb8(rgb, in_w, in_h, out_w, out_h)
                                 : resize_bilinear_rgb8(rgb, in_w, in_h, out_w, out_h);
#endif
}

}  // namespace mineru
