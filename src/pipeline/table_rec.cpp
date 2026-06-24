// Copyright (c) mlx-mineru.
// SLANet+ table structure recognition + OCR->cell matching, faithful to MinerU
// slanet_plus (TableStructurer / TableLabelDecode / adapt_slanet_plus / TableMatch).
#include "mineru/table_rec.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>

#include "mineru/cv_resize.hpp"  // resize_rgb8_cv (real cv2.resize)
#include "onnxruntime_cxx_api.h"

namespace mineru {
namespace {
constexpr int kMaxLen = 488;
const std::array<float, 3> kMean{0.485f, 0.456f, 0.406f};
const std::array<float, 3> kStd{0.229f, 0.224f, 0.225f};

bool is_td(const std::string& t) { return t == "<td>" || t == "<td" || t == "<td></td>"; }
}  // namespace

struct TableRecognizer::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "mlx-mineru-table"};
  Ort::SessionOptions opts;
  std::unique_ptr<Ort::Session> session;
  std::string in_name, out_loc, out_prob;
  std::vector<std::string> character;  // 50 tokens incl sos/eos
  int sos_idx = 0, eos_idx = 0;
};

TableRecognizer::TableRecognizer() : impl_(std::make_unique<Impl>()) {}
TableRecognizer::~TableRecognizer() = default;
TableRecognizer::TableRecognizer(TableRecognizer&&) noexcept = default;
TableRecognizer& TableRecognizer::operator=(TableRecognizer&&) noexcept = default;

TableRecognizer TableRecognizer::load(const std::string& onnx, const std::string& dict) {
  TableRecognizer r;
  Impl& m = *r.impl_;
  std::ifstream din(dict, std::ios::binary);
  if (!din) throw std::runtime_error("table: cannot open dict " + dict);
  std::string line;
  while (std::getline(din, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    std::string tok;  // unescape \\ and \n
    for (size_t i = 0; i < line.size(); ++i) {
      if (line[i] == '\\' && i + 1 < line.size()) {
        char n = line[++i];
        tok += (n == 'n') ? '\n' : n;
      } else {
        tok += line[i];
      }
    }
    m.character.push_back(tok);
  }
  for (int i = 0; i < (int)m.character.size(); ++i) {
    if (m.character[i] == "sos") m.sos_idx = i;
    if (m.character[i] == "eos") m.eos_idx = i;
  }
  m.session = std::make_unique<Ort::Session>(m.env, onnx.c_str(), m.opts);
  Ort::AllocatorWithDefaultOptions alloc;
  m.in_name = m.session->GetInputNameAllocated(0, alloc).get();
  m.out_loc = m.session->GetOutputNameAllocated(0, alloc).get();   // loc_preds [1,L,8]
  m.out_prob = m.session->GetOutputNameAllocated(1, alloc).get();  // structure_probs [1,L,50]
  return r;
}

TableStructure TableRecognizer::recognize_structure(const std::vector<uint8_t>& rgb, int w,
                                                    int h) const {
  const Impl& m = *impl_;
  // ResizeTableImage(max_len=488): ratio = 488/max(h,w).
  double ratio = (double)kMaxLen / std::max(h, w);
  int rh = (int)(h * ratio), rw = (int)(w * ratio);
  rh = std::max(1, rh);
  rw = std::max(1, rw);
  std::vector<uint8_t> resized = resize_rgb8_cv(rgb, w, h, rw, rh, kInterLinear);

  // NormalizeImage (BGR, like MinerU's cv2 input) into a 488x488 zero canvas, CHW.
  std::vector<float> input(static_cast<size_t>(3) * kMaxLen * kMaxLen, 0.0f);
  for (int y = 0; y < rh; ++y)
    for (int x = 0; x < rw; ++x)
      for (int c = 0; c < 3; ++c)
        input[(static_cast<size_t>(c) * kMaxLen + y) * kMaxLen + x] =
            (resized[(static_cast<size_t>(y) * rw + x) * 3 + (2 - c)] / 255.0f - kMean[c]) /
            kStd[c];

  std::array<int64_t, 4> ishape{1, 3, kMaxLen, kMaxLen};
  Ort::MemoryInfo mi = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value in = Ort::Value::CreateTensor<float>(mi, input.data(), input.size(), ishape.data(), 4);
  const char* in_names[] = {m.in_name.c_str()};
  const char* out_names[] = {m.out_loc.c_str(), m.out_prob.c_str()};
  auto outs = const_cast<Ort::Session&>(*m.session).Run(Ort::RunOptions{nullptr}, in_names, &in, 1,
                                                        out_names, 2);
  const float* loc = outs[0].GetTensorData<float>();
  const float* prob = outs[1].GetTensorData<float>();
  auto pshape = outs[1].GetTensorTypeAndShapeInfo().GetShape();
  int L = (int)pshape[1], C = (int)pshape[2];

  // TableLabelDecode: per-step argmax; collect tokens + <td> bboxes (decoded to orig w,h).
  TableStructure ts;
  ts.tokens = {"<html>", "<body>", "<table>"};
  for (int idx = 0; idx < L; ++idx) {
    const float* row = prob + static_cast<size_t>(idx) * C;
    int ci = 0;
    float bv = row[0];
    for (int c = 1; c < C; ++c)
      if (row[c] > bv) { bv = row[c]; ci = c; }
    if (idx > 0 && ci == m.eos_idx) break;
    if (ci == m.sos_idx || ci == m.eos_idx) continue;
    const std::string& text = m.character[ci];
    if (is_td(text)) {
      const float* b = loc + static_cast<size_t>(idx) * 8;
      std::array<float, 8> cell;
      for (int k = 0; k < 8; ++k) cell[k] = b[k] * (k % 2 == 0 ? w : h);  // _bbox_decode
      ts.cells.push_back(cell);
    }
    ts.tokens.push_back(text);
  }
  ts.tokens.insert(ts.tokens.end(), {"</table>", "</body>", "</html>"});

  // adapt_slanet_plus: scale cells from orig-image space into the padded-488 space.
  double a = std::min((double)kMaxLen / h, (double)kMaxLen / w);
  double w_ratio = kMaxLen / (w * a), h_ratio = kMaxLen / (h * a);
  for (auto& cell : ts.cells)
    for (int k = 0; k < 8; ++k) cell[k] *= (k % 2 == 0 ? w_ratio : h_ratio);
  return ts;
}

// ---- TableMatch (faithful port of slanet_plus/matcher.py) -------------------

namespace {
struct Box4 { double x0, y0, x1, y1; };

Box4 cell_to_box4(const std::array<float, 8>& c) {
  double xs[4] = {c[0], c[2], c[4], c[6]}, ys[4] = {c[1], c[3], c[5], c[7]};
  return {*std::min_element(xs, xs + 4), *std::min_element(ys, ys + 4),
          *std::max_element(xs, xs + 4), *std::max_element(ys, ys + 4)};
}
}  // namespace

std::string table_match(const std::vector<std::string>& structure,
                        const std::vector<std::array<float, 8>>& cells,
                        const std::vector<TableOcrItem>& ocr) {
  // get_boxes_recs-equivalent: OCR quad -> [xmin-1,ymin-1,xmax+1,ymax+1] (no w/h clamp
  // here; matcher only uses relative geometry).
  std::vector<Box4> dt;
  std::vector<const TableOcrItem*> items;
  for (const auto& o : ocr) {
    double xs[4] = {o.box[0][0], o.box[1][0], o.box[2][0], o.box[3][0]};
    double ys[4] = {o.box[0][1], o.box[1][1], o.box[2][1], o.box[3][1]};
    Box4 b{*std::min_element(xs, xs + 4) - 1, *std::min_element(ys, ys + 4) - 1,
           *std::max_element(xs, xs + 4) + 1, *std::max_element(ys, ys + 4) + 1};
    dt.push_back(b);
    items.push_back(&o);
  }
  std::vector<Box4> cb;
  for (const auto& c : cells) cb.push_back(cell_to_box4(c));

  // _filter_ocr_result: drop OCR boxes entirely above the table (max y < cells' min y).
  std::vector<int> keep;
  if (!cb.empty()) {
    double y1min = std::numeric_limits<double>::max();
    for (const auto& c : cells)
      for (int k = 1; k < 8; k += 2) y1min = std::min(y1min, (double)c[k]);
    for (int i = 0; i < (int)dt.size(); ++i) {
      double ymax = std::max(dt[i].y1, dt[i].y0);  // dt already has max in y1
      if (ymax < y1min) continue;
      keep.push_back(i);
    }
  }

  // match_result: each kept OCR box -> best cell by (1-IoU, distance); skip if IoU ~ 0.
  const double min_iou = std::pow(0.1, 8);
  std::map<int, std::vector<int>> matched;  // cell -> [ocr indices in `keep` order]
  for (int kk = 0; kk < (int)keep.size(); ++kk) {
    int i = keep[kk];
    const Box4& d = dt[i];
    double d_area = (d.x1 - d.x0) * (d.y1 - d.y0);
    double best_inv = std::numeric_limits<double>::max(), best_dist = 0;
    int best = -1;
    for (int j = 0; j < (int)cb.size(); ++j) {
      const Box4& c = cb[j];
      double c_area = (c.x1 - c.x0) * (c.y1 - c.y0);
      double left = std::max(d.y0, c.y0), right = std::min(d.y1, c.y1);
      double top = std::max(d.x0, c.x0), bottom = std::min(d.x1, c.x1);
      double iou = 0;
      if (left < right && top < bottom) {
        double inter = (right - left) * (bottom - top);
        double uni = d_area + c_area - inter;
        if (uni != 0) iou = inter / uni;
      }
      double dis = std::abs(c.x0 - d.x0) + std::abs(c.y0 - d.y0) + std::abs(c.x1 - d.x1) +
                   std::abs(c.y1 - d.y1);
      double dis2 = std::abs(c.x0 - d.x0) + std::abs(c.y0 - d.y0);
      double dis3 = std::abs(c.x1 - d.x1) + std::abs(c.y1 - d.y1);
      double dist = dis + std::min(dis2, dis3);
      double inv = 1.0 - iou;
      if (inv < best_inv || (inv == best_inv && dist < best_dist)) {
        best_inv = inv;
        best_dist = dist;
        best = j;
      }
    }
    if (best < 0 || best_inv >= 1 - min_iou) continue;
    matched[best].push_back(kk);  // store index into `keep`
  }

  // get_pred_html: walk structure, insert matched OCR text at each </td>.
  std::string html;
  int td_index = 0;
  auto contents = [&](int keep_idx) -> const std::string& { return items[keep[keep_idx]]->text; };
  for (const std::string& tag : structure) {
    if (tag.find("</td>") == std::string::npos) {
      if (tag != "<thead>" && tag != "</thead>" && tag != "<tbody>" && tag != "</tbody>")
        html += tag;
      continue;
    }
    if (tag == "<td></td>") html += "<td>";
    auto it = matched.find(td_index);
    if (it != matched.end()) {
      const auto& idxs = it->second;
      bool b_with = false;
      if (idxs.size() > 1 && contents(idxs[0]).find("<b>") != std::string::npos) {
        b_with = true;
        html += "<b>";
      }
      for (size_t i = 0; i < idxs.size(); ++i) {
        std::string content = contents(idxs[i]);
        if (idxs.size() > 1) {
          if (content.empty()) continue;
          if (content[0] == ' ') content = content.substr(1);
          if (content.rfind("<b>", 0) == 0) content = content.substr(3);
          size_t eb = content.rfind("</b>");
          if (eb != std::string::npos && eb == content.size() - 4) content = content.substr(0, content.size() - 4);
          if (content.empty()) continue;
          if (i != idxs.size() - 1 && (content.empty() || content.back() != ' ')) content += " ";
        }
        html += content;
      }
      if (b_with) html += "</b>";
    }
    html += (tag == "<td></td>") ? "</td>" : tag;
    ++td_index;
  }
  return html;
}

std::string TableRecognizer::recognize_html(const std::vector<uint8_t>& rgb, int w, int h,
                                            const std::vector<TableOcrItem>& ocr) const {
  TableStructure ts = recognize_structure(rgb, w, h);
  return table_match(ts.tokens, ts.cells, ocr);
}

}  // namespace mineru
