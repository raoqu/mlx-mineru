// Copyright (c) mlx-mineru.
// mlx-mineru CLI — native C++/MLX MinerU. Pipeline: render a PDF page, run the
// Qwen2-VL two-step extract (layout detection -> per-block content) on Apple
// Silicon (MLX/Metal), assemble middle_json, and render Markdown via union_make.
// Zero Python at runtime. (MagicModel-level block association / post_process are
// simplified; see AGENT.md.)
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"
#include "mineru/enums.hpp"
#include "mineru/image_preprocess.hpp"
#include "mineru/mkcontent.hpp"
#include "mineru/pdf.hpp"
#include "mineru/qwen2_vl.hpp"
#include "mineru/tokenizer.hpp"
#include "mineru/vlm_layout.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;
namespace bt = mineru::block_type;
namespace ct = mineru::content_type;

static const std::vector<int> kEos = {151645, 151643};

// Build a Qwen2-VL chat prompt: system + user(image + instruction) + assistant.
static std::vector<int> build_prompt(const mineru::Qwen2Tokenizer& tok, int n_img, int pad_id,
                                     const std::string& instruction) {
  std::vector<int> ids = tok.encode(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n<|vision_start|>");
  ids.insert(ids.end(), n_img, pad_id);
  std::vector<int> suffix = tok.encode("<|vision_end|>" + instruction +
                                       "<|im_end|>\n<|im_start|>assistant\n");
  ids.insert(ids.end(), suffix.begin(), suffix.end());
  return ids;
}

// Crop a normalized bbox out of an RGB8 page image (clamped).
static std::vector<uint8_t> crop_rgb(const mineru::PageImage& pg, const std::array<double, 4>& b,
                                     int& cw, int& ch) {
  int x0 = std::clamp((int)std::floor(b[0] * pg.width), 0, pg.width - 1);
  int y0 = std::clamp((int)std::floor(b[1] * pg.height), 0, pg.height - 1);
  int x1 = std::clamp((int)std::ceil(b[2] * pg.width), x0 + 1, pg.width);
  int y1 = std::clamp((int)std::ceil(b[3] * pg.height), y0 + 1, pg.height);
  cw = x1 - x0;
  ch = y1 - y0;
  std::vector<uint8_t> out((size_t)cw * ch * 3);
  for (int y = 0; y < ch; ++y)
    for (int x = 0; x < cw; ++x)
      for (int c = 0; c < 3; ++c)
        out[((size_t)y * cw + x) * 3 + c] = pg.rgb[((size_t)(y0 + y) * pg.width + (x0 + x)) * 3 + c];
  return out;
}

// resize_by_need (minimal): upscale (bicubic) if the short edge < min_image_edge.
static std::vector<uint8_t> resize_by_need(std::vector<uint8_t> rgb, int& w, int& h, int min_edge = 28) {
  if (std::min(w, h) >= min_edge) return rgb;
  double s = (double)min_edge / std::min(w, h);
  int nw = (int)std::ceil(w * s), nh = (int)std::ceil(h * s);
  rgb = mineru::resize_bicubic_rgb8(rgb, w, h, nw, nh);
  w = nw;
  h = nh;
  return rgb;
}

// Run the VLM on one cropped block image with a type-specific instruction.
static std::string extract_content(const mineru::Qwen2VLModel& model,
                                   const mineru::Qwen2Tokenizer& tok, std::vector<uint8_t> rgb,
                                   int w, int h, const std::string& instruction, int max_new) {
  rgb = resize_by_need(rgb, w, h);
  mineru::VisionInput vi = mineru::preprocess_image(rgb, w, h);
  int n_img = vi.seq_len() / (model.config().spatial_merge_size * model.config().spatial_merge_size);
  std::vector<float> embeds = model.forward_vision(vi.pixel_values, vi.grid_thw);
  std::vector<int> prompt = build_prompt(tok, n_img, model.config().image_token_id, instruction);
  std::vector<int> gen = model.generate_multimodal(prompt, embeds, n_img, vi.grid_thw, max_new, kEos);
  std::string s = tok.decode(gen, /*skip_special=*/true);
  // trim
  size_t a = s.find_first_not_of(" \t\r\n");
  size_t z = s.find_last_not_of(" \t\r\n");
  return a == std::string::npos ? "" : s.substr(a, z - a + 1);
}

// Per-layout-type extraction instruction (DEFAULT_PROMPTS).
static std::string instruction_for(const std::string& type) {
  if (type == "table") return "\nTable Recognition:";
  if (type == "equation" || type == "equation_block") return "\nFormula Recognition:";
  if (type == "image" || type == "chart") return "\nImage Analysis:";
  return "\nText Recognition:";
}

// One text line/span wrapper.
static json text_lines(const std::string& content, const std::array<double, 4>& bb) {
  return json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                       {"spans", json::array({{{"type", ct::kText}, {"content", content}}})}}});
}

int main(int argc, char** argv) {
  CLI::App app{"mlx-mineru — native C++/MLX MinerU (PDF -> Markdown)"};
  std::string pdf_path, model_dir = "models/MinerU2.5-tokenizer", out_path;
  int page = 0, layout_max = 1200, content_max = 2048;
  bool layout_only = false;
  app.add_option("-p,--path", pdf_path, "Input PDF path")->required();
  app.add_option("-m,--model", model_dir, "Model directory (weights + tokenizer)");
  app.add_option("--page", page, "0-based page index");
  app.add_option("-o,--output", out_path, "Output file (.md, or .json with --layout-only)");
  app.add_flag("--layout-only", layout_only, "Only run layout detection, emit JSON");
  CLI11_PARSE(app, argc, argv);

  auto t0 = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] loading model ...\n";
  mineru::Qwen2Tokenizer tok = mineru::Qwen2Tokenizer::load(model_dir);
  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(model_dir + "/model.safetensors");
  const auto& cfg = model.config();

  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf_path);
  if (page < 0 || page >= doc.page_count()) { std::cerr << "page out of range\n"; return 2; }
  mineru::PageImage pg = doc.render_page(page);

  // Step 1: layout detection on the 1036x1036 page.
  std::vector<uint8_t> resized = mineru::resize_bicubic_rgb8(pg.rgb, pg.width, pg.height, 1036, 1036);
  mineru::VisionInput vi = mineru::preprocess_image(resized, 1036, 1036);
  int n_img = vi.seq_len() / (cfg.spatial_merge_size * cfg.spatial_merge_size);
  std::cerr << "[mlx-mineru] layout detection (" << n_img << " image tokens) ...\n";
  std::vector<float> lembeds = model.forward_vision(vi.pixel_values, vi.grid_thw);
  std::vector<int> lprompt = build_prompt(tok, n_img, cfg.image_token_id, "\nLayout Detection:");
  std::vector<int> lgen = model.generate_multimodal(lprompt, lembeds, n_img, vi.grid_thw, layout_max, kEos);
  std::vector<mineru::ContentBlock> blocks = mineru::parse_layout_output(tok.decode(lgen, false));
  std::cerr << "[mlx-mineru] " << blocks.size() << " blocks detected\n";

  if (layout_only) {
    json out = {{"page_idx", page}, {"num_blocks", blocks.size()}, {"blocks", json::array()}};
    for (auto& b : blocks) {
      json jb = {{"type", b.type}, {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}}};
      if (b.angle) jb["angle"] = *b.angle;
      out["blocks"].push_back(jb);
    }
    std::string s = out.dump(2);
    if (out_path.empty()) std::cout << s << "\n"; else std::ofstream(out_path) << s;
    return 0;
  }

  // Step 2: per-block content extraction + middle_json assembly.
  json para_blocks = json::array(), discarded = json::array();
  std::array<double, 2> page_size = {pg.width_pt, pg.height_pt};
  auto scaled = [&](const std::array<double, 4>& b) {
    return std::array<double, 4>{b[0] * page_size[0], b[1] * page_size[1],
                                 b[2] * page_size[0], b[3] * page_size[1]};
  };
  static const std::vector<std::string> discard_types = {
      bt::kHeader, bt::kFooter, bt::kPageNumber, bt::kPageFootnote, bt::kAsideText};

  int index = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& b = blocks[i];
    std::cerr << "[mlx-mineru] block " << (i + 1) << "/" << blocks.size() << " (" << b.type << ") ...\r";
    int cw, ch;
    std::vector<uint8_t> crop = crop_rgb(pg, b.bbox, cw, ch);
    auto bb = scaled(b.bbox);

    if (std::find(discard_types.begin(), discard_types.end(), b.type) != discard_types.end()) {
      std::string content = extract_content(model, tok, crop, cw, ch, "\nText Recognition:", 512);
      discarded.push_back({{"type", bt::kDiscarded}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                           {"lines", text_lines(content, bb)}});
      continue;
    }
    std::string instr = instruction_for(b.type);
    std::string content = extract_content(model, tok, crop, cw, ch, instr,
                                          (b.type == "table") ? content_max : 1024);

    json pb;
    if (b.type == "title") {
      pb = {{"type", bt::kTitle}, {"level", 1}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
            {"index", index}, {"lines", text_lines(content, bb)}};
    } else if (b.type == "table") {
      pb = {{"type", bt::kTable}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"blocks", json::array({{{"type", bt::kTableBody}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                {"lines", json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                    {"spans", json::array({{{"type", ct::kTable}, {"html", content}}})}}})}}})}};
    } else if (b.type == "equation" || b.type == "equation_block") {
      pb = {{"type", bt::kInterlineEquation}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"lines", json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                {"spans", json::array({{{"type", ct::kInterlineEquation}, {"content", content}}})}}})}};
    } else {
      // text-like (text/ref_text/phonetic/index/list/code/image-analysis-as-text)
      std::string btype = (b.type == "code" || b.type == "algorithm") ? bt::kText : bt::kText;
      pb = {{"type", btype}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"lines", text_lines(content, bb)}};
    }
    para_blocks.push_back(pb);
    ++index;
  }
  std::cerr << "\n";

  json middle = {{"pdf_info", json::array({{{"page_idx", page}, {"page_size", {page_size[0], page_size[1]}},
                  {"para_blocks", para_blocks}, {"discarded_blocks", discarded}}})},
                 {"_backend", "vlm"}, {"_version_name", mineru::kMineruVersion}};

  std::string md = mineru::union_make(middle["pdf_info"], mineru::make_mode::kMmMd, "images").get<std::string>();

  auto t1 = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] done in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";
  if (out_path.empty()) {
    std::cout << md << "\n";
  } else {
    std::ofstream(out_path) << md;
    std::cerr << "[mlx-mineru] wrote " << out_path << "\n";
  }
  return 0;
}
