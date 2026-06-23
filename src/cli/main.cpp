// Copyright (c) mlx-mineru.
// mlx-mineru CLI — native C++/MLX MinerU. Pipeline: render a PDF page, run the
// Qwen2-VL two-step extract (layout detection -> per-block content) on Apple
// Silicon (MLX/Metal), assemble middle_json, and render Markdown via union_make.
// Zero Python at runtime. (MagicModel-level block association / post_process are
// simplified; see AGENT.md.)
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <mutex>

#include "CLI11/CLI11.hpp"
#include "httplib/httplib.h"
#include "mineru/enums.hpp"
#include "mineru/image_preprocess.hpp"
#include "mineru/mkcontent.hpp"
#include "mineru/otsl.hpp"
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
                                   int w, int h, const std::string& instruction, int max_new,
                                   bool keep_special = false) {
  rgb = resize_by_need(rgb, w, h);
  mineru::VisionInput vi = mineru::preprocess_image(rgb, w, h);
  int n_img = vi.seq_len() / (model.config().spatial_merge_size * model.config().spatial_merge_size);
  std::vector<float> embeds = model.forward_vision(vi.pixel_values, vi.grid_thw);
  std::vector<int> prompt = build_prompt(tok, n_img, model.config().image_token_id, instruction);
  std::vector<int> gen = model.generate_multimodal(prompt, embeds, n_img, vi.grid_thw, max_new, kEos);
  // Tables need the OTSL structure tokens (<fcel>/<nl>/...) which are *special*
  // tokens — keep them so convert_otsl_to_html can rebuild the grid.
  std::string s = tok.decode(gen, /*skip_special=*/!keep_special);
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

// Process one page: layout detection -> per-block content -> page_info json.
static json process_page(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                         const mineru::PageImage& pg, int page_idx, bool layout_only,
                         std::vector<mineru::ContentBlock>* layout_out = nullptr) {
  const auto& cfg = model.config();
  std::vector<uint8_t> resized = mineru::resize_bicubic_rgb8(pg.rgb, pg.width, pg.height, 1036, 1036);
  mineru::VisionInput vi = mineru::preprocess_image(resized, 1036, 1036);
  int n_img = vi.seq_len() / (cfg.spatial_merge_size * cfg.spatial_merge_size);
  std::vector<float> lembeds = model.forward_vision(vi.pixel_values, vi.grid_thw);
  std::vector<int> lprompt = build_prompt(tok, n_img, cfg.image_token_id, "\nLayout Detection:");
  std::vector<int> lgen = model.generate_multimodal(lprompt, lembeds, n_img, vi.grid_thw, 1200, kEos);
  std::vector<mineru::ContentBlock> blocks = mineru::parse_layout_output(tok.decode(lgen, false));
  std::cerr << "[mlx-mineru] page " << page_idx << ": " << blocks.size() << " blocks\n";
  if (layout_out) *layout_out = blocks;
  if (layout_only) return json::object();

  json para_blocks = json::array(), discarded = json::array();
  std::array<double, 2> page_size = {pg.width_pt, pg.height_pt};
  auto scaled = [&](const std::array<double, 4>& b) {
    return std::array<double, 4>{b[0] * page_size[0], b[1] * page_size[1], b[2] * page_size[0],
                                 b[3] * page_size[1]};
  };
  static const std::vector<std::string> discard_types = {
      bt::kHeader, bt::kFooter, bt::kPageNumber, bt::kPageFootnote, bt::kAsideText};
  int index = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& b = blocks[i];
    std::cerr << "[mlx-mineru]  block " << (i + 1) << "/" << blocks.size() << " (" << b.type << ")   \r";
    int cw, ch;
    std::vector<uint8_t> crop = crop_rgb(pg, b.bbox, cw, ch);
    auto bb = scaled(b.bbox);
    if (std::find(discard_types.begin(), discard_types.end(), b.type) != discard_types.end()) {
      std::string content = extract_content(model, tok, crop, cw, ch, "\nText Recognition:", 512);
      discarded.push_back({{"type", bt::kDiscarded}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                           {"lines", text_lines(content, bb)}});
      continue;
    }
    bool is_table = (b.type == "table");
    std::string content = extract_content(model, tok, crop, cw, ch, instruction_for(b.type),
                                          is_table ? 2048 : 1024, /*keep_special=*/is_table);
    json pb;
    if (b.type == "title") {
      pb = {{"type", bt::kTitle}, {"level", 1}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
            {"index", index}, {"lines", text_lines(content, bb)}};
    } else if (b.type == "table") {
      // Model emits OTSL for tables; convert to HTML (post_process simple_process).
      std::string html = mineru::convert_otsl_to_html(content);
      if (html.empty()) html = content;
      pb = {{"type", bt::kTable}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"blocks", json::array({{{"type", bt::kTableBody}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                {"lines", json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                    {"spans", json::array({{{"type", ct::kTable}, {"html", html}}})}}})}}})}};
    } else if (b.type == "equation" || b.type == "equation_block") {
      pb = {{"type", bt::kInterlineEquation}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"lines", json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                {"spans", json::array({{{"type", ct::kInterlineEquation}, {"content", content}}})}}})}};
    } else {
      pb = {{"type", bt::kText}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"lines", text_lines(content, bb)}};
    }
    para_blocks.push_back(pb);
    ++index;
  }
  std::cerr << "\n";
  return {{"page_idx", page_idx}, {"page_size", {page_size[0], page_size[1]}},
          {"para_blocks", para_blocks}, {"discarded_blocks", discarded}};
}

// Convert an open PDF document (page range) -> pdf_info (middle_json pages).
static json convert_document(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                             mineru::PdfDocument& doc, int start_page, int end_page) {
  int npages = doc.page_count();
  int s = std::clamp(start_page, 0, npages - 1);
  int e = (end_page < 0) ? npages - 1 : std::clamp(end_page, s, npages - 1);
  json pdf_info = json::array();
  for (int p = s; p <= e; ++p) {
    mineru::PageImage pg = doc.render_page(p);
    pdf_info.push_back(process_page(model, tok, pg, p, /*layout_only=*/false));
  }
  return pdf_info;
}

// Minimal HTTP API (aligns loosely with cli/fast_api.py): GET /health,
// POST /file_parse (raw PDF body or multipart "files") -> {md, content_list}.
static int run_server(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                      const std::string& host, int port) {
  httplib::Server srv;
  static std::mutex infer_mtx;  // serialize model use across requests

  srv.Get("/health", [](const httplib::Request&, httplib::Response& res) {
    res.set_content("{\"status\":\"ok\"}", "application/json");
  });

  srv.Post("/file_parse", [&](const httplib::Request& req, httplib::Response& res) {
    std::string pdf;
    if (req.is_multipart_form_data() && req.form.has_file("files")) {
      pdf = req.form.get_file("files").content;
    } else {
      pdf = req.body;
    }
    if (pdf.size() < 5 || pdf.compare(0, 5, "%PDF-") != 0) {
      res.status = 400;
      res.set_content("{\"error\":\"expected a PDF body or multipart 'files'\"}", "application/json");
      return;
    }
    try {
      std::lock_guard<std::mutex> lock(infer_mtx);
      std::vector<uint8_t> bytes(pdf.begin(), pdf.end());
      mineru::PdfDocument doc = mineru::PdfDocument::open_bytes(bytes);
      json pdf_info = convert_document(model, tok, doc, 0, -1);
      std::string md = mineru::union_make(pdf_info, mineru::make_mode::kMmMd, "images").get<std::string>();
      json content_list = mineru::union_make(pdf_info, mineru::make_mode::kContentList, "images");
      json out = {{"md_content", md}, {"content_list", content_list}};
      res.set_content(out.dump(), "application/json");
    } catch (const std::exception& ex) {
      res.status = 500;
      res.set_content(std::string("{\"error\":\"") + ex.what() + "\"}", "application/json");
    }
  });

  std::cerr << "[mlx-mineru] serving on http://" << host << ":" << port
            << "  (GET /health, POST /file_parse)\n";
  if (!srv.listen(host, port)) {
    std::cerr << "[mlx-mineru] failed to bind " << host << ":" << port << "\n";
    return 1;
  }
  return 0;
}

int main(int argc, char** argv) {
  CLI::App app{"mlx-mineru — native C++/MLX MinerU (PDF -> Markdown)"};
  std::string pdf_path, model_dir = "models/MinerU2.5-tokenizer", out_dir = "output";
  std::string host = "127.0.0.1";
  int start_page = 0, end_page = -1, port = 8000;
  bool layout_only = false, server = false;
  app.add_option("-p,--path", pdf_path, "Input PDF path (not needed with --server)");
  app.add_option("-m,--model", model_dir, "Model directory (weights + tokenizer)");
  app.add_option("-s,--start", start_page, "First 0-based page (default 0)");
  app.add_option("-e,--end", end_page, "Last 0-based page inclusive (default: last)");
  app.add_option("-o,--output", out_dir, "Output directory (MinerU layout: <out>/<name>/vlm/)");
  app.add_flag("--layout-only", layout_only, "Only run layout detection, emit JSON");
  app.add_flag("--server", server, "Run the HTTP API server instead of a one-shot conversion");
  app.add_option("--host", host, "Server bind host (default 127.0.0.1)");
  app.add_option("--port", port, "Server port (default 8000)");
  CLI11_PARSE(app, argc, argv);

  auto t0 = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] loading model ...\n";
  mineru::Qwen2Tokenizer tok = mineru::Qwen2Tokenizer::load(model_dir);
  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(model_dir + "/model.safetensors");

  if (server) return run_server(model, tok, host, port);

  if (pdf_path.empty()) { std::cerr << "--path is required (or use --server)\n"; return 2; }
  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf_path);
  int npages = doc.page_count();
  int s = std::clamp(start_page, 0, npages - 1);
  int e = (end_page < 0) ? npages - 1 : std::clamp(end_page, s, npages - 1);
  std::cerr << "[mlx-mineru] " << pdf_path << ": pages " << s << ".." << e << " of " << npages << "\n";

  json pdf_info = json::array();
  json layout_json = json::array();
  for (int p = s; p <= e; ++p) {
    mineru::PageImage pg = doc.render_page(p);
    std::vector<mineru::ContentBlock> blocks;
    json page_info = process_page(model, tok, pg, p, layout_only, &blocks);
    if (layout_only) {
      json jb = json::array();
      for (auto& b : blocks) {
        json o = {{"type", b.type}, {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}}};
        if (b.angle) o["angle"] = *b.angle;
        jb.push_back(o);
      }
      layout_json.push_back({{"page_idx", p}, {"blocks", jb}});
    } else {
      pdf_info.push_back(page_info);
    }
  }

  namespace fs = std::filesystem;
  std::string stem = fs::path(pdf_path).stem().string();
  fs::path dir = fs::path(out_dir) / stem / "vlm";
  fs::create_directories(dir);

  auto write = [&](const std::string& fname, const std::string& data) {
    std::ofstream(dir / fname) << data;
    std::cerr << "[mlx-mineru] wrote " << (dir / fname).string() << "\n";
  };

  if (layout_only) {
    write(stem + "_layout.json", layout_json.dump(2));
  } else {
    // MinerU-style outputs, all from the verified union_make / middle_json contract.
    std::string md = mineru::union_make(pdf_info, mineru::make_mode::kMmMd, "images").get<std::string>();
    json content_list = mineru::union_make(pdf_info, mineru::make_mode::kContentList, "images");
    json middle = {{"pdf_info", pdf_info}, {"_backend", "vlm"}, {"_version_name", mineru::kMineruVersion}};
    write(stem + ".md", md);
    write(stem + "_content_list.json", content_list.dump(4));
    write(stem + "_middle.json", middle.dump(4));
  }

  auto t1 = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] done in " << std::chrono::duration<double>(t1 - t0).count() << "s\n";
  return 0;
}
