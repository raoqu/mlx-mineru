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
#include "mineru/image_write.hpp"
#include "mineru/mkcontent.hpp"
#include "mineru/otsl.hpp"
#include "mineru/pdf.hpp"
#include "mineru/post_process.hpp"
#include "mineru/qwen2_vl.hpp"
#include "mineru/tokenizer.hpp"
#include "mineru/vlm_layout.hpp"
#include "nlohmann/json.hpp"
#ifdef MINERU_HAVE_PIPELINE
#include <memory>

#include "mineru/formula_rec.hpp"
#include "mineru/layout_det.hpp"
#include "mineru/ocr_det.hpp"
#include "mineru/ocr_rec.hpp"
#include "mineru/pipeline_driver.hpp"
#endif

using nlohmann::json;
namespace bt = mineru::block_type;
namespace ct = mineru::content_type;

static const std::vector<int> kEos = {151645, 151643};

// Detailed phase profiler (printed at end of a one-shot CLI run).
struct Prof {
  double load = 0, raster = 0, layout_vision = 0, layout_gen = 0, content_vision = 0,
         content_gen = 0, assembly = 0;
  long layout_tok = 0, content_tok = 0;
  int pages = 0, blocks = 0;
};
static Prof g_prof;
using Clock = std::chrono::steady_clock;
static double secs(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

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

// Layout-image preprocessing for one page (1036x1036) + the Layout-Detection prompt.
struct LayoutPrep {
  std::vector<float> pv;
  std::array<int, 3> grid;
  int n_img;
  std::vector<int> prompt;
};
static LayoutPrep prep_layout(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                              const mineru::PageImage& pg) {
  const auto& cfg = model.config();
  std::vector<uint8_t> resized = mineru::resize_bicubic_rgb8(pg.rgb, pg.width, pg.height, 1036, 1036);
  mineru::VisionInput vi = mineru::preprocess_image(resized, 1036, 1036);
  LayoutPrep lp;
  lp.n_img = vi.seq_len() / (cfg.spatial_merge_size * cfg.spatial_merge_size);
  lp.grid = vi.grid_thw;
  lp.pv = std::move(vi.pixel_values);
  lp.prompt = build_prompt(tok, lp.n_img, cfg.image_token_id, "\nLayout Detection:");
  return lp;
}

// Process one page: layout detection -> per-block content -> page_info json.
// If `precomputed` is given, layout detection is skipped (blocks already known,
// e.g. from a cross-page batched layout pass).
static json process_page(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                         const mineru::PageImage& pg, int page_idx, bool layout_only,
                         std::vector<mineru::ContentBlock>* layout_out = nullptr,
                         const std::string& images_dir = "", int batch_size = 6,
                         const std::vector<mineru::ContentBlock>* precomputed = nullptr,
                         bool skip_image_rec = false) {
  const auto& cfg = model.config();
  std::vector<mineru::ContentBlock> blocks;
  if (precomputed) {
    blocks = *precomputed;
  } else {
    LayoutPrep lp = prep_layout(model, tok, pg);
    auto _lv0 = Clock::now();
    std::vector<float> lembeds = model.forward_vision(lp.pv, lp.grid);
    auto _lv1 = Clock::now();
    std::vector<int> lgen =
        model.generate_multimodal(lp.prompt, lembeds, lp.n_img, lp.grid, 1200, kEos);
    g_prof.layout_vision += secs(_lv0, _lv1);
    g_prof.layout_gen += secs(_lv1, Clock::now());
    g_prof.layout_tok += (long)lgen.size();
    blocks = mineru::parse_layout_output(tok.decode(lgen, false));
    g_prof.pages += 1;
    g_prof.blocks += (int)blocks.size();
  }
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

  // Pass 1: per block, crop + (for blocks needing content) vision tower + prompt.
  // With --no-image-rec, image/chart blocks are cropped/saved but skip the VLM
  // understanding entirely (no vision encode, no generation) -> much faster.
  struct Job { int cw, ch; bool keep_special; std::vector<uint8_t> crop; };
  std::vector<Job> jobs(blocks.size());
  std::vector<std::vector<int>> prompts(blocks.size());
  std::vector<std::vector<float>> pvs(blocks.size());  // per-block pixel_values (CPU)
  std::vector<int> nimgs(blocks.size());
  std::vector<std::array<int, 3>> grids(blocks.size());
  std::vector<bool> skip(blocks.size(), false);
  std::vector<int> active;  // block indices that need VLM content
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& b = blocks[i];
    std::cerr << "[mlx-mineru]  preprocess " << (i + 1) << "/" << blocks.size() << " (" << b.type << ")   \r";
    bool discard = std::find(discard_types.begin(), discard_types.end(), b.type) != discard_types.end();
    Job& j = jobs[i];
    j.crop = crop_rgb(pg, b.bbox, j.cw, j.ch);  // always crop (image/chart still saved)
    j.keep_special = (b.type == "table");
    if (skip_image_rec && (b.type == "image" || b.type == "chart")) { skip[i] = true; continue; }
    std::vector<uint8_t> crop = resize_by_need(j.crop, j.cw, j.ch);
    mineru::VisionInput cvi = mineru::preprocess_image(crop, j.cw, j.ch);
    nimgs[i] = cvi.seq_len() / (cfg.spatial_merge_size * cfg.spatial_merge_size);
    grids[i] = cvi.grid_thw;
    pvs[i] = std::move(cvi.pixel_values);
    std::string instr = discard ? "\nText Recognition:" : instruction_for(b.type);
    prompts[i] = build_prompt(tok, nimgs[i], cfg.image_token_id, instr);
    active.push_back((int)i);
  }
  // Encode the active crops in one synchronized batch (MLX overlaps them on the GPU).
  std::cerr << "\n[mlx-mineru]  batched vision over " << active.size() << " crops ...\n";
  std::vector<std::vector<float>> pvs_a, grids_unused;
  std::vector<std::array<int, 3>> grids_a;
  for (int i : active) { pvs_a.push_back(pvs[i]); grids_a.push_back(grids[i]); }
  auto _cv0 = Clock::now();
  auto embeds_a = model.forward_vision_batch(pvs_a, grids_a);
  std::vector<std::vector<float>> embeds(blocks.size());
  for (size_t k = 0; k < active.size(); ++k) embeds[active[k]] = std::move(embeds_a[k]);
  g_prof.content_vision += secs(_cv0, Clock::now());
  // Pass 2: batched generation over the active blocks, length-bucketed.
  const int kBatch = std::max(1, batch_size);
  std::vector<std::vector<int>> gens(blocks.size());
  std::vector<int> order = active;
  std::sort(order.begin(), order.end(),
            [&](int a, int b) { return prompts[a].size() < prompts[b].size(); });
  std::cerr << "[mlx-mineru]  batched generation over " << order.size() << " blocks (groups of "
            << kBatch << ") ...\n";
  auto _cg0 = Clock::now();
  for (size_t off = 0; off < order.size(); off += kBatch) {
    size_t end = std::min(order.size(), off + kBatch);
    std::vector<std::vector<int>> bp;
    std::vector<std::vector<float>> be;
    std::vector<int> bn;
    std::vector<std::array<int, 3>> bg;
    for (size_t k = off; k < end; ++k) {
      int i = order[k];
      bp.push_back(prompts[i]);
      be.push_back(embeds[i]);
      bn.push_back(nimgs[i]);
      bg.push_back(grids[i]);
    }
    auto out = model.generate_multimodal_batch(bp, be, bn, bg, 2048, kEos);
    for (size_t k = off; k < end; ++k) gens[order[k]] = std::move(out[k - off]);
  }
  g_prof.content_gen += secs(_cg0, Clock::now());
  for (auto& g : gens) g_prof.content_tok += (long)g.size();

  // Pass 3: decode + assemble.
  int index = 0;
  for (size_t i = 0; i < blocks.size(); ++i) {
    const auto& b = blocks[i];
    Job& j = jobs[i];
    auto bb = scaled(b.bbox);
    std::string content = tok.decode(gens[i], /*skip_special=*/!j.keep_special);
    {  // trim
      size_t a = content.find_first_not_of(" \t\r\n");
      size_t z = content.find_last_not_of(" \t\r\n");
      content = (a == std::string::npos) ? "" : content.substr(a, z - a + 1);
    }
    if (std::find(discard_types.begin(), discard_types.end(), b.type) != discard_types.end()) {
      discarded.push_back({{"type", bt::kDiscarded}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                           {"lines", text_lines(content, bb)}});
      continue;
    }
    int cw = j.cw, ch = j.ch;
    std::vector<uint8_t>& crop = j.crop;
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
      std::string latex = mineru::pp::process_equation(content);
      pb = {{"type", bt::kInterlineEquation}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"lines", json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                {"spans", json::array({{{"type", ct::kInterlineEquation}, {"content", latex}}})}}})}};
    } else if ((b.type == "image" || b.type == "chart") && !images_dir.empty()) {
      // Save the cropped region; reference it via image_path (bucket "images").
      std::string fname = mineru::content_hash_hex(crop) + ".jpg";
      mineru::write_jpeg(images_dir + "/" + fname, crop, cw, ch);
      const char* body = (b.type == "chart") ? bt::kChartBody : bt::kImageBody;
      const char* span = (b.type == "chart") ? ct::kChart : ct::kImage;
      pb = {{"type", (b.type == "chart") ? bt::kChart : bt::kImage},
            {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"blocks", json::array({{{"type", body}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                {"lines", json::array({{{"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                    {"spans", json::array({{{"type", span}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}},
                        {"image_path", fname}, {"content", content}}})}}})}}})}};
    } else {
      pb = {{"type", bt::kText}, {"bbox", {bb[0], bb[1], bb[2], bb[3]}}, {"index", index},
            {"lines", text_lines(mineru::pp::process_text(content), bb)}};
    }
    para_blocks.push_back(pb);
    ++index;
  }
  std::cerr << "\n";
  return {{"page_idx", page_idx}, {"page_size", {page_size[0], page_size[1]}},
          {"para_blocks", para_blocks}, {"discarded_blocks", discarded}};
}

// Batch layout detection across ALL pages (uniform 1036x1036 layout images -> no
// padding waste), then run per-page content. The layout generation — batch-1 and
// memory-bandwidth bound when done page-by-page — benefits most from batching.
static json convert_batched(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                            std::vector<mineru::PageImage>& pages,
                            const std::vector<int>& page_idxs, const std::string& images_dir,
                            int batch_size, bool skip_image_rec = false) {
  int P = (int)pages.size();
  std::vector<std::vector<float>> pvs(P);
  std::vector<std::array<int, 3>> grids(P);
  std::vector<int> nimgs(P);
  std::vector<std::vector<int>> prompts(P);
  for (int p = 0; p < P; ++p) {
    LayoutPrep lp = prep_layout(model, tok, pages[p]);
    pvs[p] = std::move(lp.pv);
    grids[p] = lp.grid;
    nimgs[p] = lp.n_img;
    prompts[p] = std::move(lp.prompt);
  }
  std::cerr << "[mlx-mineru] batched layout over " << P << " page(s) ...\n";
  auto _lv0 = Clock::now();
  std::vector<std::vector<float>> lembeds = model.forward_vision_batch(pvs, grids);
  auto _lg0 = Clock::now();
  g_prof.layout_vision += secs(_lv0, _lg0);
  std::vector<std::vector<int>> lgens =
      model.generate_multimodal_batch(prompts, lembeds, nimgs, grids, 1200, kEos);
  g_prof.layout_gen += secs(_lg0, Clock::now());
  for (auto& g : lgens) g_prof.layout_tok += (long)g.size();

  std::vector<std::vector<mineru::ContentBlock>> blocks(P);
  for (int p = 0; p < P; ++p) {
    blocks[p] = mineru::parse_layout_output(tok.decode(lgens[p], false));
    g_prof.pages += 1;
    g_prof.blocks += (int)blocks[p].size();
  }
  json pdf_info = json::array();
  for (int p = 0; p < P; ++p)
    pdf_info.push_back(process_page(model, tok, pages[p], page_idxs[p], /*layout_only=*/false,
                                    nullptr, images_dir, batch_size, &blocks[p], skip_image_rec));
  return pdf_info;
}

// Convert an open PDF document (page range) -> pdf_info (middle_json pages).
static json convert_document(const mineru::Qwen2VLModel& model, const mineru::Qwen2Tokenizer& tok,
                             mineru::PdfDocument& doc, int start_page, int end_page) {
  int npages = doc.page_count();
  int s = std::clamp(start_page, 0, npages - 1);
  int e = (end_page < 0) ? npages - 1 : std::clamp(end_page, s, npages - 1);
  std::vector<mineru::PageImage> pages;
  std::vector<int> idxs;
  for (int p = s; p <= e; ++p) {
    pages.push_back(doc.render_page(p));
    idxs.push_back(p);
  }
  return convert_batched(model, tok, pages, idxs, /*images_dir=*/"", /*batch_size=*/6);
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

#ifdef MINERU_HAVE_PIPELINE
// Native pipeline backend: render -> layout(+reading order) + OCR det -> model_list ->
// assemble + post-OCR text-fill -> union_make. No VLM model needed.
static int run_pipeline(const std::string& pdf_path, const std::string& models,
                        const std::string& out_dir, int start_page, int end_page, int dpi) {
  namespace fs = std::filesystem;
  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf_path);
  int npages = doc.page_count();
  int s = std::clamp(start_page, 0, npages - 1);
  int e = (end_page < 0) ? npages - 1 : std::clamp(end_page, s, npages - 1);
  std::cerr << "[mlx-mineru] pipeline backend: " << pdf_path << " pages " << s << ".." << e
            << " of " << npages << "\n";

  auto t0 = Clock::now();
  mineru::LayoutDetector layout = mineru::LayoutDetector::load(models + "/Layout");
  mineru::TextDetector det = mineru::TextDetector::load(models + "/OCR/ocr_det.onnx");
  mineru::TextRecognizer rec =
      mineru::TextRecognizer::load(models + "/OCR/ocr_rec.onnx", models + "/OCR/ppocrv6_dict.txt");
  // Formula recognition is optional (large model); load it if present.
  namespace fsx = std::filesystem;
  std::unique_ptr<mineru::FormulaRecognizer> mfr;
  if (fsx::exists(models + "/MFR/mfr_encoder.onnx")) {
    mfr = std::make_unique<mineru::FormulaRecognizer>(mineru::FormulaRecognizer::load(
        models + "/MFR/mfr_encoder.onnx", models + "/MFR/mfr_decoder.onnx",
        models + "/MFR/mfr_vocab.txt"));
    std::cerr << "[mlx-mineru] formula recognizer loaded\n";
  }
  // Table recognition is optional (needs a per-crop OCR pipeline + SLANet+).
  std::unique_ptr<mineru::OcrPipeline> table_ocr;
  std::unique_ptr<mineru::TableRecognizer> table_rec;
  std::string slanet = models + "/TabRec/SlanetPlus/slanet-plus.onnx";
  if (fsx::exists(slanet)) {
    table_ocr = std::make_unique<mineru::OcrPipeline>(mineru::OcrPipeline::load(
        models + "/OCR/ocr_det.onnx", models + "/OCR/ocr_rec.onnx",
        models + "/OCR/ppocrv6_dict.txt"));
    table_rec = std::make_unique<mineru::TableRecognizer>(mineru::TableRecognizer::load(
        slanet, models + "/TabRec/SlanetPlus/table_structure_dict.txt"));
    std::cerr << "[mlx-mineru] table recognizer loaded (wireless/SLANet+)\n";
  }
  std::cerr << "[mlx-mineru] pipeline models loaded in " << secs(t0, Clock::now()) << "s\n";

  auto t_inf = Clock::now();
  json model_list = json::array();
  std::vector<mineru::PipelinePageImage> pages;
  for (int p = s; p <= e; ++p) {
    mineru::PageImage im = doc.render_page(p, dpi);
    model_list.push_back(mineru::build_page_model(layout, det, im.rgb, im.width, im.height,
                                                  mfr.get(), table_ocr.get(), table_rec.get()));
    mineru::PipelinePageImage pg;
    pg.page_w = (int)std::lround(im.width_pt);
    pg.page_h = (int)std::lround(im.height_pt);
    pg.w = im.width;
    pg.h = im.height;
    pg.rgb = std::move(im.rgb);
    // Digital PDF: use the embedded text layer (faster + exact) instead of OCR.
    for (const mineru::PdfChar& c : doc.extract_chars(p))
      pg.chars.push_back({c.cp, c.idx, c.x0, c.y0, c.x1, c.y1});
    bool digital = !pg.chars.empty();
    pages.push_back(std::move(pg));
    std::cerr << "[mlx-mineru] page " << p << ": "
              << model_list.back()["layout_dets"].size() << " dets, "
              << (digital ? "digital text" : "OCR") << "\n";
  }
  json pdf_info = mineru::pipeline_assemble_pages(model_list, pages, rec);

  std::string stem = fs::path(pdf_path).stem().string();
  fs::path dir = fs::path(out_dir) / stem / "pipeline";
  fs::create_directories(dir);
  std::string md =
      mineru::union_make(pdf_info, mineru::make_mode::kMmMd, "images").get<std::string>();
  json content_list = mineru::union_make(pdf_info, mineru::make_mode::kContentList, "images");
  json middle = {{"pdf_info", pdf_info}, {"_backend", "pipeline"},
                 {"_version_name", mineru::kMineruVersion}};
  auto write = [&](const std::string& fn, const std::string& data) {
    std::ofstream(dir / fn) << data;
    std::cerr << "[mlx-mineru] wrote " << (dir / fn).string() << "\n";
  };
  write(stem + ".md", md);
  write(stem + "_content_list.json", content_list.dump(4));
  write(stem + "_middle.json", middle.dump(4));
  std::cerr << "[mlx-mineru] pipeline inference " << secs(t_inf, Clock::now()) << "s; total "
            << secs(t0, Clock::now()) << "s\n";
  return 0;
}
#endif

int main(int argc, char** argv) {
  CLI::App app{"mlx-mineru — native C++/MLX MinerU (PDF -> Markdown)"};
  std::string pdf_path, model_dir = "models/MinerU2.5-tokenizer", out_dir = "output";
  std::string host = "127.0.0.1", backend = "vlm", pipeline_models = "models/pipeline";
  int start_page = 0, end_page = -1, port = 8000, bits = 4, batch_size = 6, pipeline_dpi = 200;
  bool layout_only = false, server = false, no_image_rec = false;
  app.add_option("-b,--backend", backend, "Backend: vlm (default) | pipeline (native ONNX CV)");
  app.add_option("--pipeline-models", pipeline_models,
                 "Pipeline backend model dir (default models/pipeline)");
  app.add_option("--pipeline-dpi", pipeline_dpi, "Pipeline render DPI (default 200)");
  app.add_option("-p,--path", pdf_path, "Input PDF path (not needed with --server)");
  app.add_option("-m,--model", model_dir, "Model directory (weights + tokenizer)");
  app.add_option("-s,--start", start_page, "First 0-based page (default 0)");
  app.add_option("-e,--end", end_page, "Last 0-based page inclusive (default: last)");
  app.add_option("-o,--output", out_dir, "Output directory (MinerU layout: <out>/<name>/vlm/)");
  app.add_flag("--layout-only", layout_only, "Only run layout detection, emit JSON (block types + bboxes), no content recognition");
  app.add_flag("--no-image-rec", no_image_rec, "VLM backend: skip image/chart understanding (still saves the cropped image), much faster");
  app.add_flag("--server", server, "Run the HTTP API server instead of a one-shot conversion");
  app.add_option("--host", host, "Server bind host (default 127.0.0.1)");
  app.add_option("--port", port, "Server port (default 8000)");
  app.add_option("--bits", bits, "Weight quantization bits: 4 (default) / 8 / 0 (full bf16)");
  app.add_option("--batch", batch_size, "Block generation batch size (default 6; 1 = sequential)");
  CLI11_PARSE(app, argc, argv);

  if (backend == "pipeline") {
#ifdef MINERU_HAVE_PIPELINE
    if (pdf_path.empty()) { std::cerr << "--path is required for the pipeline backend\n"; return 2; }
    return run_pipeline(pdf_path, pipeline_models, out_dir, start_page, end_page, pipeline_dpi);
#else
    std::cerr << "pipeline backend not built (ONNX Runtime unavailable)\n";
    return 2;
#endif
  } else if (backend != "vlm") {
    std::cerr << "unknown backend '" << backend << "' (use vlm | pipeline)\n";
    return 2;
  }

  auto t0 = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] loading model (quantize " << (bits ? std::to_string(bits) + "-bit" : "off")
            << ") ...\n";
  mineru::Qwen2Tokenizer tok = mineru::Qwen2Tokenizer::load(model_dir);
  mineru::Qwen2VLConfig cfg;
  cfg.quantize_bits = bits;
  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(model_dir + "/model.safetensors", cfg);
  auto t_loaded = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] model loaded in "
            << std::chrono::duration<double>(t_loaded - t0).count() << "s\n";

  if (server) return run_server(model, tok, host, port);

  if (pdf_path.empty()) { std::cerr << "--path is required (or use --server)\n"; return 2; }
  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf_path);
  int npages = doc.page_count();
  int s = std::clamp(start_page, 0, npages - 1);
  int e = (end_page < 0) ? npages - 1 : std::clamp(end_page, s, npages - 1);
  std::cerr << "[mlx-mineru] " << pdf_path << ": pages " << s << ".." << e << " of " << npages << "\n";

  namespace fs = std::filesystem;
  std::string stem = fs::path(pdf_path).stem().string();
  fs::path dir = fs::path(out_dir) / stem / "vlm";
  fs::create_directories(dir);
  std::string images_dir;
  if (!layout_only) {
    images_dir = (dir / "images").string();
    fs::create_directories(images_dir);
  }

  g_prof.load = std::chrono::duration<double>(t_loaded - t0).count();
  json pdf_info = json::array();
  json layout_json = json::array();
  if (layout_only) {
    for (int p = s; p <= e; ++p) {
      auto _r0 = Clock::now();
      mineru::PageImage pg = doc.render_page(p);
      g_prof.raster += secs(_r0, Clock::now());
      std::vector<mineru::ContentBlock> blocks;
      process_page(model, tok, pg, p, /*layout_only=*/true, &blocks, images_dir, batch_size);
      json jb = json::array();
      for (auto& b : blocks) {
        json o = {{"type", b.type}, {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}}};
        if (b.angle) o["angle"] = *b.angle;
        jb.push_back(o);
      }
      layout_json.push_back({{"page_idx", p}, {"blocks", jb}});
    }
  } else {
    // Render all pages, then batch layout detection across them.
    std::vector<mineru::PageImage> pages;
    std::vector<int> idxs;
    for (int p = s; p <= e; ++p) {
      auto _r0 = Clock::now();
      pages.push_back(doc.render_page(p));
      g_prof.raster += secs(_r0, Clock::now());
      idxs.push_back(p);
    }
    pdf_info = convert_batched(model, tok, pages, idxs, images_dir, batch_size, no_image_rec);
  }

  auto write = [&](const std::string& fname, const std::string& data) {
    std::ofstream(dir / fname) << data;
    std::cerr << "[mlx-mineru] wrote " << (dir / fname).string() << "\n";
  };

  auto _a0 = Clock::now();
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
  g_prof.assembly += secs(_a0, Clock::now());

  auto t1 = std::chrono::steady_clock::now();
  double infer = std::chrono::duration<double>(t1 - t_loaded).count();
  std::cerr << "[mlx-mineru] inference in " << infer << "s; total "
            << std::chrono::duration<double>(t1 - t0).count() << "s\n";

  // ---- Detailed phase breakdown -------------------------------------------
  auto& P = g_prof;
  double wall = std::chrono::duration<double>(t1 - t0).count();
  auto row = [&](const char* name, double s, const std::string& extra = "") {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "  %-22s %7.2fs  %5.1f%%   %s", name, s,
                  wall > 0 ? 100.0 * s / wall : 0.0, extra.c_str());
    std::cerr << buf << "\n";
  };
  auto tps = [](long t, double s) {
    char b[64];
    std::snprintf(b, sizeof(b), "%ld tok, %.0f tok/s", t, s > 0 ? t / s : 0.0);
    return std::string(b);
  };
  std::cerr << "\n=== profile (" << P.pages << " page(s), " << P.blocks << " blocks) ===\n";
  std::cerr << "  phase                    time    share   detail\n";
  row("model load", P.load, "weights mmap + quantize");
  row("pdf rasterize", P.raster, "pdfium");
  row("layout: vision", P.layout_vision, "page ViT encode");
  row("layout: generate", P.layout_gen, tps(P.layout_tok, P.layout_gen));
  row("content: vision", P.content_vision, std::to_string(P.blocks) + " crops ViT encode");
  row("content: generate", P.content_gen, tps(P.content_tok, P.content_gen));
  row("output assembly", P.assembly, "union_make + write");
  double accounted = P.load + P.raster + P.layout_vision + P.layout_gen + P.content_vision +
                     P.content_gen + P.assembly;
  row("other/overhead", wall - accounted, "tokenize, json, crops, sync");
  std::cerr << "  ----------------------------------------------\n";
  row("TOTAL (wall)", wall);
  return 0;
}
