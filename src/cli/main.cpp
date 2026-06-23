// Copyright (c) mlx-mineru.
// mlx-mineru CLI — native C++/MLX MinerU. Current pipeline: render a PDF page,
// run the Qwen2-VL layout-detection step on Apple Silicon (MLX/Metal), and emit
// the detected document layout. (Per-block content extraction + MagicModel ->
// middle_json -> Markdown are the in-progress next steps; see AGENT.md.)
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "CLI11/CLI11.hpp"
#include "mineru/image_preprocess.hpp"
#include "mineru/pdf.hpp"
#include "mineru/qwen2_vl.hpp"
#include "mineru/tokenizer.hpp"
#include "mineru/vlm_layout.hpp"
#include "nlohmann/json.hpp"

using nlohmann::json;

// Build the Qwen2-VL chat prompt for layout detection: system + user(image +
// "\nLayout Detection:") + assistant, with `n_img` image-pad placeholders.
static std::vector<int> build_layout_prompt(const mineru::Qwen2Tokenizer& tok, int n_img,
                                            int image_pad_id) {
  std::vector<int> ids = tok.encode(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
      "<|im_start|>user\n<|vision_start|>");
  ids.insert(ids.end(), n_img, image_pad_id);
  std::vector<int> suffix = tok.encode(
      "<|vision_end|>\nLayout Detection:<|im_end|>\n<|im_start|>assistant\n");
  ids.insert(ids.end(), suffix.begin(), suffix.end());
  return ids;
}

int main(int argc, char** argv) {
  CLI::App app{"mlx-mineru — native C++/MLX MinerU (PDF document parsing)"};
  std::string pdf_path, model_dir = "models/MinerU2.5-tokenizer", out_path;
  int page = 0, max_new = 1200;
  bool md_layout = false;
  app.add_option("-p,--path", pdf_path, "Input PDF path")->required();
  app.add_option("-m,--model", model_dir, "Model directory (weights + tokenizer)");
  app.add_option("--page", page, "0-based page index to process");
  app.add_option("-o,--output", out_path, "Write layout JSON to this file (default: stdout)");
  app.add_option("--max-new", max_new, "Max layout tokens to generate");
  CLI11_PARSE(app, argc, argv);

  auto t0 = std::chrono::steady_clock::now();
  std::cerr << "[mlx-mineru] loading model from " << model_dir << " ...\n";
  mineru::Qwen2Tokenizer tok = mineru::Qwen2Tokenizer::load(model_dir);
  mineru::Qwen2VLModel model = mineru::Qwen2VLModel::load(model_dir + "/model.safetensors");
  const auto& cfg = model.config();

  std::cerr << "[mlx-mineru] rendering page " << page << " of " << pdf_path << " ...\n";
  mineru::PdfDocument doc = mineru::PdfDocument::open_file(pdf_path);
  if (page < 0 || page >= doc.page_count()) {
    std::cerr << "page out of range (0.." << doc.page_count() - 1 << ")\n";
    return 2;
  }
  mineru::PageImage pg = doc.render_page(page);

  // prepare_for_layout: resize the page to 1036x1036 (BICUBIC), then preprocess.
  const int kLayoutSize = 1036;
  std::vector<uint8_t> resized =
      mineru::resize_bicubic_rgb8(pg.rgb, pg.width, pg.height, kLayoutSize, kLayoutSize);
  mineru::VisionInput vi = mineru::preprocess_image(resized, kLayoutSize, kLayoutSize);
  int n_img = vi.seq_len() / (cfg.spatial_merge_size * cfg.spatial_merge_size);
  std::cerr << "[mlx-mineru] vision grid " << vi.grid_thw[0] << "x" << vi.grid_thw[1] << "x"
            << vi.grid_thw[2] << " -> " << n_img << " image tokens; running vision tower ...\n";

  std::vector<float> embeds = model.forward_vision(vi.pixel_values, vi.grid_thw);

  std::vector<int> prompt = build_layout_prompt(tok, n_img, cfg.image_token_id);
  std::cerr << "[mlx-mineru] generating layout (" << prompt.size()
            << " prompt tokens; this is the slow KV-cache-less path) ...\n";
  std::vector<int> gen = model.generate_multimodal(prompt, embeds, n_img, vi.grid_thw, max_new,
                                                   {151645, 151643});
  std::string layout_text = tok.decode(gen, /*skip_special=*/false);

  std::vector<mineru::ContentBlock> blocks = mineru::parse_layout_output(layout_text);

  json out = json::object();
  out["page_idx"] = page;
  out["page_size_pt"] = {pg.width_pt, pg.height_pt};
  out["num_blocks"] = blocks.size();
  out["blocks"] = json::array();
  for (const auto& b : blocks) {
    json jb = {{"type", b.type},
               {"bbox", {b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]}},
               {"merge_prev", b.merge_prev}};
    if (b.angle) jb["angle"] = *b.angle;
    out["blocks"].push_back(jb);
  }
  auto t1 = std::chrono::steady_clock::now();
  out["elapsed_s"] = std::chrono::duration<double>(t1 - t0).count();

  std::string dumped = out.dump(2);
  if (out_path.empty()) {
    std::cout << dumped << "\n";
  } else {
    std::ofstream(out_path) << dumped;
    std::cerr << "[mlx-mineru] wrote " << out_path << "\n";
  }
  std::cerr << "[mlx-mineru] done: " << blocks.size() << " blocks in "
            << out["elapsed_s"].get<double>() << "s\n";
  return 0;
}
