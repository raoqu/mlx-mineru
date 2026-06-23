# AGENT.md — mlx-mineru 工作指南

> 在本仓库工作的 AI/开发者**先读本文件**，再看 [GOAL.md](GOAL.md)（目标）与 [PLAN.md](PLAN.md)（计划）。
> 参考源项目：`~/research/MinerU`（Python，版本 `3.4.0`）。本项目是它的 **C++ 原生复刻**。

## 简要目标
用 C++ 复刻 MinerU 的文档解析（PDF/图片/Office → Markdown/JSON），**完全脱离 Python**，在 Apple Silicon 上用 **MLX/Metal** 加速推理。CLI + 可选 `--server` HTTP API。

## 核心原则（必须遵守）
1. **对齐优先**：严格遵循源项目逻辑与输出。依赖三方库时（mlx-vlm、mineru-vl-utils、pytorchocr/PaddleOCR、pdfium、safetensors…）**先取三方库源码对齐其实现方式**，无法对齐再迁移实现。
2. **中间表示即契约**：`middle_json` 贯穿全流程，保持**字段级兼容**。所有后端产出它，所有输出由它生成。
3. **垂直切片**：先打通端到端（Office 最简、无 ML），再加 ML 后端；每阶段对 Python 输出做 **golden diff** 验收。
4. **MLX/Metal**：张量计算走 MLX C++（mlx 本身是 C++ 库）；热点必要时写 Metal kernel。
5. **零 Python**：最终二进制不依赖 Python 解释器或 `.py`。
6. 改代码前先更新/参照 PLAN.md 的阶段；新发现的契约细节回写本文件或 PLAN.md。
7. **每个环节可验证**：每一步（不只每阶段）都必须有可执行的验证手段——能编译、能跑测试、能对 golden 做 diff。没有验证方式的改动不算完成。验证方式与结果要可复现（写进 `tests/` 或在提交信息里说明如何复跑）。
8. **每阶段提交 Git**：完成一个阶段（或一个可独立验证的子环节）后，立即 `git commit`，提交信息写清"做了什么 + 如何验证 + 验证结果"，使进度可追溯、可回滚。绝不在未验证通过时提交。

## 工作循环（每个环节都照此执行）
1. 对照 PLAN.md 选定当前环节，明确**验收标准**（对齐 Python 的哪个输出 / 哪个测试通过）。
2. 写代码。
3. **验证**：`cmake --build` 通过 + 运行该环节的测试 / golden diff 通过。
4. 验证通过 → `git commit`（信息含验证方式与结果）→ 更新本文件"当前状态"。
5. 进入下一环节。失败则修复后重验，不通过不提交、不前进。

## 架构速览（源项目数据流）
```
输入 → [office: zip+XML] / [pdf: pdfium@200DPI → 页图]
     → backend.doc_analyze → middle_json{ pdf_info:[ page{ para_blocks:[ block{ lines:[span] } ] } ] }
     → union_make(MakeMode) → .md / content_list.json / content_list_v2.json + images/
```
四后端：`office`(纯逻辑·P0) · `vlm`(单一 Qwen2-VL·MLX 旗舰·P1) · `pipeline`(7 个经典 CV 模型·P2) · `hybrid`(P3)。

## 关键事实卡（来自源码分析）

### 枚举（`mineru/utils/enum_class.py`）
- `MakeMode`: `mm_markdown` / `nlp_markdown` / `content_list` / `content_list_v2`
- `BlockType`: text, title, image/table/chart/code(+_body/_caption/_footnote), list, index, interline_equation, equation, header, footer, page_number, aside_text, ref_text, phonetic, discarded, abstract, doc_title, paragraph_title, vertical_text, formula_number …
- `ContentType`: text, inline_equation, interline_equation, image, table, chart, hyperlink
- `ContentTypeV2`: paragraph, title, list(text_list/reference_list), index, code, algorithm, equation_interline, image, table(simple/complex), span(text/equation_inline/phonetic/code_inline) …

### 模型仓库与路径（`ModelPath`）
- VLM：HF `opendatalab/MinerU2.5-Pro-2605-1.2B` / MS `OpenDataLab/MinerU2.5-Pro-2605-1.2B`（Qwen2-VL，~1.2B）
- Pipeline 根：HF `opendatalab/PDF-Extract-Kit-1.0` / MS 同名，子路径：
  - 版面 `models/Layout/PP-DocLayoutV2`（torch，RT-DETR/HGNetV2）
  - 公式 `models/MFR/unimernet_hf_small_2503`（torch，Swin+mBART）或 `models/MFR/pp_formulanet_plus_m`
  - OCR `models/OCR/paddleocr_torch`（torch；DBNet 检测 + CTC 识别，PP-OCRv6）
  - 表格 `models/TabRec/SlanetPlus/slanet-plus.onnx`、`models/TabRec/UnetStructure/unet.onnx`、`models/TabCls/paddle_table_cls/PP-LCNet_x1_0_table_cls.onnx`（**均为 ONNX**）
- **框架分布**：表格 3 件 = ONNX（用 ONNX Runtime C++）；版面/OCR/公式 = torch（在 MLX 重写）。

### 关键参数
- PDF 栅格化：pypdfium2，默认 **200 DPI**，`scale=dpi/72`，最大边 3500px。
- CLI 默认后端 `hybrid-engine`；本项目 MVP 主推 `vlm`(MLX) 与 `office`。
- 配置文件 `~/mineru.json`（env `MINERU_TOOLS_CONFIG_JSON`）；env：`MINERU_MODEL_SOURCE`、`MINERU_DEVICE_MODE`、`MINERU_FORMULA_ENABLE`、`MINERU_TABLE_ENABLE`、`MINERU_PROCESSING_WINDOW_SIZE`(64) 等。
- 输出布局：`<output>/<name>/<method>/{<name>.md, *_middle.json, *_model.json, *_content_list[_v2].json, images/}`。

### CLI flag（对齐 `cli/client.py`）
`-p/--path` `-o/--output` `-b/--backend` `-m/--method(auto|txt|ocr)` `-l/--lang(ch)` `-s/--start` `-e/--end` `-f/--formula` `-t/--table` `--effort(medium|high)` `--image-analysis` `-u/--url` `--api-url` `--client-side-output-generation` `-v/--version`。

### HTTP API（对齐 `cli/fast_api.py`，protocol v2）
`POST /file_parse`（multipart：files, lang_list, backend, parse_method, formula/table_enable, return_md/middle_json/content_list/images, response_format_zip…）、`POST /tasks`、`GET /tasks/{id}`、`GET /tasks/{id}/result`、`GET /health`。响应头 `X-MinerU-Task-{Id,Status,Status-Url,Result-Url}`。

## ⚠️ 已知阻塞 / 重要风险
- ~~`mineru-vl-utils` 与 `mlx-vlm` 源码不在磁盘上~~ **已解除**：`scripts/fetch_reference.sh` → `third_party/reference/`（gitignored）。Phase 4 两步抽取契约已记入 PLAN.md §3。
- Qwen2-VL 的 MLX C++ 实现 + Qwen2 BPE tokenizer 对齐是最大工作量与隐藏成本。
- 输出需对 Python 逐字节对齐：CJK 空格、markdown 转义、bbox 取整、`json.dumps(ensure_ascii=False, indent=4)`。

## 当前状态
- 已完成：对 MinerU 3.4.0 的源码分析；写入 GOAL.md / PLAN.md / AGENT.md。
- **Phase 0 ✅**：CMake 脚手架；`core/` 的 `middle_json` 类型化模型（Span/Line/Block/Page/MiddleJson，未知字段经 `extra` 无损保留）+ 枚举（`enums.hpp`）+ 无损 JSON 往返。验证：`./scripts/build_and_test.sh`（ctest `middle_json_roundtrip` 通过）。
- **Phase 1 ✅**：`union_make` 输出契约（`include/mineru/mkcontent.hpp` + `src/output/mkcontent.cpp` + `text_utils.*`），忠实移植 `vlm_middle_json_mkcontent.py` 的 4 种模式（mm/nlp markdown + content_list v1/v2）及全部 helper。验证：`scripts/gen_golden.py` 用**真实 Python union_make** 生成 golden（`tests/golden/`），C++ 输出做 diff（markdown 逐字符、content_list 语义 JSON 相等），ctest `mkcontent_golden` 通过。
- 验证基建：`scripts/build_and_test.sh` 一键 build+ctest；`scripts/gen_golden.py` 重生成 golden（需 `~/research/MinerU` + loguru + fast-langdetect）；测试用 `tests/test_util.hpp` 轻量 CHECK 宏。
- 三方库：`third_party/nlohmann/json.hpp` 3.12.0、`third_party/CLI11/CLI11.hpp`（vendored，单头）。
- **Phase 3 ✅**：PDF 栅格化（`include/mineru/pdf.hpp` + `src/io/pdf.cpp`），pdfium C API，忠实移植 `page_to_image`（scale=dpi/72，长边封顶 3500px，`ceil` 取整，RGB8）。验证：`scripts/gen_pdf_golden.py` 用真实 pypdfium2 生成 golden（`tests/golden/pdf_raster.json`），C++ 比对**逐页尺寸/scale/页面点数精确相等** + ink/亮度内容指标在跨构建容差内（pdfium 7906 vs pypdfium2 7891）。ctest `pdf_raster` 通过。
- 三方库（二进制，gitignore，用脚本取）：`scripts/fetch_pdfium.sh`（pdfium mac-arm64）。
- ⚠️ **已知近似**：(1) `detect_lang` 用 CJK 脚本启发式替代 fast-langdetect；(2) PDF 像素非逐字节对齐（pdfium 构建版本差异的抗锯齿），但尺寸契约精确对齐。
- **Phase 4a ✅**：MLX C++ 工具链（CMake 从 pip `mlx` 包定位 headers/libmlx/metallib，target `mlx`）+ Metal GPU smoke（ctest `mlx_smoke`）。
- **Phase 4b ✅**：Qwen2 字节级 BPE tokenizer（`include/mineru/tokenizer.hpp` + `src/vlm/tokenizer.cpp`），手写 Qwen2 预分词正则（Unicode L/N/空白表由 `scripts/gen_unicode_tables.py` 生成）。验证：`scripts/gen_tokenizer_golden.py` 用真实 transformers 生成 golden，C++ encode **逐 id 精确相等** + decode（含/不含 special）文本相等，ctest `tokenizer`（12 用例：缩写/CJK/逐位数字/多空格/换行/chat 标记）。
- 模型文件（gitignore）：`scripts/fetch_tokenizer.sh` 取 tokenizer/config（~16MB）；Qwen2-VL 配置见 `models/MinerU2.5-tokenizer/config.json`（LLM hidden 896 / 24 层 / GQA 14:2 / vocab 151936 / tie embeddings；vision depth 32 / 1280d / patch14 / merge2；贪心解码 top_k=1）。
- 🎯 **PDF → Markdown 主线**：栅格化(P3 ✅) → [图像预处理 4c → Qwen2-VL/MLX 4d → 两步抽取/输出解析/magic_model 4e] → middle_json → union_make(P1 ✅)。
- **Phase 4c ✅**：图像预处理（`include/mineru/image_preprocess.hpp`+`src/vlm/image_preprocess.cpp`）忠实移植 Qwen2VLImageProcessor（smart_resize 银行家舍入+min/max 像素夹取、PIL 8bit bicubic 两遍定点、CLIP 归一化、patchify reshape/transpose）。验证 ctest `preprocess`：smart_resize 精确、grid_thw 精确、pixel_values **逐采样 bit-exact**（max diff 0.0 vs transformers）。
- **Phase 4d-LLM ✅**：Qwen2-VL 语言模型 MLX C++（`include/mineru/qwen2_vl.hpp`+`src/vlm/qwen2_llm.cpp`），完整 decoder：GQA(14:2)、MRoPE(chunked [8,12,12]→rotate_half 按频率轴选择)、`fast::rms_norm`/`fast::scaled_dot_product_attention`、SwiGLU、tie embeddings；从真实 safetensors 加载（`mx::load_safetensors`）。验证 ctest `llm_forward`：vs transformers **首 token argmax 精确**、top10 logits 容差内、5 步贪心（近似平局 gap<0.25 容忍）。
- 权重（gitignore，2.2GB）：`scripts/fetch_weights.sh`。MLX target 由 CMake 从 pip `mlx` 定位。
- **Phase 4d-vision ✅**：视觉塔（patch_embed 以 matmul 实现、32 层 ViT 块带 2D-rope+full attention、quick_gelu MLP、PatchMerger）→ ctest `vision`（vs transformers，mean/std 0.1% 内、MAE/std 1.5%、248/256 采样达标）。
- **Phase 4d-multimodal ✅**：`get_rope_index`（单图 3D 位置）+ 视觉特征拼接进 image_token 流 + 贪心生成 → ctest `vlm`：完整 预处理→视觉→合并→LLM 端到端复现 transformers 12 步贪心生成。
- **Phase 4e-layout ✅**：版面输出解析器（`include/mineru/vlm_layout.hpp`+`src/vlm/vlm_layout.cpp`），`<|box_start|>…<|ref_*|>…` 文法 + bbox 0..1000→[0,1] + 表内块过滤，对齐 mineru-vl-utils → ctest `layout`。
- 🎯 **PDF→Markdown 进度**：栅格化(P3 ✅)→图像预处理(4c ✅)→Qwen2-VL 全模型 MLX C++(LLM✅/视觉✅/多模态✅，均已验证)→版面解析(4e-layout ✅)→[**剩余**：两步抽取编排+KV cache、每块内容抽取、post_process、MagicModel→middle_json、CLI]→union_make(P1 ✅)。
- **CLI ✅ PDF→Markdown 端到端跑通**：`src/cli/main.cpp` → `build/mlx-mineru`。`-p PDF --page N -o out.md`：渲染→layout 检测→**逐块裁剪+按类型 prompt 抽内容（两步抽取 step2）**→组装 middle_json→`union_make`→Markdown。`--layout-only` 仅输出版面 JSON。**实测 demo1.pdf p0：~9s 产出高质量 Markdown**（标题成 #、作者带行内 LaTeX 上标、`# Abstract`、含行内公式的摘要、Keywords）。零 Python。
- **KV cache ✅**：`forward_cached`/`generate_cached`（prefill+逐 token 解码），layout 由 38.3s→3.2s（~12×），输出一致。
- **剩余保真工作（非阻塞，已可用）**：(1) MagicModel(`vlm_magic_model.py` 856 行) 完整块归类/caption 关联/span 截图（当前简化：header/footer/page_number/page_footnote/aside→discarded，其余按主类型）；(2) `post_process/*`（otsl2html 表格、公式修复、cross-page table）；(3) image/chart 存图+image_path；(4) 多页/批处理、`--server` HTTP API、`content_list` 输出；(5) 与 MinerU Python 逐字节对齐（需可跑其 VLM）。

## 参考样本
`~/research/MinerU/demo/`：`pdfs/{demo1,demo2,demo3,small_ocr}.pdf`、`office_docs/{docx_01.docx,pptx_01.pptx,xlsx_01.xlsx}`——用作 golden 对比输入。
