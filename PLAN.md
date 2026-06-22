# PLAN — mlx-mineru 实施计划

本计划基于对 `~/research/MinerU`（3.4.0）的源码分析。先读 [GOAL.md](GOAL.md) 与 [AGENT.md](AGENT.md)。

---

## 0. 源项目数据流（必须吃透）

```
输入字节 (pdf/img/office)
  → read_fn / guess_suffix            # 识别类型；图片→单页 PDF
  → [office]  zip+XML 解析 ────────────┐
  → [pdf]     pdfium 栅格化@200DPI → 页图 ┤
                                        ▼
                         backend.doc_analyze(...)
   ├ office:   converter → blocks → result_to_middle_json
   ├ vlm:      Qwen2-VL.batch_two_step_extract(页图) → 块语法文本
   │           → vlm_magic_model 解析 → model_output_to_middle_json
   └ pipeline: 版面检测 → (每区域) OCR/公式/表格 → magic_model → para_split
                                        ▼
                          middle_json  { pdf_info:[ page{ para_blocks:[ block{ lines:[ span ] } ] } ] }
                                        ▼
              union_make(middle_json, MakeMode, image_dir)   # mkcontent
   ├ MM_MD / NLP_MD       → Markdown 字符串
   ├ CONTENT_LIST         → 扁平结构化 JSON (v1)
   └ CONTENT_LIST_V2      → 分页 span 层级 JSON (v2)
                                        ▼
                _process_output → 写 .md / *.json / images/ / 可视化 PDF
```

**关键契约对象 `middle_json`：**
```
middle_json = {
  "pdf_info": [ page, ... ],
  "_backend": "pipeline"|"vlm"|"office"|...,
  "_version_name": "3.4.0"
}
page = {
  "page_idx": int,
  "page_size": [w, h],                 # 像素/点
  "para_blocks": [ block, ... ],       # 段落级块（阅读顺序，含 index）
  "discarded_blocks": [ block, ... ],  # 页眉/页脚/页码等
  "preproc_blocks": [...]              # pipeline 段前原始块
}
block = {
  "type": BlockType,                   # text/title/image/table/chart/code/list/index/equation...
  "bbox": [x0,y0,x1,y1],
  "index": int,                        # 阅读顺序
  "level": int,                        # 标题层级 1..n
  "lines": [ { "bbox":[...], "spans":[ span ] } ],
  "blocks": [ 子块 ],                   # 复合块：image_body+caption+footnote 等
  "sub_type": str, "guess_lang": str, ...
}
span = {
  "type": ContentType,                 # text/inline_equation/interline_equation/image/table/chart/hyperlink
  "bbox": [...],
  "content": str,                      # 文本 / LaTeX / HTML(表格)
  "image_path": str, "html": str, "style": [...], "url": str
}
```
枚举权威定义见 `mineru/utils/enum_class.py`（已复制要点到 AGENT.md）。

---

## 1. 技术栈与第三方库

| 关注点 | 选型 | 理由 / 对应 Python 依赖 |
|--------|------|------------------------|
| 构建 | CMake + Ninja | — |
| 张量/推理 | **MLX (C++)** `ml-explore/mlx` | mlx 本身是 C++ 库，替代 mlx/mlx-vlm；Metal 加速 |
| PDF 栅格化 | **PDFium (C++ API)** | pypdfium2 即 pdfium 封装，原生 C++，直接用 |
| 图像处理 | OpenCV（pipeline CV 预处理）/ stb_image（轻量解码） | opencv-python / pillow |
| ONNX 模型 | ONNX Runtime (C++) | pipeline 表格模型为 .onnx |
| JSON | nlohmann/json | json 序列化 |
| XML | pugixml 或 libxml2 | lxml / python-docx oxml |
| ZIP | libzip 或 miniz | zipfile（office/safetensors 容器） |
| HTTP 服务 | cpp-httplib 或 Crow | fastapi + uvicorn |
| HTTP 客户端 | libcurl | requests/httpx（模型下载） |
| CLI 解析 | CLI11 | click |
| 分词器 | tokenizers-cpp (HF) 或自实现 Qwen2 BPE | transformers tokenizer |
| 权重格式 | safetensors（自解析 header+mmap）| safetensors |
| 语言检测 | 移植 fast-langdetect 思路 / 简化 | fast-langdetect |
| 日志 | spdlog | loguru |

> **原则**：每个三方库先取其源码/头文件对齐 MinerU 的调用方式（尤其 pdfium 的 DPI/scale、Qwen2-VL 的图像预处理、safetensors 的张量布局），再决定直接链接还是迁移实现。

---

## 2. 目标目录结构

```
mlx-mineru/
  CMakeLists.txt
  third_party/            # mlx, pdfium, onnxruntime, nlohmann_json, pugixml, miniz, cpp-httplib, CLI11, ...
  include/mineru/
  src/
    core/                 # middle_json 数据模型、枚举、版本
    io/                   # 文件读写抽象、pdfium 栅格化、图像工具、suffix 嗅探
    output/               # mkcontent：middle_json -> md / content_list / content_list_v2 (union_make 对齐)
    office/               # docx/pptx/xlsx 解析 + office magic_model + result_to_middle_json
    backend/
      vlm/                # vlm_analyze / vlm_magic_model / model_output_to_middle_json / finalize
      pipeline/           # (P2) batch_analyze / magic_model / para_split
    model/
      vlm/                # Qwen2-VL（视觉编码器 + Qwen2 解码器）、tokenizer、image_processor
      layout/ ocr/ mfr/ table/   # (P2) 经典模型
    download/             # HF/ModelScope 模型拉取（libcurl）
    cli/                  # CLI11 入口 + --server (httplib)，对齐 client.py / fast_api.py
  models/                 # 运行时下载缓存
  tests/                  # golden 对比测试（对 Python 输出）
  GOAL.md  PLAN.md  AGENT.md
```

---

## 3. 分阶段实施（垂直切片 + golden 对齐）

> 每个阶段结束前，用 Python MinerU 生成 golden 产物存入 `tests/golden/`，C++ 输出做 diff 验收。

### Phase 0 — 脚手架与核心数据模型
- CMake 工程、third_party 接入（先 nlohmann/json、CLI11、spdlog）。
- `core/`：`middle_json` 的 C++ 结构体（Page/Block/Line/Span）、`BlockType`/`ContentType`/`ContentTypeV2`/`MakeMode`/`ModelPath` 枚举（照抄 `enum_class.py`）、JSON 读写。
- `io/`：`FileBasedDataReader/Writer`、输出目录布局（`output/<name>/<method>/...`）、`guess_suffix_by_bytes`。
- **验收**：能读入一个 Python 产出的 `*_middle.json` 并无损序列化回去。

### Phase 1 — 输出契约 mkcontent（最关键，先锁定）
- 移植 `vlm_middle_json_mkcontent.py` 与 `pipeline_middle_json_mkcontent.py` 的 `union_make` 及 `make_blocks_to_content_list[_v2]`。
- 规则要点（详见两文件）：
  - 标题：`level` → `#` 数量（>n 截断）；行首 markdown 符号转义。
  - 表格：`html` 原样插入；`<eq>`→`$...$`；`img src` 前缀 `image_dir/`。
  - 公式：行内 `$...$`、行间 `\n$$\n...\n$$\n`（分隔符可配 `latex-delimiter-config`）。
  - 图片：`![](image_dir/xxx.jpg)`；复合块按 caption/footnote 组装。
  - 列表/代码/CJK-vs-西文空格/连字符换行/markdown 转义等。
- **验收**：对多份 golden `middle.json`，C++ 的 `.md`/`content_list`/`content_list_v2` 与 Python 完全一致。

### Phase 2 — Office 后端（首个端到端，纯逻辑无 ML）
- `office/`：docx/pptx/xlsx 解析。底层为 **zip 解压 + XML 遍历**（对照 `mineru/model/{docx,pptx,xlsx}/` 与 `backend/office/*`）。
  - docx：段落/run/样式、列表(numId/ilevel)、表格(HTML)、OMML 数学→LaTeX（移植 `tools/math/omml.py` + `latex_dict.py`）、图片(base64)、图表(OOXML)。
  - pptx：逐 slide 的 shapes、表格、图片、XY-cut 排序（移植 `xycut_pp_sorter.py`）。
  - xlsx：逐 sheet 单元格→HTML 表格、共享字符串、内嵌图片。
  - `office_magic_model.py`：caption 归类、span 解析、list/index、表格 HTML 清洗、两层块配对。
  - `result_to_middle_json`：标题编号、anchor 校验。
- **验收**：`demo/office_docs/{docx_01,pptx_01,xlsx_01}` 的 md/json 与 Python 对齐。这条切片同时验证 Phase 1 的输出契约。

### Phase 3 — PDF 栅格化与图像基础设施
- `io/pdf`：PDFium C++ 加载 PDF、按 200 DPI（`scale=dpi/72`，上限 3500px）渲染为位图 → 图像缓冲。对齐 `pdf_image_tools.py`/`pdf_reader.py`。
- pdfium 损坏页跳过/重写逻辑（对齐 `pdfium_guard.py`/`convert_pdf_bytes_to_bytes`）。
- 图像裁切/保存（`cut_image`）、bbox 工具（`bbox_utils`/`boxbase`）。
- **验收**：渲染页图与 Python pypdfium2 像素级一致；裁图命名与路径一致。

### Phase 4 — VLM 后端（旗舰，MLX/Metal）
> ⚠️ **关键依赖缺口**：VLM 的提示词、`batch_two_step_extract` 两步抽取流程、输出块语法解码都在外部包 **`mineru-vl-utils`**（以及 `mlx-vlm`）中，**当前磁盘上未安装**。必须先从 PyPI/GitHub 取得这两个包的源码作为对齐基准（见 §5 风险）。

- **4a 模型架构**：在 MLX C++ 实现 `MinerU2.5-Pro-2605-1.2B`（Qwen2-VL 系）：
  - 视觉编码器（ViT，patchify + window attention）+ Qwen2 LLM 解码器 + 视觉-文本投影。
  - 参照 `ml-explore/mlx-examples` 与 `mlx-vlm` 的 qwen2-vl 实现对齐。
- **4b 预处理/分词**：Qwen2-VL 图像处理（smart-resize、patch 化、归一化）；Qwen2 BPE tokenizer（含视觉占位 token）。
- **4c 权重加载**：从 safetensors 读取（自解析 header + mmap，按需量化 4/8-bit 对齐 mlx-vlm）。
- **4d 推理与解码**：构造两步抽取 prompt → 自回归生成（贪心/采样）→ 得到块语法文本；解码循环走 MLX/Metal。**MLX 路径需串行执行锁**（对齐 `vlm_analyze.py` 的 `_maybe_enable_serial_execution`）。
- **4e 解析与组装**：移植 `vlm_magic_model.py`（块语法→归一化块、bbox 反归一化、行内 `\(...\)` 公式抽取、复合块归组）与 `model_output_to_middle_json.py` + `finalize_middle_json`。
- **验收**：`demo/pdfs/*.pdf` 的 `middle.json` 结构/文本与 Python `vlm-engine` 对齐；性能 ≥ Python `mlx` 后端。

### Phase 5 — CLI 与 --server
- CLI（CLI11）对齐 `client.py` 全部 flag：`-p/-o/-b/-m/-l/-s/-e/-f/-t/--effort/--image-analysis/-u/--api-url/...`（清单见 AGENT.md）。
- `--server`（cpp-httplib）对齐 `fast_api.py`：`POST /file_parse`、`POST /tasks`、`GET /tasks/{id}`、`GET /tasks/{id}/result`、`GET /health`；任务队列与状态机、ZIP 打包响应、`X-MinerU-Task-*` 头。
- **验收**：CLI/HTTP 行为与 Python 对齐（同输入同输出）。

### Phase 6 — 模型下载
- `download/`：从 HuggingFace / ModelScope 拉取并缓存（libcurl），对齐 `models_download_utils.py` 的 repo id、子路径、`model-source=auto/hf/modelscope/local`、本地缓存优先。
- repo/路径清单见 AGENT.md（`ModelPath`）。

### Phase 7 — Pipeline 后端（P2，最重，stretch）
- 表格三件套（SLANet++ / UNet / table-cls）为 **.onnx** → 直接用 ONNX Runtime C++（无需重写）。
- 版面 `PP-DocLayoutV2`（RT-DETR/HGNetV2）、OCR（DBNet 检测 + CTC 识别，pytorchocr 系）、公式（UniMERNet：Swin+mBART / PP-FormulaNet）为 **torch** → 在 MLX C++ 重写，源码对照 `mineru/model/utils/pytorchocr/` 与各 `predict_*.py`。
- 组装：`pipeline_magic_model.py` + `para_split.py`（阅读顺序、段落切分、跨页表格合并）。
- **验收**：`demo/pdfs/*.pdf` 用 `pipeline` 后端与 Python 对齐。

---

## 4. 里程碑顺序与依赖

```
P0 核心模型 ─┬─> P1 输出契约 ─┬─> P2 Office 端到端 ✅ 首个可交付
             │                └─> P3 PDF 栅格化 ──> P4 VLM(MLX) ✅ 旗舰
             └────────────────────────────────────> P5 CLI/Server
                                                     P6 模型下载（P4 前置依赖）
                                                     P7 Pipeline（最后）
```
建议交付节奏：**P0→P1→P2** 拿到第一个端到端（Office）→ **P6→P3→P4** 拿到 VLM(MLX) 旗舰 → **P5** 补齐 CLI/Server → **P7** 视需要补 pipeline。

## 5. 风险与未决项

1. **🔴 `mineru-vl-utils` / `mlx-vlm` 源码缺失**：磁盘未安装，且 VLM 的 prompt、两步抽取、输出语法解码都在其中。**行动**：实现 P4 前先 `pip download` 或克隆 GitHub 取源对齐；在此之前 P4 无法精确动工。
2. **Qwen2-VL 在 MLX C++ 的工作量大**：视觉编码器 + LLM 解码器 + 量化，需对照 mlx-examples。tokenizer 对齐是隐藏成本。
3. **ONNX Runtime 依赖**：pipeline 表格模型需要；评估其 C++ 包体积与 CoreML EP 可用性。
4. **输出逐字节对齐难点**：CJK 空格、转义、浮点 bbox 取整、JSON `ensure_ascii=False, indent=4` 等细节需精确复刻。
5. **pdfium / onnxruntime 预编译产物**：优先取官方 prebuilt（Apple Silicon arm64），避免自行编译。
6. **模型许可与下载**：MinerU2.5 模型许可、HF/ModelScope 在网络环境下的可达性。

## 6. 立即下一步（开始编码前）

1. 搭 Phase 0 脚手架（CMake + core 数据模型 + 枚举），用一份 golden `middle.json` 跑通"读入→序列化回写"。
2. 取 `mineru-vl-utils`、`mlx-vlm` 源码（解封 P4 的关键阻塞）。
3. 用 Python MinerU 对 `demo/` 全量样本生成 golden 产物，建 `tests/golden/` 基线。
