# 忠实度审查：mlx-mineru ↔ MinerU 3.4.0

逐引擎对比当前 C++/MLX 实现与源项目（`~/research/MinerU`）的处理流程、阶段数据、文件输出、
三方库版本。原则：**除 pytorch→MLX、Python→C++ 外，应忠实于源项目实现**。

图例：✅ 已对齐 / 🟡 部分对齐或近似 / ⬜ 未实现（已记录原因） / 🔧 本轮修复。

---

## 0. 三方库版本（与源项目可见模块对齐）

| 库 | MinerU（源） | 本项目 | 状态 |
|---|---|---|---|
| pdfium（原生） | pypdfium2 4.30.0 → build **6462**（126.0.6462.0） | **6462**（`fetch_pdfium.sh` 固定 `chromium/6462`）🔧 | ✅ 取字逐字节一致 |
| onnxruntime | `>1.17.0` | 1.26.0 | ✅ 兼容 |
| OpenCV | `opencv-python>=4.11` | Homebrew 4.13.0（真实 cv2，部分阶段直接调用） | ✅ 兼容 |
| MLX | `mlx<=0.31.1` | 跟随本机 pip mlx | 🟡 未固定（建议 pin） |

> 关键修复：此前 pdfium 取 `latest`(7906)，导致数字 PDF 取字码位与 MinerU 不同（康熙部首 vs
> 统一汉字）。固定到 6462 后 `pipeline_digital` 由 4/19 → **19/19 逐字节一致**。
> 残留：pdfium *光栅化* 仍有 ~8% 边缘抗锯齿差异（渲染 flag，非版本），仅影响 VLM/MFR 个别歧义字形。

---

## 1. pipeline backend

| 阶段 | MinerU | 本项目 | 状态 |
|---|---|---|---|
| 版面检测（PP-DocLayoutV2，含阅读序） | batch_analyze 版面推理 | `LayoutDetector`（layout.onnx，硬编码标签表） | ✅ |
| 公式 MFD/MFR | display/inline_formula→UniMERNet | `FormulaRecognizer`（mfr_*.onnx） | ✅ 逐位一致 |
| OCR det/rec | DBNet + SVTR_LCNet（PP-OCRv6） | `TextDetector`/`TextRecognizer`，含 batch、sorted/merge | ✅（cv2 几何逐位一致） |
| 表格 cls→rec | 方向/有无线分类 → SLANet+ 或 UNet | 无线 SLANet+ ✅ / 有线 UNet ✅（5 阶段）/ **分类路由未接** | 🟡 未跑 TableClassifier 路由 |
| 数字取字 / OCR 选择 | `_get_ocr_enable`（auto/ocr） | `is_ocr` 选项 🔧 + 数字层 `fill_chars_in_page` / 回落 OCR | ✅ |
| middle_json（magic_model） | result_to_middle_json（span 预处理、阅读序、para_split、视觉块嵌套、丢弃块） | `pipeline_assemble.cpp`（贪心 span 匹配、classify_visual_blocks、para_split、formula_number→\tag） | 🟡 见下「未实现」 |
| union_make | pipeline_middle_json_mkcontent | `mkcontent.cpp`（VLM 版 union_make，签名带 formula/table_enable） | 🟡 见 §4 |
| 选项 formula/table_enable, lang, is_ocr | batch 内 gating / env | 🔧 `ConvertOpts` 全部接通（withhold 识别器 / union_make flag / 强制 OCR） | ✅ |

**未实现（已记录）**：
- ⬜ `cross_page_table_merge`（跨页表格合并）— finalize_middle_json 步骤。
- ⬜ `apply_title_leveling`（标题分级）— 当前 paragraph_title 一律 level 2。
- ⬜ 表格方向/有无线**分类路由**（TableClassifier 已实现但未在 driver 中按分类选 SLANet/UNet）。
- 🟡 magic_model 的复杂 span 关联用「贪心重叠匹配」近似（a.pdf/demo 验证下与 MinerU 一致）。

---

## 2. vlm backend（Qwen2-VL / MinerU2.5）

| 项 | MinerU | 本项目 | 状态 |
|---|---|---|---|
| 提示词（layout + 每块指令） | `\nLayout Detection:` / `\nText|Table|Formula Recognition:` / `\nImage Analysis:` | `instruction_for` / build_prompt 逐字一致 | ✅ |
| 两步流程 | 整页版面 → 每块裁剪 → 内容生成 | `convert_batched`/`process_page`（批量 vision+生成，长度分桶） | ✅ |
| 块类型 / 版面正则 | mineru_vl_utils 正则 | `vlm_layout.cpp` 逐字节一致 | ✅ |
| OTSL→HTML / 公式后处理 | otsl2html / post_process | `otsl.cpp` / `post_process.cpp` | ✅ 等价 |
| 丢弃块（header/footer/…） | 一致 | 一致 | ✅ |
| image_analysis（跳过图理解） | batch_two_step_extract 参数 | `--no-image-rec` / `skip_image_rec` | ✅ |
| formula/table_enable | `MINERU_VLM_*_ENABLE` env → union_make | 🔧 `ConvertOpts`→union_make flag（web/server）；一次性 CLI 默认 on | 🟡 一次性 CLI 未加旗标 |
| 采样参数（每块 presence/frequency penalty） | mineru_client.py 每类不同 | 贪心解码（无采样调参） | 🟡 解码策略差异（MLX 等价范围内） |
| middle_json 后处理 | blocks_to_page_info + MagicModel | process_page 内联组装（简化） | 🟡 同 §1 |

---

## 3. hybrid-engine

源项目 hybrid = pipeline 结构 + VLM 关键区域理解。本项目 hybrid = **pipeline 全量结构
（版面/OCR/公式/表格）+ VLM 仅对 image/chart 裁剪做理解**（`image_analysis` 控制）。
🟡 近似：源项目 hybrid 的 `effort`(medium/high) 与具体融合策略更细；本项目 effort 已在 UI 暴露
但当前 hybrid 恒用 VLM 处理图像块。

---

## 4. union_make / Markdown 渲染

- 源项目：pipeline 与 vlm 各有一份 `*_middle_json_mkcontent.py`，签名 `union_make(pdf_info,
  make_mode, img_bucket)`；vlm 版从 `MINERU_VLM_*_ENABLE` env 读 formula/table_enable。
- 本项目：单一 `mkcontent.cpp`（移植 vlm 版），签名额外带 `formula_enable/table_enable` 形参，
  web/server 路径已按 `ConvertOpts` 传入 ✅。content_list / content_list_v2 / mm / nlp 四模式齐备。
- 🟡 两份 mkcontent 的细微差异（pipeline 版 vs vlm 版）未逐行核对；对 text/title/table/formula/
  image 的渲染在 golden（test_mkcontent_golden）下逐字节一致。

---

## 5. 文件输出（do_parse 工件集）

| 文件 | MinerU 默认 | 本项目 | 状态 |
|---|---|---|---|
| `{name}.md` | ✅ | ✅ | ✅ |
| `{name}_content_list.json` | ✅ | ✅ | ✅ |
| `{name}_content_list_v2.json` | ✅ | 🔧 ✅ | ✅ |
| `{name}_middle.json` | ✅ | ✅ | ✅ |
| `{name}_model.json` | ✅ | 🔧 ✅（pipeline；结构同 model_list） | ✅ |
| `{name}_origin.pdf` | ✅ | 🔧 ✅ | ✅ |
| `{name}_layout.pdf` | ✅ | ⬜（需移植 draw_bbox.py） | ⬜ |
| `{name}_span.pdf` | ✅ | ⬜（需移植 draw_bbox.py） | ⬜ |
| `images/` | ✅ | ✅（VLM 落盘 / web 内联 base64 data URI，仿 `_encode_table_inline_image`） | ✅ |
| 目录结构 | `<out>/<name>/<parse_method>/` | `<out>/<stem>/{pipeline,vlm}/` | ✅ 等价 |

---

## 待办（按优先级）

1. ⬜ 表格分类路由（SLANet+ vs UNet 由 TableClassifier 选）—— 模型与两条路径都已就绪，只差接线。
2. ⬜ `cross_page_table_merge` + `apply_title_leveling`（finalize_middle_json 两步）。
3. ⬜ `_layout.pdf` / `_span.pdf`（移植 draw_bbox.py 的 PDF 注记绘制）。
4. 🟡 pin MLX 版本到 `<=0.31.1`。
5. 🟡 逐行核对 pipeline 版 union_make 与本项目 mkcontent 的差异。
6. 🟡 pdfium 光栅化 flag 对齐（关闭最后一个 VLM 歧义字形残差）。
