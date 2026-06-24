# GAP — mlx-mineru vs MinerU 3.4.0（重新对齐）

对照 `~/research/MinerU`（v3.4.0）梳理当前实现的差距。
图例：✅ 已对齐并验证 · 🟡 部分/简化 · ❌ 未实现。

> 现状一句话：**VLM 后端**（Qwen2-VL）与 **pipeline 后端**（layout/OCR/formula/table →
> middle_json → Markdown）的核心链路均已原生 C++/ONNX 实现并逐级对齐 MinerU；剩余差距
> 集中在 pipeline 的"非纯文本路径"（数字 PDF 取字、可视块装配、有线表格）、装配细节
> （para_split/标题分级）、以及输入/后端广度（图片、Office、hybrid、多语言）。

---

## 1. 后端

| 后端 | MinerU | mlx-mineru | 说明 |
|---|---|---|---|
| `vlm`（Qwen2-VL） | ✅ | ✅ 原生 MLX C++ | 输出契约/光栅化/预处理/分词/OTSL→HTML 已验证 |
| `pipeline`（layout/MFR/OCR/table 经典 CV） | ✅ | 🟡 **本会话新建，核心已通** | 见 §3 逐项 |
| `hybrid`（pipeline+vlm） | ✅（CLI 默认） | ❌ | — |
| `office`（docx/pptx/xlsx） | ✅ | ❌ | 纯 zip+XML 解析，无 ML |

## 2. 输入格式

| 输入 | MinerU | mlx-mineru |
|---|---|---|
| PDF | ✅ | ✅（pdfium，光栅化逐像素一致） |
| 图片（png/jpg/webp/tiff…） | ✅ | ❌ |
| Office（docx/pptx/xlsx） | ✅ | ❌ |

## 3. Pipeline 后端逐阶段

| 阶段 | MinerU | mlx-mineru | 验证 |
|---|---|---|---|
| 版面检测 PP-DocLayoutV2 | ✅ | ✅ | 框/类/score 对齐核心后处理 golden |
| 阅读顺序头（order head 解码） | ✅ | ✅ | a.pdf p0 10/10 顺序一致，框 ≤3px |
| OCR 检测（DBNet） | ✅ | ✅ | 19/19 框 ≤1px |
| OCR 识别（CTC SVTR） | ✅ | ✅ | 文本逐字一致 |
| OCR 整链（sort/merge/crop/batch） | ✅ | ✅ | 19/19 行，18/19 文本精确 |
| 公式识别 MFR（UniMERNet） | ✅ | ✅ | greedy 逐 token bit-exact |
| 表格分类（有/无线） | ✅ | ✅ | argmax 一致 |
| 无线表格结构 SLANet+ → HTML | ✅ | ✅ | 结构 token 精确，HTML 与 MinerU 一致 |
| 装配 MagicModel（文本/标题路径） | ✅ | ✅ | a.pdf p0 全 10 块结构精确 |
| 后置 OCR 取字（post-OCR text-fill） | ✅ | ✅ | 全 span 填充，ASCII 精确 |
| 多页驱动 + union_make → Markdown | ✅ | ✅ | 整篇 a.pdf ASCII 精确 |
| 从零生成 model_list（无 MinerU 依赖） | ✅ | ✅ | render→layout+OCR det→md 精确 |
| **数字 PDF 取字（pdftext / fill_char_in_spans）** | ✅ | ✅ | `PdfDocument::extract_chars`（pdfium loose char box）+ `fill_chars_in_page`（calculate_char_in_span + chars_to_content）；driver/CLI 有嵌入文本则走数字路径、否则 OCR，空 span 回落 OCR。a.pdf p0 与 MinerU **19/19 部首等价**（结构/间距/标点一致；仅康熙部首↔统一汉字码位差，源于 vendored pdfium 与 pypdfium2 版本差异，二者 NFKC 等价） |
| **可视块装配**（image/table/formula → span/块） | ✅ | ❌ | 识别器已就绪，但未接入 `__build_page_blocks`/`cut_image_and_table` |
| **有线表格结构（UNet）** | ✅ | ✅ | `WiredTableRecognizer`（链接 Homebrew OpenCV 4.13.0，与 MinerU 同一 cv2，逐位一致）。Stage 1 推理（preprocess+unet.onnx→线分割，**0% 像素差**）；Stage 2 线提取→单元格（cv::morphologyEx/connectedComponentsWithStats/minAreaRect/line + 全线延长，**19/19 格 ≤1px**）；Stage 3 TableRecover（→逻辑网格 logi_points，**19/19 完全一致**，含合并表头 colspan 与边缘细条）；Stage 4 plot_html_table（噪声边裁剪 + rowspan/colspan，结构 HTML 逐字一致）；Stage 5 match_ocr_cell（OCR→单元格 + 同行合并，含文字 HTML 逐字一致）。`ctest wired_table` 端到端与 MinerU 完全一致。**说明**：依据"优先用三方库源码"指引，线提取/形态学等直接调用真实 OpenCV，避免自实现方差。 |
| **表格方向分类（TableOrientationCls）** | ✅ | ❌ | — |
| **行内公式 mask（OCR det 前遮挡）** | ✅ | ❌ | `mask_formula_regions_for_ocr_det` |
| **版面启发式过滤层**（框抑制/公式去重/重排 ~500 行） | ✅ | 🟡 | 简单单栏可行；复杂重叠版面未覆盖 |

## 4. 装配/后处理细节

| 项 | MinerU | mlx-mineru |
|---|---|---|
| `para_split`：bbox_fs | ✅ | ✅ |
| `para_split`：list/index 检测（狗牙状/顶格） | ✅ | ❌ |
| `para_split`：title/equation 分组、跨块段落合并 | ✅ | ❌ |
| 标题分级：doc_title→L1 / paragraph_title→L2 | ✅ | ✅ |
| 标题分级：LLM-aided 层级推断 | ✅（可选） | ❌ |
| `optimize_formula_number_blocks` / `cross_page_table_merge` | ✅ | ❌ |
| discarded（页眉/页脚/页码/边注）分流 | ✅ | ✅（分流逻辑在；未端到端验证） |
| 弃用低置信度/小宽度 OCR（OcrConfidence） | ✅ | 🟡 仅 drop_score 0.5 |

## 5. 输出/语言/工程

| 项 | MinerU | mlx-mineru |
|---|---|---|
| Markdown（mm/nlp） | ✅ | ✅（union_make 字节级一致） |
| content_list（v1/v2） | ✅ | ✅（renderer 一致；pipeline 图片抽取未接） |
| 图片抽取落盘（cut_image_and_table） | ✅ | 🟡 VLM 路径有；pipeline 未接 |
| OCR 多语言（~14 语，模型/字典各异） | ✅ | ❌ 仅 `ch`（ppocrv6） |
| CLI `--backend pipeline` | ✅ | ✅ `mlx-mineru --backend pipeline -p a.pdf` 出 md/content_list/middle |
| 验证语料广度 | 多类文档 | 🟡 仅 a.pdf（单栏数字 PDF，无表/公式/图/多栏） |

---

## 6. 建议（按投入产出排序）

1. ~~**接 CLI（`--backend pipeline`）**~~ ✅ 已完成：`mlx-mineru --backend pipeline -p a.pdf`
   直接出 Markdown/content_list/middle（整篇 a.pdf ~7s，无需 VLM 模型）。
2. **可视块装配（image/table/formula 入 middle_json）** — 🟡 公式+表格已接通：
   - 公式：装配产出 `interline_equation`（latex span），`build_page_model` 切图调 MFR，
     CLI 接 FormulaRecognizer，`$$...$$` 已入 Markdown。
   - 表格：装配产出两层 `{type:table, blocks:[table_body(html)]}`（demo1 p5 html 与
     MinerU 逐字一致 2/2）；`build_page_model` 切图→逐块 OCR→SLANet+ 写回 html（去
     `<html><body>` 壳），CLI 接 OcrPipeline+TableRecognizer，`<table>` 已入 Markdown
     （从零 1/2 与 MinerU 完全一致，另一张结构因 crop/OCR 方差略异）。
   - formula_number `\tag{N}`：✅ `optimize_formula_numbers` 把编号块与相邻行间公式合并
     （编号经 OCR → 去括号 → `\tag{N}`，未匹配降级为 text），driver 在 text-fill 后调用，
     demo1 p2 输出 `\tag{1}`/`\tag{2}`。
   - table caption/footnote 嵌套：✅ `classify_visual_blocks`（`find_best_visual_parent`：
     visual-neighbor + 垂直间隔/重叠判定 + 有效阅读序差 + 边/中心距离）把 figure_title→
     table_caption、vision_footnote→table_footnote 关联到表格，产出两层
     `[table_caption, table_body, table_footnote]`（demo1 p5 嵌套与 MinerU 完全一致）；
     post-OCR/数字取字递归填充嵌套块内文本。
   **剩余**：有线表 UNet、image 块、inline_formula 内联 span、cut_image 落盘。
3. **para_split 完整逻辑** — list/index 检测与跨块合并，显著提升列表/多行段落的 Markdown 质量。
4. **数字 PDF 取字路径（pdftext）** — 对"带文本层"的常见 PDF 做到字节级一致（含 CJK 码位）；
   当前 OCR 路径语义正确但码位为 CJK 统一汉字而非源 PDF 的兼容码位。工作量较大但价值高。
5. **有线表格 UNet + 表格方向分类** — 补齐表格另一半（有线表）。
6. **版面启发式过滤层** — 复杂/重叠版面的鲁棒性（框抑制、公式去重、重排）。
7. **广度类（次优先）** — 多语言 OCR、图片输入、Office/hybrid 后端。

**验证建议**：尽快引入含表格/公式/图片/多栏的测试 PDF，建立对应 golden（沿用本会话
"MinerU 真值 → C++ 逐级比对"的范式），把上面 2–6 的对齐做成可回归的 ctest。
