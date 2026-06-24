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
| **数字 PDF 取字（pdftext / fill_char_in_spans）** | ✅ | ❌ | a.pdf 是数字 PDF，MinerU 走嵌入文本层（康熙部首码位）；我们走 OCR（NFKC 等价但码位不同） |
| **可视块装配**（image/table/formula → span/块） | ✅ | ❌ | 识别器已就绪，但未接入 `__build_page_blocks`/`cut_image_and_table` |
| **有线表格结构（UNet）** | ✅ | ❌ | 仅无线 SLANet+ |
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
2. **可视块装配（image/table/formula 入 middle_json）** — 🟡 公式已接通：装配产出
   `interline_equation` 块（latex span），`build_page_model` 切图调 MFR 写回 latex，CLI
   接 FormulaRecognizer，`$$...$$` 已出现在 Markdown（demo1 验证）。**剩余**：table（嵌套
   caption/body/footnote 的 `__classify_visual_blocks` + html 写回）、image、inline_formula
   内联 span、formula_number 的 `\tag{N}` 关联、cut_image 落盘。
3. **para_split 完整逻辑** — list/index 检测与跨块合并，显著提升列表/多行段落的 Markdown 质量。
4. **数字 PDF 取字路径（pdftext）** — 对"带文本层"的常见 PDF 做到字节级一致（含 CJK 码位）；
   当前 OCR 路径语义正确但码位为 CJK 统一汉字而非源 PDF 的兼容码位。工作量较大但价值高。
5. **有线表格 UNet + 表格方向分类** — 补齐表格另一半（有线表）。
6. **版面启发式过滤层** — 复杂/重叠版面的鲁棒性（框抑制、公式去重、重排）。
7. **广度类（次优先）** — 多语言 OCR、图片输入、Office/hybrid 后端。

**验证建议**：尽快引入含表格/公式/图片/多栏的测试 PDF，建立对应 golden（沿用本会话
"MinerU 真值 → C++ 逐级比对"的范式），把上面 2–6 的对齐做成可回归的 ctest。
