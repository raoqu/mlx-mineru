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
| 表格 cls→rec | 方向/有无线分类 → SLANet+ 或 UNet（跑两者择优） | 🔧 SLANet+ 全量 + TableClassifier 路由 + UNet（有线/低置信无线）+ 择优启发式 | ✅ 路由对齐 batch_analyze |
| 数字取字 / OCR 选择 | `_get_ocr_enable`（auto/ocr） | `is_ocr` 选项 🔧 + 数字层 `fill_chars_in_page` / 回落 OCR | ✅ |
| middle_json（magic_model） | result_to_middle_json（span 预处理、阅读序、para_split、视觉块嵌套、丢弃块） | `pipeline_assemble.cpp`（贪心 span 匹配、classify_visual_blocks、para_split、formula_number→\tag） | 🟡 见下「未实现」 |
| union_make | pipeline_middle_json_mkcontent | `mkcontent.cpp`（VLM 版 union_make，签名带 formula/table_enable） | 🟡 见 §4 |
| 选项 formula/table_enable, lang, is_ocr | batch 内 gating / env | 🔧 `ConvertOpts` 全部接通（withhold 识别器 / union_make flag / 强制 OCR） | ✅ |

**finalize_middle_json_from_preproc（文档级后处理）**：
- ✅ 🔧 `cross_page_table_merge`（跨页表格合并）— 本轮移植，见下。
- ✅ `apply_title_leveling`（标题分级）— 已对齐：确定性路径就是 `_post_block_process` 的
  doc_title→level 1 / paragraph_title→level 2（本项目 `pipeline_assemble.cpp` 一致）；MinerU 的
  LLM-aided 重新分级是可选外部能力（默认关闭），不属于确定性管线，故不移植。
- 🟡 magic_model 的复杂 span 关联用「贪心重叠匹配」近似（a.pdf/demo 验证下与 MinerU 一致）。

**🔧 本轮：跨页表格合并（对齐 `mineru/utils/table_merge.py`）**
- 反向遍历页：若第 N 页首块与第 N-1 页末块均为 table，构建轻量 HTML 表模型（tr/td/th +
  colspan/rowspan，逐字节序列化），按占用矩阵算有效列/总列，结构+视觉双重表头检测，
  caption/footnote/宽度/列匹配门控，跳过重复表头后逐行拼接，footnote 带 `cross_page`+重排
  index 搬到上页，被并走的表 `lines=[]`+`lines_deleted=true`。
- 验证：`tests/golden/table_merge_golden.json`（由 MinerU 真 `merge_table` 生成，10 例覆盖
  表头重复/无表头/续表 caption/footnote/列宽不匹配/rowspan 表头/三页链式）→ C++ 端 `table_merge`
  ctest **逐字节深度相等 10/10**。生成器 `scripts/gen_table_merge_golden.py`。
- 未移植（已记录，pipeline 不触发）：`_apply_cell_merge`（VLM 专有 `cell_merge` 字段，pipeline
  从不产生）；`_clip_overlapped_blank_rowspan_cells`/`_carry_rowspan_structure_to_next_row`
  （仅当 rowspan 跨页延续时触发的空白占位裁剪）。

**🔧 本轮：表格有无线分类路由（对齐 `batch_analyze` + `UnetTableModel.predict`）**
- 所有表格先跑 SLANet+（无线）→ `wireless_html`。
- 分类器判定 `wired_table`、或 `wireless_table` 且 `score<0.9` → 再跑 UNet（有线）→ `wired_html`。
- 择优启发式逐行移植：物理单元格数（含 `<thead>` 计数同款）、非空单元格规模（`round(√n)` 半值
  取偶）、OCR 文字覆盖数、空格率 → 决定回落无线还是采用有线。
- 验证：demo1 五个学术无线表，分类 wireless(<0.9)→触发 UNet 复核→启发式正确保留无线
  （有线 UNet 仅得 1/11/9 格 vs 无线 22/122/63）；`wired_table` 单测仍逐字一致（真有线表走 UNet）。

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
（版面/OCR/公式/表格）+ VLM 对 image/chart 裁剪做理解**（`image_analysis` 控制）。

🔧 **effort 门控（对齐 `_resolve_effective_image_analysis`）**：源项目 hybrid 的 `medium`（默认）
**强制关闭 image/chart 理解**走快速路径，仅 `high` 才按 `image_analysis` 跑 VLM 理解。此前本项目
hybrid 恒跑 VLM 理解（与源项目 medium 不一致：多出每张图/图表一次 VLM 推理，且输出多余的
`<details>chart content</details>`）。现 `understand = hybrid && image_analysis && effort!="medium"`：
- medium（默认）：不跑理解 → 图表渲染为纯 `![]()`，无 "chart content"，~2s（验证）。
- high：跑理解 → 含 "chart content"（验证 count=1）。
- pipeline：从不理解图表 → 纯图像（与源项目 pipeline 一致，验证 count=0）。

🟡 仍近似：源项目 hybrid 在 medium 下用 VLM 做整窗文本抽取 + pipeline 结构融合的细节更复杂；
本项目 medium 仅跳过视觉理解，文本仍走 pipeline OCR。

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
| `{name}_layout.pdf` | ✅（reportlab+pypdf 矢量叠加） | 🔧 ✅（pdfium 矢量叠加：分类色块+阅读序编号，原页文本仍可选） | ✅ |
| `{name}_span.pdf` | ✅ | 🔧 ✅（pdfium 矢量 span 描边框） | ✅ |
| `images/` | ✅ | ✅（VLM 落盘 / web 内联 base64 data URI，仿 `_encode_table_inline_image`） | ✅ |
| 目录结构 | `<out>/<name>/<parse_method>/` | `<out>/<stem>/{pipeline,vlm}/` | ✅ 等价 |

---

## 待办（按优先级）

1. ✅ ~~表格分类路由（SLANet+ vs UNet 由 TableClassifier 选）~~ —— 已完成（见 §1）。
2. ✅ ~~`cross_page_table_merge` + `apply_title_leveling`~~ —— 本轮完成（合并已移植并 golden 验证；
   标题分级确定性路径本已对齐，LLM 路径不适用）。
3. ✅ ~~`_layout.pdf` / `_span.pdf`（移植 draw_bbox.py）~~ —— 本轮完成。**矢量叠加**（pdfium 编辑 API
   画 rect/text 页对象 → GenerateContent → SaveAsCopy），等价于 MinerU 的 reportlab+pypdf：原页内容
   保持矢量、文本可选（demo1 layout.pdf p0 仍可抽取 3586 字符）；分类色块/颜色/0.3 alpha/1pt 描边/
   阅读序编号（Helvetica 10，右上）忠实于 draw_layout_bbox/draw_span_bbox。web 预览改为高亮 layout
   PDF，对齐 gradio（gradio 预览即 _layout.pdf）。
4. 🟡 pin MLX 版本到 `<=0.31.1`。
5. 🟡 逐行核对 pipeline 版 union_make 与本项目 mkcontent 的差异。
6. 🟡 pdfium 光栅化 flag 对齐（关闭最后一个 VLM 歧义字形残差）。
7. 🟡 跨页 rowspan 空白裁剪 / VLM cell_merge（当前 pipeline 不触发，见 §1）。
