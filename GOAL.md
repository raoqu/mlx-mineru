# GOAL — mlx-mineru

用 C++ 原生复刻 [MinerU](https://github.com/opendatalab/MinerU) `3.4.0` 的文档解析能力，完全脱离 Python 运行时，并在 Apple Silicon 上以 **MLX / Metal** 做推理加速。

参考源项目：`~/research/MinerU`（Python，版本 `3.4.0`）。

---

## 1. 我们要做什么

一个**单一可执行文件**的 C++ 原生应用 `mineru`，对齐 MinerU 的核心功能：

- **输入**：PDF、图片（png/jpg/jpeg/jp2/webp/gif/bmp/tiff）、Office（docx/pptx/xlsx）。
- **输出**：与 MinerU **逐字节尽量对齐**的产物：
  - `<name>.md`（Markdown，含图/表/公式）
  - `<name>_content_list.json` / `<name>_content_list_v2.json`（结构化内容）
  - `<name>_middle.json`（中间表示，核心契约）
  - `<name>_model.json`（模型原始输出）
  - `images/`（裁切出的图片）
  - 可选的 layout / span bbox 可视化 PDF
- **两种运行形态**：
  - CLI：`mineru -p <input> -o <output> -b <backend> ...`，对齐 `mineru/cli/client.py` 的参数。
  - `--server`：内置 HTTP 服务，对齐 `mineru/cli/fast_api.py` 的 REST API（`/file_parse`、`/tasks`、`/health` 等）。

## 2. 后端（与源项目一致的四类）

| 后端 | 说明 | 复刻优先级 |
|------|------|-----------|
| `office` | docx/pptx/xlsx，纯 zip+XML 解析，**无 ML** | **P0**（最易，先打通输出契约） |
| `vlm` | 单一视觉语言模型 `MinerU2.5-Pro-2605-1.2B`（Qwen2-VL 架构）一次性产出版面/OCR/公式/表格 | **P1（旗舰，MLX 加速主战场）** |
| `pipeline` | 经典 CV 模型链（版面检测 + OCR + 公式 + 表格，7 个模型） | **P2（最重，后置）** |
| `hybrid` | 文本层抽取 + VLM 融合 | **P3（依赖 vlm + pipeline）** |

**战略重点**：`vlm` 后端是 Apple Silicon 上的主路径——架构上只有**一个** Qwen2-VL 模型（而非 pipeline 的 7 个），最适合用 MLX C++ 实现并打 Metal 加速。`office` 先行用于锁定输出契约，`pipeline` 后置。

## 3. 成功标准（Definition of Done）

按里程碑递进，每一步都以**与 Python MinerU 输出做 golden 对比**为验收：

1. **契约对齐**：给定相同 `middle.json`，C++ 生成的 `.md` / `content_list*.json` 与 Python `union_make` 输出一致。
2. **Office 端到端**：`demo/office_docs/*` 三个文件的 Markdown/JSON 与 Python 结果对齐。
3. **VLM 端到端**：`demo/pdfs/*.pdf` 用 `vlm` 后端（MLX）产出的 `middle.json` 在结构与文本上与 Python `vlm-engine` 对齐（允许浮点/分词层面的可解释差异）。
4. **性能**：VLM 推理在 Apple Silicon 上达到或优于 Python `mlx` 后端（同模型同量化）。
5. **零 Python**：最终二进制不链接、不调用任何 Python 解释器或 `.py`。

## 4. 非目标（本阶段不做 / 暂缓）

- 不复刻国产 AI 芯片（Ascend/寒武纪等）后端——仅聚焦 Apple Silicon + CPU 回退。
- 不复刻 `vllm` / `lmdeploy` / `gradio` / S3 / 远程 OpenAI-server 等周边形态（保留 `vlm-http-client` 思路但不优先）。
- 不复刻 `llm-aided`（标题润色等需外部 LLM 的可选增强）。
- 训练 / 微调不在范围内，仅做推理。
- `pipeline` 后端列为 P2，MVP 不要求完成。

## 5. 核心原则（详见 AGENT.md）

1. **对齐优先**：严格遵循源项目逻辑；若依赖三方库（mlx-vlm、mineru-vl-utils、pytorchocr/PaddleOCR、pdfium 等），**先取三方库源码对齐其实现方式**，无法对齐再迁移实现。
2. **中间表示即契约**：`middle_json` 是贯穿全流程的核心数据结构，保持字段级兼容。
3. **垂直切片**：尽早打通"输入 → middle_json → 输出"端到端（先 office），再逐个加 ML 后端。
4. **MLX/Metal 加速**：Apple Silicon 上的张量计算走 MLX C++（mlx 本身就是 C++ 库），必要处写 Metal kernel。
