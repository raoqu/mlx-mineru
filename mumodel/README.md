---
license: agpl-3.0
language:
  - zh
  - en
tags:
  - mineru
  - document-parsing
  - pdf-extraction
  - ocr
  - layout-analysis
  - table-recognition
  - formula-recognition
  - mlx
  - onnx
base_model:
  - opendatalab/MinerU2.5-Pro-2605-1.2B
  - opendatalab/PDF-Extract-Kit-1.0
pipeline_tag: image-to-text
---

# mlx-mineru runtime models (`mumodel`)

**English follows the Chinese section.**

这是 [`mlx-mineru`](https://github.com/) —— 一个 **纯 C++ / [MLX](https://github.com/ml-explore/mlx) 原生实现的 [MinerU](https://github.com/opendatalab/MinerU) 文档解析引擎**（Apple Silicon / Metal 加速，运行时零 Python）—— 在推理时**实际加载**的全部模型文件。
本仓库**只包含运行时需要的文件**：上游原始的 PyTorch/Paddle 权重与 HF 元数据已被转换/裁剪掉，不在此处。

> ⚠️ 这是一个**模型分发包**，不是独立可运行的模型。它需要配合 `mlx-mineru` 可执行文件使用。

---

## 📦 组成 / Components

| 模块 | 文件 | 格式 | 上游来源 | 大小 |
|---|---|---|---|---|
| **VLM**（MinerU2.5，Qwen2-VL 架构，1.2B） | `MinerU2.5-tokenizer/model.safetensors` + `vocab.json` / `tokenizer.json` / `merges.txt` | safetensors（MLX 直接加载） | [`opendatalab/MinerU2.5-Pro-2605-1.2B`](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) | ~2.2 GB |
| **版面检测**（PP-DocLayoutV2，RT-DETR + 阅读序） | `pipeline/Layout/layout.onnx` | ONNX | [`opendatalab/PDF-Extract-Kit-1.0`](https://huggingface.co/opendatalab/PDF-Extract-Kit-1.0) → 导出 | ~205 MB |
| **OCR**（PP-OCRv6：DBNet 检测 + CTC/SVTR 识别） | `pipeline/OCR/ocr_det.onnx` · `ocr_rec.onnx` · `ppocrv6_dict.txt` | ONNX | PaddleOCR PP-OCRv6 → 导出 | ~83 MB |
| **公式识别**（UniMERNet：Swin 编码器 + mBART 解码器） | `pipeline/MFR/mfr_encoder.onnx` · `mfr_decoder.onnx` · `mfr_vocab.txt` | ONNX | UniMERNet → 导出 | ~775 MB |
| **表格分类**（有线/无线，PP-LCNet） | `pipeline/TabCls/PP-LCNet_x1_0_table_cls.onnx` | ONNX | `PDF-Extract-Kit-1.0` | ~6.5 MB |
| **表格结构**（无线 SLANet+ / 有线 UNet） | `pipeline/TabRec/SlanetPlus/slanet-plus.onnx` (+`table_structure_dict.txt`) · `pipeline/TabRec/UnetStructure/unet.onnx` | ONNX | `PDF-Extract-Kit-1.0` | ~15 MB |

合计 ≈ **3.2 GB**。

### 目录结构 / Layout

```
mumodel/
├── MinerU2.5-tokenizer/          # VLM 后端（Qwen2-VL / MinerU2.5）
│   ├── model.safetensors
│   ├── vocab.json
│   ├── tokenizer.json
│   └── merges.txt
└── pipeline/                     # pipeline / hybrid 后端（原生 ONNX）
    ├── Layout/layout.onnx
    ├── OCR/{ocr_det.onnx, ocr_rec.onnx, ppocrv6_dict.txt}
    ├── MFR/{mfr_encoder.onnx, mfr_decoder.onnx, mfr_vocab.txt}
    ├── TabCls/PP-LCNet_x1_0_table_cls.onnx
    └── TabRec/
        ├── SlanetPlus/{slanet-plus.onnx, table_structure_dict.txt}
        └── UnetStructure/unet.onnx
```

三种后端共用本目录：
- **`pipeline`**：纯 ONNX 流水线（版面 / OCR / 公式 / 表格），低资源、无幻觉。
- **`vlm`**：MinerU2.5 视觉大模型整页端到端解析。
- **`hybrid-engine`**：pipeline 结构 + VLM 图像/图表理解（`high` 强度时）。

---

## 🚀 使用 / Usage

将本目录命名为 `mumodel/`，放在 `mlx-mineru` **可执行文件同级目录**或**当前工作目录**下即可——程序会自动发现（先查工作目录，再查可执行文件所在目录及其上级）。

```bash
# 下载本仓库到 ./mumodel
huggingface-cli download <this-repo> --local-dir mumodel
# 或 ModelScope:  modelscope download --model <this-repo> --local_dir mumodel

# 运行（无需 --model / --pipeline-models，自动发现 mumodel/）
mlx-mineru --backend pipeline -p input.pdf -o output
mlx-mineru --web          # 启动本地 Web UI（vlm / pipeline / hybrid-engine 可选）
```

也可用 `--model <dir>/MinerU2.5-tokenizer` 与 `--pipeline-models <dir>/pipeline` 显式指定。

---

## 🔧 来源与转换 / Provenance & conversion

- **VLM** 权重为 `opendatalab/MinerU2.5-Pro-2605-1.2B` 的原始 `safetensors`，由 MLX 直接加载（未二次转换）。
- **layout / OCR / 公式** 的 ONNX 由上游 PyTorch/Paddle 模型**离线导出**（RepVGG 重参数化、贪心解码逐 token 校验等），导出脚本见 `mlx-mineru` 仓库 `scripts/export_*_onnx.py`；运行时只跑 ONNX，无 torch 依赖。
- **表格分类 / 结构（SLANet+ / UNet）** 直接取自 `PDF-Extract-Kit-1.0` 的 ONNX。
- 已剔除的原始文件（PyTorch `model.safetensors`、HF `config/tokenizer` 元数据、重复 ONNX）保存在上游仓库或 `mlx-mineru` 的 `orgmodel/`，**运行时不需要**。

---

## 📄 许可证 / License

本分发包打包了多个上游模型，请遵循各自的许可证：

| 组件 | 上游 | 许可证 |
|---|---|---|
| MinerU2.5 VLM | OpenDataLab MinerU | **AGPL-3.0**（同 MinerU 项目） |
| PP-DocLayoutV2 / PP-LCNet / SLANet+ | PaddlePaddle / PDF-Extract-Kit | Apache-2.0 |
| PP-OCRv6（DBNet + SVTR） | PaddleOCR | Apache-2.0 |
| UniMERNet | UniMERNet | Apache-2.0 |

由于打包了 AGPL-3.0 的 MinerU2.5 权重，**整体分发以 AGPL-3.0 为准**。以商业或闭源方式使用前，请核对 [MinerU](https://github.com/opendatalab/MinerU/blob/master/LICENSE.md) 及各上游的授权条款。本仓库**不**重新授权任何上游权重，仅作格式转换与再分发。

---

## 🙏 致谢 / Acknowledgements

感谢 [OpenDataLab MinerU](https://github.com/opendatalab/MinerU)、[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)、[UniMERNet](https://github.com/opendatalab/UniMERNet) 等上游项目。`mlx-mineru` 是对 MinerU 的忠实 C++/MLX 重实现（除 PyTorch→MLX/ONNX 外，力求逐级对齐）。

## 📚 引用 / Citation

```bibtex
@misc{mineru,
  title  = {MinerU: A One-stop, Open-source, High-quality Data Extraction Tool},
  author = {OpenDataLab},
  year   = {2024},
  url    = {https://github.com/opendatalab/MinerU}
}
```

---

## English summary

This repository bundles **exactly the model files the [`mlx-mineru`](https://github.com/) engine loads at inference time** — a native **C++/MLX reimplementation of [MinerU](https://github.com/opendatalab/MinerU)** for Apple Silicon, with **zero Python at runtime**. It is a *model distribution*, not a standalone runnable model.

**Components:** a 1.2B MinerU2.5 VLM (Qwen2-VL, `safetensors`, loaded directly by MLX) for the `vlm`/`hybrid` backends, plus ONNX models for the `pipeline` backend — PP-DocLayoutV2 (layout), PP-OCRv6 (DBNet detection + SVTR recognition), UniMERNet (formula), PP-LCNet (table classification), and SLANet+/UNet (table structure). The layout/OCR/formula ONNX were exported offline from the upstream PyTorch/Paddle models (`scripts/export_*_onnx.py`); the VLM weights are the original OpenDataLab `safetensors`.

**Usage:** place this folder as `mumodel/` next to the `mlx-mineru` executable or in your working directory — it is auto-discovered (no `--model` flag needed).

**License:** mixed upstream licenses — the MinerU2.5 VLM is **AGPL-3.0**; the PaddleOCR/UniMERNet/PDF-Extract-Kit ONNX are Apache-2.0. Because the bundle includes AGPL-3.0 weights, treat the whole distribution as **AGPL-3.0** and verify each upstream's terms before commercial/closed use. No upstream weights are re-licensed here.
