# mlx-mineru

C++ native reimplementation of [MinerU](https://github.com/opendatalab/MinerU) `3.4.0`
document parsing (PDF / image / Office → Markdown / JSON), **with no Python runtime**,
MLX/Metal-accelerated on Apple Silicon.

See [GOAL.md](GOAL.md), [PLAN.md](PLAN.md), and [AGENT.md](AGENT.md) for scope, plan,
and working principles.

## Build & test

```bash
./scripts/build_and_test.sh
```

Requires CMake ≥ 3.20 and a C++20 compiler (Apple clang). Header-only dependencies
(`nlohmann/json`, `CLI11`) are vendored under `third_party/`.

## Status

Under construction, phase by phase (see PLAN.md). Every phase is built, tested, and
committed before the next begins.

- **Phase 0 — core data model** ✅ `middle_json` typed model + enums + lossless JSON
  round-trip (`ctest`: `middle_json_roundtrip`).
