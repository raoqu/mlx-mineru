// Validate the MLX mBART decoder against the ONNX reference: decode the same fixed encoder
// (scripts/mfr_decode_ref.py -> /tmp/mfr_enc.f32, /tmp/mfr_ref_ids.txt) for 30 forced steps and
// require byte-identical token ids.
// argv: <safetensors> <config.json> [enc.f32] [ref_ids.txt]
#include <cstdio>
#include <fstream>
#include <sstream>
#include <vector>

#include "mineru/mfr_mlx_decoder.hpp"

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <safetensors> <config.json> [enc.f32] [ref_ids.txt]\n", argv[0]);
    return 2;
  }
  std::string st = argv[1], cfg = argv[2];
  std::string encp = argc > 3 ? argv[3] : "/tmp/mfr_enc.f32";
  std::string refp = argc > 4 ? argv[4] : "/tmp/mfr_ref_ids.txt";
  const int N = 50, D = 768, STEPS = 30;

  std::ifstream ef(encp, std::ios::binary);
  if (!ef) { std::fprintf(stderr, "missing %s (run scripts/mfr_decode_ref.py)\n", encp.c_str()); return 2; }
  std::vector<float> enc((size_t)N * D);
  ef.read(reinterpret_cast<char*>(enc.data()), (std::streamsize)enc.size() * sizeof(float));

  std::vector<int> ref;
  { std::ifstream rf(refp); int v; while (rf >> v) ref.push_back(v); }
  if ((int)ref.size() < STEPS) { std::fprintf(stderr, "ref ids too short\n"); return 2; }

  auto dec = mineru::MfrMlxDecoder::load(st, cfg);
  if (!dec) { std::fprintf(stderr, "MLX decoder load failed\n"); return 1; }

  auto out = dec->decode({enc}, {N}, /*max*/STEPS, /*force_steps*/STEPS);
  const std::vector<int>& ids = out[0];

  int mism = 0;
  for (int i = 0; i < STEPS; ++i) {
    int a = i < (int)ids.size() ? ids[i] : -1;
    if (a != ref[i]) { ++mism; if (mism <= 5) std::fprintf(stderr, "  step %d: mlx=%d ref=%d\n", i, a, ref[i]); }
  }
  std::fprintf(stderr, "[test_mfr_mlx] %d/%d ids match\n", STEPS - mism, STEPS);
  if (mism) { std::fprintf(stderr, "FAIL\n"); return 1; }
  std::fprintf(stderr, "PASS (MLX decoder == ONNX reference)\n");
  return 0;
}
