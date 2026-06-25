#!/usr/bin/env bash
# Trim two tiny abseil usages out of the pdfium source so the from-source static build
# needs neither abseil (481 files) nor its build config. Both are mechanical, behaviour-
# preserving swaps to std:: equivalents:
#   - absl::InlinedVector<T,16,Alloc>  -> std::vector<T,Alloc>      (cpdf_sampledfunc.cpp)
#   - absl::flat_hash_set<T>           -> std::unordered_set<T>     (cpdf_nametree.cpp; only
#                                          .insert(x).second is used)
# Idempotent: re-running is a no-op once patched.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC="third_party/.pdfium-src"

sf="$SRC/core/fpdfapi/page/cpdf_sampledfunc.cpp"
nt="$SRC/core/fpdfdoc/cpdf_nametree.cpp"

if grep -q "abseil-cpp/absl/container/inlined_vector.h" "$sf"; then
  sed -i '' \
    -e 's#include "third_party/abseil-cpp/absl/container/inlined_vector.h"#include <vector>#' \
    -e 's#absl::InlinedVector<\([^,]*\), 16, \(FxAllocAllocator<[^>]*>\)>#std::vector<\1, \2>#g' \
    "$sf"
  echo "patched $sf"
fi

if grep -q "abseil-cpp/absl/container/flat_hash_set.h" "$nt"; then
  sed -i '' \
    -e 's#include "third_party/abseil-cpp/absl/container/flat_hash_set.h"#include <unordered_set>#' \
    -e 's#absl::flat_hash_set#std::unordered_set#g' \
    "$nt"
  echo "patched $nt"
fi

echo "pdfium source patched (abseil removed)."
