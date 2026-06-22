// Copyright (c) mlx-mineru.
// Minimal test harness: CHECK macros that count failures, no external framework.
#pragma once

#include <cstdio>
#include <string>

namespace mtest {
inline int g_failures = 0;
inline int g_checks = 0;

inline void report(bool ok, const char* expr, const char* file, int line,
                   const std::string& detail = {}) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::fprintf(stderr, "  FAIL %s:%d  CHECK(%s)%s%s\n", file, line, expr,
                 detail.empty() ? "" : "  -- ", detail.c_str());
  }
}
}  // namespace mtest

#define CHECK(expr) ::mtest::report((expr), #expr, __FILE__, __LINE__)
#define CHECK_MSG(expr, msg) ::mtest::report((expr), #expr, __FILE__, __LINE__, (msg))

// Place at end of main(); returns process exit code and prints a summary.
#define TEST_SUMMARY()                                                       \
  (std::fprintf(stderr, "%s: %d checks, %d failures\n",                      \
                ::mtest::g_failures ? "RESULT FAIL" : "RESULT OK",           \
                ::mtest::g_checks, ::mtest::g_failures),                     \
   ::mtest::g_failures == 0 ? 0 : 1)
