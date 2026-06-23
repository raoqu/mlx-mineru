// Copyright (c) mlx-mineru.
// OTSL -> HTML table conversion — faithful port of mineru-vl-utils
// post_process/otsl2html.py. The Qwen2-VL "Table Recognition" output is OTSL
// (<fcel>/<ecel>/<lcel>/<ucel>/<xcel>/<nl>); this turns it into an HTML <table>.
#pragma once

#include <string>

namespace mineru {

// Convert OTSL content to an HTML table. If already <table>...</table>, returns
// it unchanged. Returns "" on empty input.
std::string convert_otsl_to_html(const std::string& otsl_content);

}  // namespace mineru
