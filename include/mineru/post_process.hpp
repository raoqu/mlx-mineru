// Copyright (c) mlx-mineru.
// Text/equation post-processing — faithful ports of mineru-vl-utils
// post_process/* transforms applied to VLM block content.
#pragma once

#include <string>

namespace mineru::pp {

// text chain (applied to text blocks).
std::string try_convert_display_to_inline(const std::string& text);
std::string try_fix_macro_spacing_in_markdown(const std::string& text);
std::string try_move_underscores_outside(const std::string& text);
std::string process_text(const std::string& text);  // all three, in order

// equation chain (subset; applied to equation blocks).
std::string try_fix_equation_delimiters(const std::string& latex);
std::string try_fix_unbalanced_braces(const std::string& latex);
std::string process_equation(const std::string& latex);  // delimiters + braces

}  // namespace mineru::pp
