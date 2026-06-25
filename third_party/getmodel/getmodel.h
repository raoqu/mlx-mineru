#pragma once

#include <filesystem>
#include <string>
#include <vector>

void DownloadModel(const std::vector<std::string>& urls,
                   const std::filesystem::path& target_path = {},
                   bool force_update = false,
                   int retry_count = 5,
                   long long delay_seconds = 0);
