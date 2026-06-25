#include "getmodel.h"

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

void usage() {
    std::cerr << "Usage: getmodel [--target <path>] [--force|--force-update] [--retry <n>] [--delay HH:MM:SS] <huggingface_url|modelscope_url> [huggingface_url|modelscope_url]\n";
}

int parseRetryCount(const std::string& value) {
    if (value.empty()) {
        throw std::runtime_error("--retry requires a non-negative integer");
    }
    for (char ch : value) {
        if (ch < '0' || ch > '9') {
            throw std::runtime_error("--retry requires a non-negative integer");
        }
    }
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error("--retry value is too large");
    }
}

long long parseDelaySeconds(const std::string& value) {
    if (value.size() != 8 || value[2] != ':' || value[5] != ':') {
        throw std::runtime_error("--delay requires HH:MM:SS");
    }
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i == 2 || i == 5) {
            continue;
        }
        if (value[i] < '0' || value[i] > '9') {
            throw std::runtime_error("--delay requires HH:MM:SS");
        }
    }

    const int hours = std::stoi(value.substr(0, 2));
    const int minutes = std::stoi(value.substr(3, 2));
    const int seconds = std::stoi(value.substr(6, 2));
    if (minutes >= 60 || seconds >= 60) {
        throw std::runtime_error("--delay minutes and seconds must be less than 60");
    }
    return static_cast<long long>(hours) * 3600 + minutes * 60 + seconds;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            usage();
            return 2;
        }

        std::vector<std::string> urls;
        fs::path target_path;
        bool force_update = false;
        int retry_count = 5;
        long long delay_seconds = 0;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                usage();
                return 0;
            } else if (arg == "--target" || arg == "-o") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(arg + " requires a path");
                }
                target_path = argv[++i];
            } else if (arg.rfind("--target=", 0) == 0) {
                target_path = arg.substr(std::string("--target=").size());
            } else if (arg == "--retry") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(arg + " requires a count");
                }
                retry_count = parseRetryCount(argv[++i]);
            } else if (arg.rfind("--retry=", 0) == 0) {
                retry_count = parseRetryCount(arg.substr(std::string("--retry=").size()));
            } else if (arg == "--delay") {
                if (i + 1 >= argc) {
                    throw std::runtime_error(arg + " requires HH:MM:SS");
                }
                delay_seconds = parseDelaySeconds(argv[++i]);
            } else if (arg.rfind("--delay=", 0) == 0) {
                delay_seconds = parseDelaySeconds(arg.substr(std::string("--delay=").size()));
            } else if (arg == "--force" || arg == "--force-update" || arg == "--force_update" || arg == "force" || arg == "true") {
                force_update = true;
            } else {
                urls.push_back(arg);
            }
        }

        if (urls.empty() || urls.size() > 2) {
            usage();
            return 2;
        }

        DownloadModel(urls, target_path, force_update, retry_count, delay_seconds);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
}
