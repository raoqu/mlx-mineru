#include <curl/curl.h>

#include "getmodel.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr long kDefaultTimeoutMs = 30000;
constexpr long kProbeTimeoutMs = 3000;
constexpr curl_off_t kProbeBytes = 1024 * 1024;

enum class ProviderType {
    HuggingFace,
    ModelScope,
};

struct RemoteFile {
    std::string path;
    std::uintmax_t size = 0;
    std::string sha256;
};

struct RepoSpec {
    ProviderType type;
    std::string original_url;
    std::string host;
    std::string repo_id;
    std::string revision;
    std::string repo_path;
};

struct Candidate {
    RepoSpec spec;
    std::vector<RemoteFile> files;
};

struct CandidateLoadResult {
    bool ok = false;
    Candidate candidate;
    std::string error;
};

struct ProbeResult {
    bool ok = false;
    std::size_t bytes = 0;
    double seconds = 0.0;
    std::string error;

    double bytes_per_second() const {
        if (seconds <= 0.0) {
            return 0.0;
        }
        return static_cast<double>(bytes) / seconds;
    }
};

struct UrlParts {
    std::string scheme;
    std::string host;
    std::string path;
};

class NetworkDownloadError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::uintmax_t fileSizeOrZero(const fs::path& path);
fs::path safeRelativePath(const std::string& remote_path);
bool hasPartialFiles(const fs::path& output_dir);

std::string providerName(ProviderType type) {
    return type == ProviderType::HuggingFace ? "huggingface" : "modelscope";
}

void ensureCurlGlobalInit() {
    static const bool initialized = []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
        return true;
    }();
    (void)initialized;
}

std::string envValue(const char* key) {
    const char* value = std::getenv(key);
    return value == nullptr ? std::string{} : std::string(value);
}

std::string authToken(ProviderType type) {
    if (type == ProviderType::HuggingFace) {
        std::string token = envValue("HF_TOKEN");
        if (token.empty()) {
            token = envValue("HUGGINGFACE_TOKEN");
        }
        return token;
    }
    return envValue("MODELSCOPE_SDK_TOKEN");
}

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value;
}

bool isSha256Hex(const std::string& value) {
    if (value.size() != 64) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isxdigit(c) != 0;
    });
}

std::vector<std::string> splitPath(const std::string& path) {
    std::vector<std::string> parts;
    std::string current;
    for (char ch : path) {
        if (ch == '/') {
            if (!current.empty()) {
                parts.push_back(current);
                current.clear();
            }
        } else {
            current.push_back(ch);
        }
    }
    if (!current.empty()) {
        parts.push_back(current);
    }
    return parts;
}

int hexValue(char ch) {
    if (ch >= '0' && ch <= '9') {
        return ch - '0';
    }
    if (ch >= 'a' && ch <= 'f') {
        return ch - 'a' + 10;
    }
    if (ch >= 'A' && ch <= 'F') {
        return ch - 'A' + 10;
    }
    return -1;
}

std::string urlDecode(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] == '%' && i + 2 < value.size()) {
            const int high = hexValue(value[i + 1]);
            const int low = hexValue(value[i + 2]);
            if (high >= 0 && low >= 0) {
                out.push_back(static_cast<char>((high << 4) | low));
                i += 2;
                continue;
            }
        }
        out.push_back(value[i]);
    }
    return out;
}

std::string joinPath(const std::vector<std::string>& parts, std::size_t start) {
    std::ostringstream out;
    bool first = true;
    for (std::size_t i = start; i < parts.size(); ++i) {
        if (parts[i].empty()) {
            continue;
        }
        if (!first) {
            out << '/';
        }
        out << parts[i];
        first = false;
    }
    return out.str();
}

UrlParts parseUrl(const std::string& url) {
    static const std::regex re(R"(^([A-Za-z][A-Za-z0-9+.-]*)://([^/?#]+)([^?#]*)?.*$)");
    std::smatch match;
    if (!std::regex_match(url, match, re)) {
        throw std::runtime_error("invalid URL: " + url);
    }

    UrlParts parts;
    parts.scheme = toLower(match[1].str());
    parts.host = toLower(match[2].str());
    parts.path = match[3].matched ? urlDecode(match[3].str()) : "/";
    if (parts.path.empty()) {
        parts.path = "/";
    }
    return parts;
}

RepoSpec parseRepoUrl(const std::string& url) {
    UrlParts parts = parseUrl(url);
    std::vector<std::string> path = splitPath(parts.path);

    if (parts.host.find("huggingface.co") != std::string::npos) {
        if (path.size() < 2) {
            throw std::runtime_error("HuggingFace URL must contain owner/model: " + url);
        }
        RepoSpec spec{ProviderType::HuggingFace, url, parts.host, path[0] + "/" + path[1], "main", ""};
        for (std::size_t i = 2; i + 1 < path.size(); ++i) {
            if (path[i] == "tree" || path[i] == "blob" || path[i] == "resolve") {
                spec.revision = path[i + 1];
                spec.repo_path = joinPath(path, i + 2);
                break;
            }
        }
        return spec;
    }

    if (parts.host.find("modelscope.cn") != std::string::npos) {
        std::size_t offset = 0;
        if (!path.empty() && path[0] == "models") {
            offset = 1;
        }
        if (path.size() < offset + 2) {
            throw std::runtime_error("ModelScope URL must contain owner/model: " + url);
        }
        RepoSpec spec{ProviderType::ModelScope, url, parts.host, path[offset] + "/" + path[offset + 1], "master", ""};
        for (std::size_t i = offset + 2; i + 1 < path.size(); ++i) {
            if (path[i] == "tree" || path[i] == "blob" || path[i] == "resolve") {
                spec.revision = path[i + 1];
                spec.repo_path = joinPath(path, i + 2);
                break;
            }
            if (path[i] == "file" && i + 2 < path.size() && (path[i + 1] == "view" || path[i + 1] == "download")) {
                spec.revision = path[i + 2];
                spec.repo_path = joinPath(path, i + 3);
                break;
            }
            if (path[i] == "files") {
                spec.repo_path = joinPath(path, i + 1);
                break;
            }
        }
        return spec;
    }

    throw std::runtime_error("unsupported model host: " + parts.host);
}

std::string urlEncodePath(const std::string& path) {
    std::ostringstream out;
    const char* hex = "0123456789ABCDEF";
    for (unsigned char ch : path) {
        const bool unreserved = std::isalnum(ch) || ch == '-' || ch == '_' || ch == '.' || ch == '~' || ch == '/';
        if (unreserved) {
            out << static_cast<char>(ch);
        } else {
            out << '%' << hex[ch >> 4] << hex[ch & 0x0F];
        }
    }
    return out.str();
}

std::string apiUrl(const RepoSpec& spec) {
    const std::string repo = urlEncodePath(spec.repo_id);
    if (spec.type == ProviderType::HuggingFace) {
        return "https://huggingface.co/api/models/" + repo + "?revision=" + urlEncodePath(spec.revision);
    }
    return "https://modelscope.cn/api/v1/models/" + repo + "/repo/files?Revision=" +
           urlEncodePath(spec.revision) + "&Recursive=True";
}

std::string downloadUrl(const RepoSpec& spec, const std::string& file_path) {
    const std::string repo = urlEncodePath(spec.repo_id);
    const std::string revision = urlEncodePath(spec.revision);
    const std::string path = urlEncodePath(file_path);
    if (spec.type == ProviderType::HuggingFace) {
        return "https://huggingface.co/" + repo + "/resolve/" + revision + "/" + path;
    }
    return "https://modelscope.cn/models/" + repo + "/resolve/" + revision + "/" + path;
}

curl_slist* buildHeaders(const RepoSpec& spec) {
    curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Accept: application/json, */*");
    std::string token = authToken(spec.type);
    if (!token.empty()) {
        std::string auth = "Authorization: Bearer " + token;
        headers = curl_slist_append(headers, auth.c_str());
    }
    return headers;
}

size_t writeStringCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* body = static_cast<std::string*>(userdata);
    const std::size_t bytes = size * nmemb;
    body->append(ptr, bytes);
    return bytes;
}

std::string curlError(CURLcode code, char* error_buffer) {
    if (error_buffer != nullptr && error_buffer[0] != '\0') {
        return error_buffer;
    }
    return curl_easy_strerror(code);
}

bool isRetryableHttpStatus(long status) {
    return status == 408 || status == 429 || status >= 500;
}

bool isRetryableCurlFailure(CURLcode code, long status) {
    if (code == CURLE_OK) {
        return false;
    }
    if (code == CURLE_WRITE_ERROR || code == CURLE_ABORTED_BY_CALLBACK) {
        return false;
    }
    if (code == CURLE_HTTP_RETURNED_ERROR && !isRetryableHttpStatus(status)) {
        return false;
    }
    return true;
}

std::string httpGetString(const RepoSpec& spec, const std::string& url, long timeout_ms) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        throw std::runtime_error("curl_easy_init failed");
    }

    std::string body;
    char error_buffer[CURL_ERROR_SIZE] = {0};
    curl_slist* headers = buildHeaders(spec);

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "getmodel/0.1");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, timeout_ms);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeStringCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);

    CURLcode code = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (code != CURLE_OK) {
        throw std::runtime_error("GET " + url + " failed: " + curlError(code, error_buffer));
    }
    if (status >= 400) {
        throw std::runtime_error("GET " + url + " failed with HTTP " + std::to_string(status));
    }
    return body;
}

std::string jsonUnescape(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (value[i] != '\\' || i + 1 >= value.size()) {
            out.push_back(value[i]);
            continue;
        }
        const char esc = value[++i];
        switch (esc) {
            case '"':
            case '\\':
            case '/':
                out.push_back(esc);
                break;
            case 'b':
                out.push_back('\b');
                break;
            case 'f':
                out.push_back('\f');
                break;
            case 'n':
                out.push_back('\n');
                break;
            case 'r':
                out.push_back('\r');
                break;
            case 't':
                out.push_back('\t');
                break;
            default:
                out.push_back(esc);
                break;
        }
    }
    return out;
}

bool getJsonStringField(const std::string& object, const std::string& key, std::string& value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*\"((?:\\\\.|[^\"\\\\])*)\"");
    std::smatch match;
    if (!std::regex_search(object, match, re)) {
        return false;
    }
    value = jsonUnescape(match[1].str());
    return true;
}

bool getJsonNumberField(const std::string& object, const std::string& key, std::uintmax_t& value) {
    const std::regex re("\"" + key + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(object, match, re)) {
        return false;
    }
    try {
        value = static_cast<std::uintmax_t>(std::stoull(match[1].str()));
        return true;
    } catch (...) {
        return false;
    }
}

std::vector<std::string> topLevelObjects(const std::string& json) {
    std::vector<std::string> objects;
    std::vector<std::size_t> starts;
    bool in_string = false;
    bool escape = false;

    for (std::size_t i = 0; i < json.size(); ++i) {
        const char ch = json[i];
        if (in_string) {
            if (escape) {
                escape = false;
            } else if (ch == '\\') {
                escape = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }

        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            starts.push_back(i);
        } else if (ch == '}') {
            if (!starts.empty()) {
                const std::size_t start = starts.back();
                starts.pop_back();
                objects.push_back(json.substr(start, i - start + 1));
            }
        }
    }
    return objects;
}

std::vector<RemoteFile> parseHuggingFaceFiles(const std::string& json) {
    std::vector<RemoteFile> files;
    for (const std::string& object : topLevelObjects(json)) {
        std::string path;
        if (!getJsonStringField(object, "rfilename", path) || path.empty() || path.back() == '/') {
            continue;
        }
        RemoteFile file;
        file.path = path;
        getJsonNumberField(object, "size", file.size);
        std::string sha256;
        if (getJsonStringField(object, "oid", sha256) && isSha256Hex(sha256)) {
            file.sha256 = toLower(sha256);
        }
        files.push_back(file);
    }
    return files;
}

std::vector<RemoteFile> parseModelScopeFiles(const std::string& json) {
    std::vector<RemoteFile> files;
    for (const std::string& object : topLevelObjects(json)) {
        std::string path;
        if (!getJsonStringField(object, "Path", path)) {
            getJsonStringField(object, "path", path);
        }
        if (path.empty()) {
            getJsonStringField(object, "Name", path);
        }
        if (path.empty() || path.back() == '/') {
            continue;
        }

        std::string type;
        getJsonStringField(object, "Type", type);
        if (type.empty()) {
            getJsonStringField(object, "type", type);
        }
        type = toLower(type);
        if (type == "tree" || type == "dir" || type == "directory") {
            continue;
        }

        RemoteFile file;
        file.path = path;
        if (!getJsonNumberField(object, "Size", file.size)) {
            getJsonNumberField(object, "size", file.size);
        }
        std::string sha256;
        if (!getJsonStringField(object, "Sha256", sha256)) {
            getJsonStringField(object, "sha256", sha256);
        }
        if (isSha256Hex(sha256)) {
            file.sha256 = toLower(sha256);
        }
        files.push_back(file);
    }
    return files;
}

bool pathMatchesSelection(const std::string& file_path, const std::string& selected_path) {
    if (selected_path.empty()) {
        return true;
    }
    return file_path == selected_path ||
           (file_path.size() > selected_path.size() &&
            file_path.compare(0, selected_path.size(), selected_path) == 0 &&
            file_path[selected_path.size()] == '/');
}

std::vector<RemoteFile> filterSelectedFiles(const RepoSpec& spec, const std::vector<RemoteFile>& files) {
    if (spec.repo_path.empty()) {
        return files;
    }

    std::vector<RemoteFile> selected;
    for (const RemoteFile& file : files) {
        if (pathMatchesSelection(file.path, spec.repo_path)) {
            selected.push_back(file);
        }
    }
    return selected;
}

std::string specLabel(const RepoSpec& spec) {
    std::string label = providerName(spec.type) + ":" + spec.repo_id;
    if (!spec.repo_path.empty()) {
        label += "/" + spec.repo_path;
    }
    return label;
}

std::vector<RemoteFile> fetchFileList(const RepoSpec& spec, long timeout_ms) {
    const std::string body = httpGetString(spec, apiUrl(spec), timeout_ms);
    std::vector<RemoteFile> files = spec.type == ProviderType::HuggingFace
                                        ? parseHuggingFaceFiles(body)
                                        : parseModelScopeFiles(body);
    files = filterSelectedFiles(spec, files);
    std::sort(files.begin(), files.end(), [](const RemoteFile& lhs, const RemoteFile& rhs) {
        return lhs.path < rhs.path;
    });
    files.erase(std::unique(files.begin(), files.end(), [](const RemoteFile& lhs, const RemoteFile& rhs) {
                    return lhs.path == rhs.path;
                }),
                files.end());
    if (files.empty()) {
        throw std::runtime_error("no downloadable files found for " + specLabel(spec));
    }
    return files;
}

CandidateLoadResult loadCandidate(const RepoSpec& spec, long timeout_ms) {
    CandidateLoadResult result;
    result.candidate.spec = spec;
    try {
        result.candidate.files = fetchFileList(spec, timeout_ms);
        result.ok = true;
    } catch (const std::exception& ex) {
        result.error = ex.what();
    }
    return result;
}

std::vector<Candidate> loadCandidates(const std::vector<RepoSpec>& specs) {
    const long timeout_ms = specs.size() > 1 ? kProbeTimeoutMs : kDefaultTimeoutMs;
    std::vector<CandidateLoadResult> results(specs.size());

    if (specs.size() == 1) {
        std::cerr << "Listing " << specLabel(specs.front()) << "...\n";
        results.front() = loadCandidate(specs.front(), timeout_ms);
    } else {
        std::cerr << "Listing sources, each with a 3 second timeout...\n";
        std::vector<std::thread> threads;
        for (std::size_t i = 0; i < specs.size(); ++i) {
            threads.emplace_back([&results, &specs, timeout_ms, i]() {
                results[i] = loadCandidate(specs[i], timeout_ms);
            });
        }
        for (auto& thread : threads) {
            thread.join();
        }
    }

    std::vector<Candidate> candidates;
    std::vector<std::string> errors;
    for (const CandidateLoadResult& result : results) {
        const RepoSpec& spec = result.candidate.spec;
        std::cerr << "  " << specLabel(spec) << ": ";
        if (result.ok) {
            std::cerr << "found " << result.candidate.files.size() << " files\n";
            candidates.push_back(result.candidate);
        } else {
            std::cerr << "unavailable (" << result.error << ")\n";
            errors.push_back(specLabel(spec) + ": " + result.error);
        }
    }

    if (candidates.empty()) {
        std::ostringstream message;
        message << "no source is available";
        for (const std::string& error : errors) {
            message << "\n  " << error;
        }
        throw std::runtime_error(message.str());
    }
    return candidates;
}

RemoteFile chooseProbeFile(const Candidate& candidate) {
    return *std::max_element(candidate.files.begin(), candidate.files.end(), [](const RemoteFile& lhs, const RemoteFile& rhs) {
        return lhs.size < rhs.size;
    });
}

size_t probeWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* bytes = static_cast<std::atomic_size_t*>(userdata);
    const std::size_t written = size * nmemb;
    bytes->fetch_add(written, std::memory_order_relaxed);
    (void)ptr;
    return written;
}

ProbeResult probeCandidate(const Candidate& candidate) {
    ProbeResult result;
    const RemoteFile file = chooseProbeFile(candidate);
    const std::string url = downloadUrl(candidate.spec, file.path);

    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        result.error = "curl_easy_init failed";
        return result;
    }

    std::atomic_size_t bytes{0};
    char error_buffer[CURL_ERROR_SIZE] = {0};
    curl_slist* headers = buildHeaders(candidate.spec);
    const std::string range = "0-" + std::to_string(kProbeBytes - 1);
    const auto started = std::chrono::steady_clock::now();

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "getmodel/0.1");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, kProbeTimeoutMs);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, kProbeTimeoutMs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, probeWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &bytes);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);

    CURLcode code = curl_easy_perform(curl);
    const auto ended = std::chrono::steady_clock::now();
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    result.bytes = bytes.load(std::memory_order_relaxed);
    result.seconds = std::chrono::duration<double>(ended - started).count();
    result.ok = (code == CURLE_OK || result.bytes > 0) && status < 500;
    if (!result.ok) {
        result.error = curlError(code, error_buffer);
    }
    return result;
}

fs::path defaultOutputDir(const std::vector<RepoSpec>& specs) {
    const std::string repo = specs.front().repo_id;
    std::string sanitized = repo;
    std::replace(sanitized.begin(), sanitized.end(), '/', '-');
    return fs::current_path() / "models" / sanitized;
}

fs::path statePath(const fs::path& output_dir) {
    return output_dir / ".getmodel" / "state.txt";
}

fs::path manifestPath(const fs::path& output_dir) {
    return output_dir / ".getmodel" / "manifest.txt";
}

std::map<std::string, std::string> readState(const fs::path& output_dir) {
    std::map<std::string, std::string> state;
    std::ifstream in(statePath(output_dir));
    std::string line;
    while (std::getline(in, line)) {
        const std::size_t pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        state[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return state;
}

void writeState(const fs::path& output_dir, const RepoSpec& spec) {
    fs::create_directories(statePath(output_dir).parent_path());
    std::ofstream out(statePath(output_dir), std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write state file: " + statePath(output_dir).string());
    }
    out << "source=" << providerName(spec.type) << "\n";
    out << "repo_id=" << spec.repo_id << "\n";
    out << "revision=" << spec.revision << "\n";
    out << "path=" << spec.repo_path << "\n";
    out << "url=" << spec.original_url << "\n";
}

bool stateMatchesSpecs(const std::map<std::string, std::string>& state, const std::vector<RepoSpec>& specs) {
    const auto source_it = state.find("source");
    const auto repo_it = state.find("repo_id");
    const auto revision_it = state.find("revision");
    const auto path_it = state.find("path");
    if (source_it == state.end() || repo_it == state.end() || revision_it == state.end()) {
        return false;
    }
    const std::string state_path = path_it == state.end() ? "" : path_it->second;

    for (const RepoSpec& spec : specs) {
        if (providerName(spec.type) == source_it->second &&
            spec.repo_id == repo_it->second &&
            spec.revision == revision_it->second &&
            spec.repo_path == state_path) {
            return true;
        }
    }
    return false;
}

void writeManifest(const fs::path& output_dir, const std::vector<RemoteFile>& files) {
    fs::create_directories(manifestPath(output_dir).parent_path());
    std::ofstream out(manifestPath(output_dir), std::ios::trunc);
    if (!out) {
        throw std::runtime_error("cannot write manifest file: " + manifestPath(output_dir).string());
    }

    out << "version=1\n";
    for (const RemoteFile& file : files) {
        out << file.size << "\t" << file.sha256 << "\t" << file.path << "\n";
    }
}

bool manifestComplete(const fs::path& output_dir) {
    std::ifstream in(manifestPath(output_dir));
    if (!in) {
        return false;
    }

    std::string line;
    bool saw_file = false;
    while (std::getline(in, line)) {
        if (line.empty() || line == "version=1") {
            continue;
        }

        const std::size_t first_tab = line.find('\t');
        const std::size_t second_tab = first_tab == std::string::npos ? std::string::npos : line.find('\t', first_tab + 1);
        if (first_tab == std::string::npos || second_tab == std::string::npos) {
            return false;
        }

        std::uintmax_t expected_size = 0;
        try {
            expected_size = static_cast<std::uintmax_t>(std::stoull(line.substr(0, first_tab)));
        } catch (...) {
            return false;
        }

        const std::string remote_path = line.substr(second_tab + 1);
        if (remote_path.empty()) {
            return false;
        }

        const fs::path local_path = output_dir / safeRelativePath(remote_path);
        if (!fs::exists(local_path) || !fs::is_regular_file(local_path)) {
            return false;
        }
        if (expected_size > 0 && fileSizeOrZero(local_path) != expected_size) {
            return false;
        }
        saw_file = true;
    }

    return saw_file;
}

bool localDownloadComplete(const fs::path& output_dir, const std::vector<RepoSpec>& specs) {
    if (!fs::exists(output_dir) || hasPartialFiles(output_dir)) {
        return false;
    }
    const auto state = readState(output_dir);
    return stateMatchesSpecs(state, specs) && manifestComplete(output_dir);
}

bool hasPartialFiles(const fs::path& output_dir) {
    if (!fs::exists(output_dir)) {
        return false;
    }
    for (const auto& entry : fs::recursive_directory_iterator(output_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".part") {
            return true;
        }
    }
    return false;
}

std::string formatBytes(double bytes) {
    const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int unit = 0;
    while (bytes >= 1024.0 && unit < 4) {
        bytes /= 1024.0;
        ++unit;
    }
    std::ostringstream out;
    out << std::fixed << std::setprecision(unit == 0 ? 0 : 1) << bytes << ' ' << units[unit];
    return out.str();
}

std::string formatDuration(double seconds) {
    if (seconds < 0.0 || !std::isfinite(seconds)) {
        return "--:--";
    }
    const auto total = static_cast<long long>(seconds + 0.5);
    const long long hours = total / 3600;
    const long long minutes = (total % 3600) / 60;
    const long long secs = total % 60;
    std::ostringstream out;
    if (hours > 0) {
        out << hours << ":" << std::setw(2) << std::setfill('0') << minutes << ":"
            << std::setw(2) << std::setfill('0') << secs;
    } else {
        out << minutes << ":" << std::setw(2) << std::setfill('0') << secs;
    }
    return out.str();
}

std::string formatClockDuration(long long seconds) {
    if (seconds < 0) {
        seconds = 0;
    }
    const long long hours = seconds / 3600;
    const long long minutes = (seconds % 3600) / 60;
    const long long secs = seconds % 60;
    std::ostringstream out;
    out << std::setw(2) << std::setfill('0') << hours << ":"
        << std::setw(2) << std::setfill('0') << minutes << ":"
        << std::setw(2) << std::setfill('0') << secs;
    return out.str();
}

void waitBeforeDownload(long long delay_seconds) {
    if (delay_seconds <= 0) {
        return;
    }

    for (long long remaining = delay_seconds; remaining > 0; --remaining) {
        std::cerr << "\rStarting download in " << formatClockDuration(remaining) << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cerr << "\rStarting download in " << formatClockDuration(0) << "\n";
}

std::uintmax_t fileSizeOrZero(const fs::path& path) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    return ec ? 0 : size;
}

std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
public:
    Sha256() = default;

    void update(const unsigned char* data, std::size_t len) {
        for (std::size_t i = 0; i < len; ++i) {
            data_[datalen_++] = data[i];
            if (datalen_ == 64) {
                transform();
                bitlen_ += 512;
                datalen_ = 0;
            }
        }
    }

    std::array<unsigned char, 32> final() {
        std::size_t i = datalen_;

        if (datalen_ < 56) {
            data_[i++] = 0x80;
            while (i < 56) {
                data_[i++] = 0x00;
            }
        } else {
            data_[i++] = 0x80;
            while (i < 64) {
                data_[i++] = 0x00;
            }
            transform();
            data_.fill(0);
        }

        bitlen_ += datalen_ * 8;
        data_[63] = static_cast<unsigned char>(bitlen_);
        data_[62] = static_cast<unsigned char>(bitlen_ >> 8);
        data_[61] = static_cast<unsigned char>(bitlen_ >> 16);
        data_[60] = static_cast<unsigned char>(bitlen_ >> 24);
        data_[59] = static_cast<unsigned char>(bitlen_ >> 32);
        data_[58] = static_cast<unsigned char>(bitlen_ >> 40);
        data_[57] = static_cast<unsigned char>(bitlen_ >> 48);
        data_[56] = static_cast<unsigned char>(bitlen_ >> 56);
        transform();

        std::array<unsigned char, 32> hash{};
        for (std::size_t j = 0; j < 4; ++j) {
            for (std::size_t k = 0; k < 8; ++k) {
                hash[j + (k * 4)] = static_cast<unsigned char>((state_[k] >> (24 - j * 8)) & 0x000000ff);
            }
        }
        return hash;
    }

private:
    void transform() {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

        std::uint32_t m[64];
        for (std::size_t i = 0, j = 0; i < 16; ++i, j += 4) {
            m[i] = (static_cast<std::uint32_t>(data_[j]) << 24) |
                   (static_cast<std::uint32_t>(data_[j + 1]) << 16) |
                   (static_cast<std::uint32_t>(data_[j + 2]) << 8) |
                   (static_cast<std::uint32_t>(data_[j + 3]));
        }
        for (std::size_t i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(m[i - 15], 7) ^ rotr(m[i - 15], 18) ^ (m[i - 15] >> 3);
            const std::uint32_t s1 = rotr(m[i - 2], 17) ^ rotr(m[i - 2], 19) ^ (m[i - 2] >> 10);
            m[i] = m[i - 16] + s0 + m[i - 7] + s1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];

        for (std::size_t i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + ch + k[i] + m[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<unsigned char, 64> data_{};
    std::size_t datalen_ = 0;
    std::uint64_t bitlen_ = 0;
    std::array<std::uint32_t, 8> state_{
        0x6a09e667,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19};
};

std::string toHex(const std::array<unsigned char, 32>& bytes) {
    const char* hex = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes) {
        out.push_back(hex[byte >> 4]);
        out.push_back(hex[byte & 0x0F]);
    }
    return out;
}

std::string sha256File(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("cannot open file for hash: " + path.string());
    }

    Sha256 sha;
    std::array<char, 1024 * 1024> buffer{};
    while (in) {
        in.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) {
            sha.update(reinterpret_cast<const unsigned char*>(buffer.data()), static_cast<std::size_t>(got));
        }
    }
    return toHex(sha.final());
}

fs::path safeRelativePath(const std::string& remote_path) {
    if (remote_path.empty() || remote_path.front() == '/' || remote_path.front() == '\\') {
        throw std::runtime_error("unsafe remote path: " + remote_path);
    }

    fs::path safe;
    std::string segment;
    for (char ch : remote_path) {
        if (ch == '/' || ch == '\\') {
            if (segment.empty() || segment == "." || segment == ".." || segment.find(':') != std::string::npos) {
                throw std::runtime_error("unsafe remote path: " + remote_path);
            }
            safe /= segment;
            segment.clear();
        } else {
            segment.push_back(ch);
        }
    }
    if (segment.empty() || segment == "." || segment == ".." || segment.find(':') != std::string::npos) {
        throw std::runtime_error("unsafe remote path: " + remote_path);
    }
    safe /= segment;
    return safe;
}

std::uintmax_t remoteFileSize(const RepoSpec& spec, const std::string& url) {
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        return 0;
    }

    char error_buffer[CURL_ERROR_SIZE] = {0};
    curl_slist* headers = buildHeaders(spec);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "getmodel/0.1");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, kDefaultTimeoutMs);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, kDefaultTimeoutMs);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);

    const CURLcode code = curl_easy_perform(curl);
    curl_off_t length = -1;
    if (code == CURLE_OK) {
        curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return length > 0 ? static_cast<std::uintmax_t>(length) : 0;
}

struct ProgressState {
    std::string label;
    std::uintmax_t initial = 0;
    std::uintmax_t total = 0;
    std::uintmax_t last_bytes = 0;
    std::uintmax_t last_render_bytes = 0;
    double smoothed_speed = 0.0;
    bool rendered_complete = false;
    std::chrono::steady_clock::time_point started_at = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point last_sample_at = started_at;
    std::chrono::steady_clock::time_point last_render_at = started_at;
};

int progressCallback(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t, curl_off_t) {
    auto* state = static_cast<ProgressState*>(clientp);
    const auto now = std::chrono::steady_clock::now();
    const double sample_elapsed = std::chrono::duration<double>(now - state->last_sample_at).count();
    const double render_elapsed = std::chrono::duration<double>(now - state->last_render_at).count();
    std::uintmax_t current = state->initial + static_cast<std::uintmax_t>(std::max<curl_off_t>(0, dlnow));
    const std::uintmax_t total = state->total > 0 ? state->total : state->initial + static_cast<std::uintmax_t>(std::max<curl_off_t>(0, dltotal));
    if (total > 0) {
        current = std::min(current, total);
    }

    const std::uintmax_t delta = current >= state->last_bytes ? current - state->last_bytes : 0;
    if (sample_elapsed >= 0.20) {
        const double instant_speed = static_cast<double>(delta) / sample_elapsed;
        if (instant_speed > 0.0) {
            if (state->smoothed_speed <= 0.0) {
                state->smoothed_speed = instant_speed;
            } else {
                state->smoothed_speed = (state->smoothed_speed * 0.80) + (instant_speed * 0.20);
            }
        }
        state->last_bytes = current;
        state->last_sample_at = now;
    }

    const bool complete = total > 0 && current >= total;
    if (complete && state->rendered_complete) {
        return 0;
    }
    const std::uintmax_t rendered_delta = current >= state->last_render_bytes ? current - state->last_render_bytes : 0;
    const bool meaningful_change = total > 0
                                       ? rendered_delta >= std::max<std::uintmax_t>(1, total / 200)
                                       : current != state->last_render_bytes;
    if (!complete && (render_elapsed < 0.50 || !meaningful_change)) {
        return 0;
    }

    const double total_elapsed = std::chrono::duration<double>(now - state->started_at).count();
    const double average_speed = total_elapsed > 0.0 && current > state->initial
                                     ? static_cast<double>(current - state->initial) / total_elapsed
                                     : 0.0;
    const double display_speed = state->smoothed_speed > 0.0 ? state->smoothed_speed : average_speed;
    const double eta = total > current && display_speed > 0.0
                           ? static_cast<double>(total - current) / display_speed
                           : 0.0;
    state->last_bytes = current;
    state->last_render_bytes = current;
    state->last_render_at = now;
    if (complete) {
        state->rendered_complete = true;
    }

    constexpr int bar_width = 24;
    int filled = 0;
    double percent = 0.0;
    if (total > 0) {
        percent = 100.0 * static_cast<double>(current) / static_cast<double>(total);
        filled = std::min(bar_width, static_cast<int>((percent / 100.0) * bar_width));
    }

    std::ostringstream line;
    line << '\r' << state->label << " [";
    for (int i = 0; i < bar_width; ++i) {
        line << (i < filled ? '#' : '-');
    }
    line << "] ";
    if (total > 0) {
        line << std::setw(6) << std::fixed << std::setprecision(1) << percent << "% ";
        line << formatBytes(static_cast<double>(current)) << "/" << formatBytes(static_cast<double>(total));
    } else {
        line << formatBytes(static_cast<double>(current));
    }
    line << " " << formatBytes(display_speed) << "/s";
    if (total > 0 && !complete) {
        line << " ETA " << formatDuration(eta);
    }
    std::cerr << line.str() << std::flush;
    return 0;
}

size_t fileWriteCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::ofstream*>(userdata);
    const std::size_t bytes = size * nmemb;
    out->write(ptr, static_cast<std::streamsize>(bytes));
    return out->good() ? bytes : 0;
}

void downloadOneFile(const RepoSpec& spec, const RemoteFile& remote_file, const fs::path& output_dir, std::size_t index, std::size_t count, bool force_update) {
    const fs::path final_path = output_dir / safeRelativePath(remote_file.path);
    fs::create_directories(final_path.parent_path());
    fs::path part_path = final_path;
    part_path += ".part";

    std::uintmax_t expected_size = remote_file.size;
    const std::string url = downloadUrl(spec, remote_file.path);
    if (expected_size == 0) {
        expected_size = remoteFileSize(spec, url);
    }

    if (fs::exists(final_path)) {
        const std::uintmax_t local_size = fileSizeOrZero(final_path);
        if (force_update && !remote_file.sha256.empty()) {
            if ((expected_size == 0 || local_size == expected_size) && sha256File(final_path) == remote_file.sha256) {
                std::cerr << "[" << index << "/" << count << "] skip " << remote_file.path << " (hash match)\n";
                return;
            }
            std::cerr << "[" << index << "/" << count << "] update " << remote_file.path << " (hash changed)\n";
            std::error_code ec;
            fs::remove(final_path, ec);
            fs::remove(part_path, ec);
        } else if (expected_size > 0 && local_size == expected_size) {
            std::cerr << "[" << index << "/" << count << "] skip " << remote_file.path << " (complete)\n";
            return;
        } else if (expected_size == 0) {
            std::cerr << "[" << index << "/" << count << "] skip " << remote_file.path << " (exists, no remote hash)\n";
            return;
        } else if (expected_size > 0 && local_size > expected_size) {
            std::error_code ec;
            fs::remove(final_path, ec);
            fs::remove(part_path, ec);
        }
    }

    if (fs::exists(final_path) && expected_size > 0 && fileSizeOrZero(final_path) < expected_size) {
        if (!fs::exists(part_path)) {
            fs::rename(final_path, part_path);
        } else {
            std::error_code ec;
            fs::remove(final_path, ec);
        }
    }

    std::uintmax_t offset = fileSizeOrZero(part_path);
    if (expected_size > 0 && offset > expected_size) {
        std::error_code ec;
        fs::remove(part_path, ec);
        offset = 0;
    }
    if (expected_size > 0 && offset == expected_size) {
        if (!force_update || remote_file.sha256.empty() || sha256File(part_path) == remote_file.sha256) {
            fs::rename(part_path, final_path);
            std::cerr << "[" << index << "/" << count << "] complete " << remote_file.path << "\n";
            return;
        }
        std::error_code ec;
        fs::remove(part_path, ec);
        offset = 0;
    }

    std::ofstream out(part_path, std::ios::binary | std::ios::app);
    if (!out) {
        throw std::runtime_error("cannot open output file: " + part_path.string());
    }

    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        throw std::runtime_error("curl_easy_init failed");
    }

    char error_buffer[CURL_ERROR_SIZE] = {0};
    curl_slist* headers = buildHeaders(spec);
    std::ostringstream label;
    label << "[" << index << "/" << count << "] " << remote_file.path;
    ProgressState progress;
    progress.label = label.str();
    progress.initial = offset;
    progress.total = expected_size;
    progress.last_bytes = offset;
    progress.last_render_bytes = offset;
    progress.started_at = std::chrono::steady_clock::now();
    progress.last_sample_at = progress.started_at;
    progress.last_render_at = progress.started_at;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "getmodel/0.1");
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, kDefaultTimeoutMs);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, fileWriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &out);
    curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progressCallback);
    curl_easy_setopt(curl, CURLOPT_XFERINFODATA, &progress);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, error_buffer);
    if (offset > 0) {
        curl_easy_setopt(curl, CURLOPT_RESUME_FROM_LARGE, static_cast<curl_off_t>(offset));
    }

    const CURLcode code = curl_easy_perform(curl);
    long status = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &status);
    out.close();
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    std::cerr << "\n";

    if (code != CURLE_OK) {
        std::string message = "download failed for " + remote_file.path + ": " + curlError(code, error_buffer);
        if (status > 0) {
            message += " (HTTP " + std::to_string(status) + ")";
        }
        if (isRetryableCurlFailure(code, status)) {
            throw NetworkDownloadError(message);
        }
        throw std::runtime_error(message);
    }

    const std::uintmax_t downloaded = fileSizeOrZero(part_path);
    if (expected_size > 0 && downloaded != expected_size) {
        throw NetworkDownloadError("downloaded size mismatch for " + remote_file.path + ": got " +
                                   std::to_string(downloaded) + ", expected " + std::to_string(expected_size));
    }
    if (!remote_file.sha256.empty()) {
        const std::string actual_hash = sha256File(part_path);
        if (actual_hash != remote_file.sha256) {
            throw NetworkDownloadError("sha256 mismatch for " + remote_file.path + ": got " + actual_hash +
                                       ", expected " + remote_file.sha256);
        }
    }
    fs::rename(part_path, final_path);
}

int retryDelaySeconds(int retry_number) {
    return retry_number > 10 ? 5 : 2;
}

void downloadOneFileWithRetry(const RepoSpec& spec,
                              const RemoteFile& remote_file,
                              const fs::path& output_dir,
                              std::size_t index,
                              std::size_t count,
                              bool force_update,
                              int retry_count) {
    int retries = 0;
    while (true) {
        try {
            downloadOneFile(spec, remote_file, output_dir, index, count, force_update);
            return;
        } catch (const NetworkDownloadError& ex) {
            ++retries;
            if (retry_count > 0 && retries > retry_count) {
                throw;
            }

            const int delay = retryDelaySeconds(retries);
            std::cerr << "[" << index << "/" << count << "] retry " << retries;
            if (retry_count > 0) {
                std::cerr << "/" << retry_count;
            }
            std::cerr << " for " << remote_file.path << " in " << delay << "s: " << ex.what() << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(delay));
        }
    }
}

const Candidate& selectCandidate(const std::vector<Candidate>& candidates, const fs::path& output_dir) {
    const auto state = readState(output_dir);
    const auto source_it = state.find("source");
    if (source_it != state.end() && hasPartialFiles(output_dir)) {
        for (const Candidate& candidate : candidates) {
            if (providerName(candidate.spec.type) == source_it->second) {
                std::cerr << "Resuming previous source: " << source_it->second << "\n";
                return candidate;
            }
        }
    }

    if (candidates.size() == 1) {
        return candidates.front();
    }

    std::cerr << "Probing sources for up to 3 seconds...\n";
    std::vector<ProbeResult> results(candidates.size());
    std::vector<std::thread> threads;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        threads.emplace_back([&results, &candidates, i]() {
            results[i] = probeCandidate(candidates[i]);
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    std::size_t best = 0;
    for (std::size_t i = 0; i < candidates.size(); ++i) {
        std::cerr << "  " << providerName(candidates[i].spec.type) << ": ";
        if (results[i].ok) {
            std::cerr << formatBytes(results[i].bytes_per_second()) << "/s\n";
        } else {
            std::cerr << "failed (" << results[i].error << ")\n";
        }
        if (results[i].bytes_per_second() > results[best].bytes_per_second()) {
            best = i;
        }
    }
    return candidates[best];
}

}  // namespace

void DownloadModel(const std::vector<std::string>& urls, const fs::path& target_path, bool force_update, int retry_count, long long delay_seconds) {
    ensureCurlGlobalInit();

    if (urls.empty() || urls.size() > 2) {
        throw std::runtime_error("DownloadModel requires one or two repository URLs");
    }
    if (retry_count < 0) {
        throw std::runtime_error("retry_count must be non-negative");
    }
    if (delay_seconds < 0) {
        throw std::runtime_error("delay_seconds must be non-negative");
    }

    std::vector<RepoSpec> specs;
    specs.reserve(urls.size());
    for (const std::string& url : urls) {
        specs.push_back(parseRepoUrl(url));
    }

    const fs::path output_dir = target_path.empty() ? defaultOutputDir(specs) : target_path;
    fs::create_directories(output_dir);

    if (!force_update && localDownloadComplete(output_dir, specs)) {
        std::cout << fs::absolute(output_dir).string() << "\n";
        return;
    }

    waitBeforeDownload(delay_seconds);

    std::vector<Candidate> candidates = loadCandidates(specs);

    const Candidate& selected = selectCandidate(candidates, output_dir);
    writeState(output_dir, selected.spec);

    std::cerr << "Selected source: " << providerName(selected.spec.type) << "\n";
    std::cerr << "Output: " << output_dir.string() << "\n";
    if (force_update) {
        std::cerr << "Force update: enabled\n";
    }
    std::cerr << "Retry: " << (retry_count == 0 ? std::string("unlimited") : std::to_string(retry_count)) << "\n";

    for (std::size_t i = 0; i < selected.files.size(); ++i) {
        downloadOneFileWithRetry(selected.spec, selected.files[i], output_dir, i + 1, selected.files.size(), force_update, retry_count);
    }

    writeManifest(output_dir, selected.files);
    std::cerr << "Done.\n";
}
