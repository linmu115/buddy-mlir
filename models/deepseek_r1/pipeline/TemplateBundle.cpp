#include "TemplateBundle.h"
#include <climits>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <openssl/sha.h>
#include <set>

namespace buddy::runtime {
using Json = nlohmann::json;
namespace {
void require(bool ok, const std::string &field) {
  if (!ok)
    throw std::runtime_error("invalid Route B " + field);
}
int number(const Json &j, const char *key, int min = 1) {
  const auto &v = j.at(key);
  require(v.is_number_integer() && v >= min && v <= INT_MAX, key);
  return v.get<int>();
}
std::string digestField(const Json &j, const std::string &key) {
  std::string v = j.at(key);
  require(v.size() == 64 && v.find_first_not_of("0123456789abcdef") == v.npos,
          key);
  return v;
}
std::filesystem::path file(const std::filesystem::path &root,
                           const Json &entry) {
  std::filesystem::path name = entry.get<std::string>();
  require(!name.empty() && name == name.filename() && name != "." &&
              name != "..",
          "filename");
  return root / name;
}
int boundary(const std::string &text) {
  require(!text.empty() && text.front() >= '0' && text.front() <= '9',
          "unsigned range boundary '" + text + "'");
  int n;
  auto end = text.data() + text.size();
  auto [ptr, ec] = std::from_chars(text.data(), end, n);
  require(ec == std::errc{} && ptr == end && n >= 0,
          "range boundary '" + text + "'");
  return n;
}
} // namespace
std::string hexDigest(const uint8_t *bytes) {
  const char digits[] = "0123456789abcdef";
  std::string s;
  for (int i = 0; i < 32; ++i) {
    s += digits[bytes[i] >> 4];
    s += digits[bytes[i] & 15];
  }
  return s;
}
std::string contentDigest(const void *data, size_t bytes) {
  uint8_t digest[32];
  if (!SHA256(static_cast<const uint8_t *>(data), bytes, digest))
    throw std::runtime_error("SHA256 failed");
  return hexDigest(digest);
}
TemplateBundle TemplateBundle::load(const std::filesystem::path &index) {
  std::ifstream stream(index);
  if (!stream)
    throw std::runtime_error("cannot open " + index.string());
  auto j = Json::parse(stream);
  TemplateBundle b;
  b.id = digestField(j, "bundle_id");
  j.erase("bundle_id");
  auto canonical = j.dump();
  require(contentDigest(canonical.data(), canonical.size()) == b.id,
          "bundle_id digest");
  require(
      j.at("format") == "bj47.route-b/1" && j.at("activation_type") == "f32" &&
          (j.at("architecture") == "Qwen2ForCausalLM" ||
           j.at("architecture") == "Qwen3MoeForCausalLM") &&
          j.at("variant") == "w4g32" && j.at("kernels") == "k3" &&
          j.at("prefill_ime") == true && j.at("pack_version") == "k3-w4g32/1",
      "format or model type");
  b.shape = digestField(j, "shape_id");
  b.model = j.at("model_id");
  auto &c = b.config;
  c = {0,
       number(j, "total_layers"),
       number(j, "total_layers"),
       number(j, "context"),
       number(j, "activation_width"),
       number(j, "vocab_size"),
       number(j, "eos_id", 0),
       ActivationType::F32,
       number(j, "prefill_chunk")};
  b.kvHeads = number(j, "kv_heads");
  b.headDim = number(j, "head_dim");
  require(c.prefillChunk == 64 && c.context >= c.prefillChunk &&
              c.eosId < c.vocabSize && b.headDim % 2 == 0,
          "shape");
  b.spec = {0,     c.endLayer,        c.totalLayers,
            "f32", c.activationWidth, c.prefillChunk};
  b.spec.tokenizer = j.at("tokenizer");
  require(b.spec.tokenizer == "deepseek_r1" || b.spec.tokenizer == "qwen3",
          "tokenizer");
  require(j.at("stop_token_ids").is_array(), "stop_token_ids");
  for (const auto &v : j.at("stop_token_ids")) {
    require(v.is_number_integer() && v >= 0 && v < c.vocabSize,
            "stop_token_ids");
    b.spec.stopTokenIds.push_back(v.get<int>());
  }
  const auto root = std::filesystem::absolute(index).parent_path();
  b.vocab = file(root, j.at("vocab_file"));
  const auto &layouts = j.at("layouts");
  const std::map<std::string, std::string> owners{
      {"embed", "embed"}, {"decoder_layer", "decoder"}, {"lm_head", "head"}};
  std::set<std::filesystem::path> paths;
  for (const auto &k : j.at("templates")) {
    std::string role = k.at("role");
    TemplateEntry e{file(root, k.at("path")), digestField(k, "layout_id"),
                    number(k, "abi_version"), size_t(number(k, "nbytes")),
                    digestField(k, "sha256")};
    require(owners.count(role) &&
                e.layout == digestField(layouts, owners.at(role)) &&
                digestField(k, "shape_id") == b.shape &&
                e.abi == (role == "lm_head" ? 2 : 1) &&
                paths.insert(e.path).second &&
                b.kernels.emplace(role, e).second,
            role + " template");
  }
  require(b.kernels.size() == 3, "template count");
  for (const auto &w : j.at("weights")) {
    std::string owner = w.at("owner"), bucket = w.at("bucket");
    bool layer = owner == "decoder", f32 = bucket == "f32";
    require(layer || owner == "head" || owner == "rope" || owner == "embed",
            "owner");
    require((f32 || ((layer || owner == "head") && bucket == "q4")) &&
                w.at("dtype") == (f32 ? "f32" : "i8") &&
                layer == w.contains("global_layer_id") &&
                digestField(w, "layout_id") == digestField(layouts, owner),
            "weight bucket/layout");
    int id = layer ? number(w, "global_layer_id", 0) : -1;
    require(id < c.totalLayers, "global_layer_id");
    const auto &el = w.at("elements");
    require(el.is_number_unsigned() && el > 0 && el <= SIZE_MAX / (f32 ? 4 : 1),
            "elements");
    size_t elements = el.get<size_t>(), bytes = elements * (f32 ? 4 : 1);
    require(w.at("nbytes").is_number_unsigned() && w.at("nbytes") == bytes,
            "nbytes");
    WeightEntry e{file(root, w.at("path")), elements, bytes,
                  digestField(w, "sha256")};
    require(paths.insert(e.path).second &&
                b.weights.emplace(std::make_tuple(owner, id, bucket), e).second,
            "duplicate weight");
  }
  require(b.weights.size() == 2ULL * c.totalLayers + 4, "weight count");
  return b;
}
void validateChain(const std::vector<LayerRange> &ranges, int total) {
  int next = 0;
  for (auto [a, b] : ranges) {
    require(a == next && b > a && b <= total,
            "range chain at " + std::to_string(next));
    next = b;
  }
  require(next == total, "range coverage");
}
std::vector<LayerRange> parseRanges(const std::string &text, int total,
                                    bool full) {
  std::vector<LayerRange> ranges;
  size_t i = 0;
  while (true) {
    auto comma = text.find(',', i);
    auto field = text.substr(i, comma - i);
    auto colon = field.find(':');
    require(colon != field.npos, "range " + field);
    int a = boundary(field.substr(0, colon)),
        b = boundary(field.substr(colon + 1));
    require(a < b && b <= total, "range " + field);
    ranges.emplace_back(a, b);
    if (comma == text.npos)
      break;
    i = comma + 1;
  }
  if (full)
    validateChain(ranges, total);
  else
    require(ranges.size() == 1, "--stage-range");
  return ranges;
}
} // namespace buddy::runtime
