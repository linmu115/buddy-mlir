#include "TemplateBundle.h"
#include <array>
#include <climits>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <openssl/evp.h>
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
void verifyFileDigest(const std::filesystem::path &path,
                      const std::string &digest, const char *role) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error(std::string("cannot open ") + role + ": " +
                             path.string());
  std::unique_ptr<EVP_MD_CTX, decltype(&EVP_MD_CTX_free)> ctx(EVP_MD_CTX_new(),
                                                              EVP_MD_CTX_free);
  auto check = [](int rc) {
    if (rc != 1)
      throw std::runtime_error("SHA256 failed");
  };
  if (!ctx)
    throw std::bad_alloc();
  check(EVP_DigestInit_ex(ctx.get(), EVP_sha256(), nullptr));
  std::array<char, 65536> buffer;
  while (stream.read(buffer.data(), buffer.size()) || stream.gcount())
    check(EVP_DigestUpdate(ctx.get(), buffer.data(), stream.gcount()));
  if (stream.bad() || !stream.eof())
    throw std::runtime_error("cannot read " + path.string());
  unsigned char result[32];
  check(EVP_DigestFinal_ex(ctx.get(), result, nullptr));
  if (hexDigest(result) != digest)
    throw std::runtime_error(std::string(role) + " digest: " + path.string());
}

void TemplateBundle::verifyVocab() const {
  if (!vocabDigest.empty())
    verifyFileDigest(vocab, vocabDigest, "vocab");
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
  const auto &arch = j.at("architecture");
  b.hybrid =
      arch == "Qwen3_5ForConditionalGeneration" || arch == "Qwen3_5ForCausalLM";
  const bool hybridFormat =
      b.hybrid && j.contains("layer_types") && j.at("variant") == "f32" &&
      j.at("prefill_ime") == false && j.at("pack_version") == "qwen3_5-f32/1";
  const bool legacyFormat =
      (arch == "Qwen2ForCausalLM" || arch == "Qwen3MoeForCausalLM") &&
      !j.contains("layer_types") && j.at("variant") == "w4g32" &&
      j.at("prefill_ime") == true && j.at("pack_version") == "k3-w4g32/1";
  require(j.at("format") == "bj47.route-b/1" &&
              j.at("activation_type") == "f32" && j.at("kernels") == "k3" &&
              (hybridFormat || legacyFormat),
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
  if (b.hybrid) {
    require((b.spec.tokenizer == "qwen3_5" || b.spec.tokenizer == "qwen3_8") &&
                c.vocabSize > 248069 && c.eosId == 248044 &&
                c.context % c.prefillChunk == 0,
            "hybrid tokenizer/shape");
    b.layerTypes = j.at("layer_types").get<std::vector<std::string>>();
    require(b.layerTypes.size() == size_t(c.totalLayers), "layer_types size");
    for (const auto &type : b.layerTypes)
      require(type == "gdn" || type == "full_attn", "layer_types entry");
  } else {
    require(b.spec.tokenizer == "deepseek_r1" || b.spec.tokenizer == "qwen3",
            "tokenizer");
    b.layerTypes.assign(c.totalLayers, "decoder_layer");
  }
  require(j.at("stop_token_ids").is_array(), "stop_token_ids");
  for (const auto &v : j.at("stop_token_ids")) {
    require(v.is_number_integer() && v >= 0 && v < c.vocabSize,
            "stop_token_ids");
    b.spec.stopTokenIds.push_back(v.get<int>());
  }
  const auto root = std::filesystem::absolute(index).parent_path();
  b.vocab = file(root, j.at("vocab_file"));
  if (j.contains("vocab_sha256"))
    b.vocabDigest = digestField(j, "vocab_sha256");
  if (j.contains("runtime_sha256")) {
    b.runtimeDigest = digestField(j, "runtime_sha256");
    b.runtime = std::filesystem::absolute(root / "libbj47_kernel_runtime.so");
  }
  const auto &layouts = j.at("layouts");
  std::map<std::string, std::string> owners{{"embed", "embed"},
                                            {"lm_head", "head"}};
  for (const auto &type : b.layerTypes)
    owners.emplace(type == "decoder_layer" ? type : type + "_layer",
                   b.hybrid ? type : "decoder");
  std::set<std::filesystem::path> paths;
  for (const auto &k : j.at("templates")) {
    std::string role = k.at("role");
    TemplateEntry e{file(root, k.at("path")), digestField(k, "layout_id"),
                    number(k, "abi_version"), size_t(number(k, "nbytes")),
                    digestField(k, "sha256")};
    require(owners.count(role) &&
                e.layout == digestField(layouts, owners.at(role)) &&
                digestField(k, "shape_id") == b.shape &&
                e.abi == (b.hybrid || role == "lm_head" ? 2 : 1) &&
                paths.insert(e.path).second &&
                b.kernels.emplace(role, e).second,
            role + " template");
  }
  require(b.kernels.size() == owners.size(), "template count");
  for (const auto &w : j.at("weights")) {
    std::string owner = w.at("owner"), bucket = w.at("bucket");
    const bool layer =
        b.hybrid ? owner == "gdn" || owner == "full_attn" : owner == "decoder";
    const bool f32 = bucket == "f32", f16 = bucket == "f16";
    require(layer || owner == "head" || owner == "rope" || owner == "embed",
            "owner");
    const bool validBucket =
        f32 || (b.hybrid && owner == "embed" && f16) ||
        (!b.hybrid && (layer || owner == "head") && bucket == "q4");
    require(validBucket &&
                w.at("dtype") == (f32   ? "f32"
                                  : f16 ? "f16"
                                        : "i8") &&
                layer == w.contains("global_layer_id") &&
                digestField(w, "layout_id") == digestField(layouts, owner),
            "weight bucket/layout");
    int id = layer ? number(w, "global_layer_id", 0) : -1;
    require(id < c.totalLayers &&
                (!b.hybrid || !layer || b.layerTypes[id] == owner),
            "global_layer_id/type");
    const size_t elementBytes = f32 ? 4 : f16 ? 2 : 1;
    const auto &el = w.at("elements");
    require(el.is_number_unsigned() && el > 0 && el <= SIZE_MAX / elementBytes,
            "elements");
    size_t elements = el.get<size_t>(), bytes = elements * elementBytes;
    require(w.at("nbytes").is_number_unsigned() && w.at("nbytes") == bytes,
            "nbytes");
    WeightEntry e{file(root, w.at("path")), elements, bytes,
                  digestField(w, "sha256")};
    require(paths.insert(e.path).second &&
                b.weights.emplace(std::make_tuple(owner, id, bucket), e).second,
            "duplicate weight");
  }
  if (b.hybrid) {
    const bool rope = owners.count("full_attn_layer");
    require(b.weights.size() == size_t(c.totalLayers) + 2 + rope &&
                b.weights.count({"embed", -1, "f32"}) +
                        b.weights.count({"embed", -1, "f16"}) ==
                    1 &&
                b.weights.count({"head", -1, "f32"}) == 1 &&
                b.weights.count({"rope", -1, "f32"}) == size_t(rope),
            "hybrid weight count");
  } else {
    require(b.weights.size() == 2ULL * c.totalLayers + 4, "weight count");
  }
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
