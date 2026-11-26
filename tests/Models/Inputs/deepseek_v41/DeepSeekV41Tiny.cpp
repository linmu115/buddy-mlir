#include "../../../../models/deepseek_v41/TinyBackbone.h"
using namespace buddy::v4;
using namespace buddy::v4::tiny;

struct Differential : Runner {
  const Store &oracle;
  std::map<string, double> diff;
  std::map<string, Vec> outputs;
  std::map<string, int> count;
  bool capture;
  Differential(const Weights &w, const Store &ref, bool record)
      : Runner(w,
               [this](auto &p, auto &name, auto &v) { compare(p, name, v); }),
        oracle(ref), capture(record) {}
  void compare(const string &prefix, const string &name, const Rows &rows) {
    auto &t = oracle[prefix + name];
    size_t i = 0;
    auto &err = diff[name];
    for (auto &row : rows)
      for (float v : row) {
        if (i >= t.bytes / (t.type == "int" ? 8 : 4))
          throw std::runtime_error("trace overflow");
        double expected = t.type == "int" ? double(t.id()[i]) : t.f()[i];
        err = std::max(err, std::isfinite(v) ? std::abs(v - expected)
                                             : double(INFINITY));
        i++;
      }
    if (i != t.bytes / (t.type == "int" ? 8 : 4))
      throw std::runtime_error("trace shape " + prefix + name);
    if (capture) {
      auto &a = outputs[prefix + name];
      for (auto &row : rows)
        a.insert(a.end(), row.begin(), row.end());
    }
    count[name]++;
  }
};
Rows referenceRows(const Tensor &t, int rows) {
  int stride = t.bytes / (4 * rows);
  Rows result;
  for (int i = 0; i < rows; i++)
    result.emplace_back(t.f() + i * stride, t.f() + (i + 1) * stride);
  return result;
}
void isolate(Differential &runner, const Store &op) {
  // All-invalid block followed by one valid slot exercises the online max
  // transition. With a unit-valued key and equally scored sink the answer is
  // .5.
  Vec query(32), val(32, 1), output(32);
  float sink = 0;
  std::vector<const float *> slots(65, nullptr);
  sparse(query.data(), slots, 1, 32, &sink, output.data());
  if (output != Vec(32, 0))
    throw std::runtime_error("all-masked sparse attention");
  slots.back() = val.data();
  sparse(query.data(), slots, 1, 32, &sink, output.data());
  if (output != Vec(32, .5f))
    throw std::runtime_error("online softmax block transition");
  // A DEAD token fills that and all earlier slots with the compressed pad ID.
  int64_t history[] = {7, -1, 5, 6}, mult[] = {3, 5, 7, 11},
          prime[] = {17, 19, 23}, offset[] = {0, 17, 36};
  if (hashRows(history, 3, 2, 4, 1, mult, prime, offset) !=
      Ids{(18 ^ 25) % 17, 17 + (18 ^ 25 ^ 14) % 19,
          36 + (18 ^ 25 ^ 14 ^ 22) % 23})
    throw std::runtime_error("Engram DEAD lookback");
  if (!(swiglu(-20, 20, 1, 10) > -1e-5f &&
        std::isfinite(swiglu(200, 20, 1, 10))))
    throw std::runtime_error("one-sided gate cap");
  Shared shared;
  for (int pos : {0, 37, 38}) {
    auto &ids = op["ids/" + std::to_string(pos)];
    int rows = ids.shape[1];
    for (int i = 0; i < rows; i++)
      runner.history.push_back(runner.w["hash.token_map"].id()[ids.id()[i]]);
    for (int l = 0; l < 6; l++) {
      string p = "layers." + std::to_string(l) + ".";
      for (string k : {"attn", "ffn", "engram"}) {
        if (k == "engram" && l != 1 && l != 4)
          continue;
        string base = "op/" + std::to_string(pos) + "/" + std::to_string(l) +
                      "/" + k + "/";
        auto x = referenceRows(op[base + "in"], rows);
        if (k == "attn")
          x = runner.attention(l, pos, p + "attn.", x, shared);
        if (k == "ffn")
          for (auto &v : x)
            v = runner.moe(p + "ffn.", v);
        if (k == "engram")
          for (int i = 0; i < rows; i++)
            runner.engram(l, pos + i, p + "engram.", x[i]);
        auto expected = referenceRows(op[base + "out"], rows);
        for (int i = 0; i < rows; i++)
          for (size_t j = 0; j < x[i].size(); j++)
            runner.diff[k] = std::max(
                runner.diff[k], double(std::abs(x[i][j] - expected[i][j])));
      }
    }
  }
  string p = "layers.0.hc_attn_";
  auto m = mixes(op["mhc/in"].f(), 128, runner.w[p + "fn"].f(),
                 runner.w[p + "scale"].f(), runner.w[p + "base"].f());
  for (int i = 0; i < 24; i++) {
    float val = i < 4   ? m.pre[i]
                : i < 8 ? m.post[i - 4]
                        : m.comb[(i - 8) / 4][i % 4];
    float ref = i < 4   ? op["mhc/pre"].f()[i]
                : i < 8 ? op["mhc/post"].f()[i - 4]
                        : op["mhc/comb"].f()[i - 8];
    runner.diff["mhc_tiny"] = std::max(
        runner.diff["mhc_tiny"],
        std::isfinite(val) ? double(std::abs(val - ref)) : double(INFINITY));
  }
}
int main(int argc, char **argv) {
  if (argc == 3 && string(argv[1]) == "--large-table") {
    Weights table(argv[2], "large");
    Ids rows(24);
    std::iota(rows.begin(), rows.end(), 1 << 24);
    auto y = table.gatherEngram("embed", rows);
    for (int j = 0; j < 24; j++)
      for (int k = 0; k < 256; k++)
        if (y[j * 256 + k] != std::ldexp(j % 2 ? 1.125f : 1.f, k / 32 - 3))
          return 1;
    return 0;
  }
  if (argc != 2)
    return 2;
  Weights weights(argv[1], "weights");
  Store expected(argv[1], "expected");
  Differential base(weights, expected, true);
  Ids prompt(37);
  for (int i = 0; i < 37; i++)
    prompt[i] = (i * 13 + 5) % 128;
  auto tokens = base.run(prompt, 16);
  Differential isolated(weights, expected, false);
  Store operators(argv[1], "operators");
  isolate(isolated, operators);
  base.diff.insert(isolated.diff.begin(), isolated.diff.end());
  int greedyMatches = 0;
  for (int step = 0; step <= 16; step++) {
    const auto &expectedLogits =
        expected[std::to_string(step ? 36 + step : 0) + "/logits"];
    int token = std::max_element(expectedLogits.f(), expectedLogits.f() + 128) -
                expectedLogits.f();
    if (step)
      greedyMatches += token == tokens[step];
    else if (token != tokens[step])
      return 1;
  }
  bool split = true;
  for (Ids cuts : {Ids{1, 3}, Ids{2}}) {
    Differential part(weights, expected, true);
    split &= part.run(prompt, 16, cuts) == tokens &&
             part.outputs.size() == base.outputs.size();
    for (auto &[name, ref] : base.outputs) {
      auto &cur = part.outputs.at(name);
      split &= cur.size() == ref.size() &&
               !std::memcmp(cur.data(), ref.data(), ref.size() * sizeof(float));
    }
  }
  std::cout << "{\"greedy_matches\":" << greedyMatches
            << ",\"split_exact\":" << (split ? "true" : "false")
            << ",\"tokens\":[";
  for (size_t i = 0; i < tokens.size(); i++)
    std::cout << (i ? "," : "") << tokens[i];
  std::cout << "],\"max_abs\":{";
  bool ok = split && greedyMatches == 16, first = true;
  for (auto &[k, e] : base.diff) {
    std::cout << (first ? "" : ",") << "\"" << k << "\":" << e;
    first = false;
    ok &= e < 1e-4;
  }
  std::cout << "}}\n";
  return ok ? 0 : 1;
}
