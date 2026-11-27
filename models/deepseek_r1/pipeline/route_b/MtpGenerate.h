// Synchronous speculative steps. All ranks commit the same number of rows.
// Only the last stage owns normalized hidden rows, predictor KV and logits.
#pragma once
#include "../ResidentGenerate.h"
#include "../TemplateBundle.h"
#include <array>
#include <cmath>

namespace buddy::runtime::pipeline {
namespace mtp {
using detail::BatchClock;
using Json = nlohmann::json;
enum class Mode { Mtp, Ngram, DSpark };
struct Options {
  Mode mode = Mode::DSpark;
  int maxDraft = 3;
  double tau = .55, minP = .35;
};

struct Draft {
  std::array<int, 3> tokens{};
  int count = 0;
};

// Longest suffix first, newest completed occurrence first. Only copy
// observed continuation tokens, including when the occurrences overlap.
inline Draft lookup(const std::vector<int64_t> &history, int minN, int limit) {
  Draft draft;
  for (int n = 5; n >= minN; --n) {
    int suffix = static_cast<int>(history.size()) - n;
    for (int i = suffix - 1; i >= 0; --i) {
      if (std::equal(history.begin() + i, history.begin() + i + n,
                     history.begin() + suffix)) {
        draft.count = std::min(limit, suffix - i);
        std::copy_n(history.begin() + i + n, draft.count, draft.tokens.begin());
        return draft;
      }
    }
  }
  return draft;
}

struct Stream {
  int prompt = 0, position = 0, limit = 0;
  Draft draft;
  std::vector<int64_t> history;
  std::array<int, 3> chosen{}, offered{}, matched{};
  std::array<int, 4> prefixes{};
  int lookupSteps = 0;
  // DSpark lookup hits need no predictor. Buffer only its committed inputs
  // until the first miss, with embeddings still represented by token IDs.
  int deferredPos = 0, deferredRows = 0;
  std::vector<float> deferredHidden;
  std::vector<int64_t> deferredIds;
  std::array<int, 3> predictorWidths{};

  size_t ordinal = 0;
  int proposed = 0, accepted = 0, steps = 0;
  std::string request;
  std::vector<int> generated;
  BatchClock::time_point arrival, first;
  int64_t ttft = 0;
  Json stats() const {
    std::array<double, 3> rates{};
    for (int i = 0; i < 3; ++i)
      rates[i] = offered[i] ? double(matched[i]) / offered[i] : 0.;
    return {{"draft_k_histogram", chosen},
            {"draft_depth_proposed", offered},
            {"draft_depth_accepted", matched},
            {"draft_depth_accept_rate", rates},
            {"accepted_prefix_histogram", prefixes},
            {"ngram_steps", lookupSteps},
            {"mtp_k_histogram", predictorWidths},
            {"avg_drafted_k", steps ? double(proposed) / steps : 0.},
            {"avg_accepted_per_step", steps ? double(accepted) / steps : 0.},
            {"mtp_proposed", proposed},
            {"mtp_accepted", accepted},
            {"mtp_accept_rate", proposed ? double(accepted) / proposed : 0.0},
            {"avg_tokens_per_step",
             steps ? double(generated.size() - 1) / steps : 0.0}};
  }
};

class Link {
  MtpStage &stage;
  MPI_Comm comm;
  int rank, last, width, vocab;
  bool halfEmbed, halfWire;
  std::vector<float> received;
  std::vector<uint16_t> wire;
  std::vector<uint16_t> halfEmbeddings;
  std::vector<float> floatEmbeddings;

public:
  Link(MtpStage &stage, MPI_Comm comm, int rank, int last, bool halfEmbed,
       bool halfWire)
      : stage(stage), comm(comm), rank(rank), last(last),
        width(stage.configuration().activationWidth),
        vocab(stage.configuration().vocabSize), halfEmbed(halfEmbed),
        halfWire(halfWire) {
    size_t capacity = size_t(width) * stage.configuration().prefillChunk;
    if (capacity > std::numeric_limits<int>::max() / sizeof(float))
      throw std::length_error("MTP frame size");
    if (rank)
      received.resize(capacity);
    if (last && halfWire)
      wire.resize(capacity);
    if (!rank || rank == last) {
      if (halfEmbed)
        halfEmbeddings.resize(capacity);
      else
        floatEmbeddings.resize(capacity);
    }
  }
  const float *backbone(const int64_t *ids, int pos, int rows, int valid) {
    const int n = rows * width;
    if (rank) {
      void *buf = halfWire ? static_cast<void *>(wire.data()) : received.data();
      mpiCheck(MPI_Recv(buf, n, halfWire ? MPI_UINT16_T : MPI_FLOAT, rank - 1,
                        410, comm, MPI_STATUS_IGNORE));
      if (halfWire)
        decodeInto(wire.data(), n, received.data(), n);
    }
    auto hidden = stage.backbone(rank ? static_cast<void *>(received.data())
                                      : const_cast<int64_t *>(ids),
                                 pos, rows, valid);
    if (rank < last) {
      if (halfWire)
        encodeInto(hidden, n, wire.data(), n);
      mpiCheck(MPI_Send(
          halfWire ? static_cast<const void *>(wire.data()) : hidden, n,
          halfWire ? MPI_UINT16_T : MPI_FLOAT, rank + 1, 410, comm));
    }
    return hidden;
  }
  void greedy(const float *hidden, int rows, int *tokens) {
    if (rank == last) {
      auto logits = stage.linear(hidden, rows);
      for (int i = 0; i < rows; ++i) {
        auto row = logits + size_t(i) * vocab;
        tokens[i] = int(std::max_element(row, row + vocab) - row);
      }
    }
    mpiCheck(MPI_Bcast(tokens, rows, MPI_INT, last, comm));
  }
  // All accepted IDs are already present on every rank. A prefix length and
  // one correction/bonus token suffice even for a four-row verify.
  std::array<int, 2> verify(const float *hidden, const Draft &draft) {
    std::array<int, 2> reply{};
    if (rank == last) {
      auto logits = stage.linear(hidden, draft.count + 1);
      for (int i = 0; i <= draft.count; ++i) {
        const float *row = logits + size_t(i) * vocab;
        int token = int(std::max_element(row, row + vocab) - row);
        if (i == draft.count || token != draft.tokens[i]) {
          reply = {i, token};
          break;
        }
      }
    }
    mpiCheck(MPI_Bcast(reply.data(), 2, MPI_INT, last, comm));
    return reply;
  }
  const float *advance(const int64_t *shifted, const float *hidden, int pos,
                       int rows, int valid) {
    const int bytes = valid * width * (halfEmbed ? 2 : 4);
    void *packed = halfEmbed ? static_cast<void *>(halfEmbeddings.data())
                             : floatEmbeddings.data();
    if (!rank) {
      stage.gatherShifted(shifted, valid, packed);
      if (last)
        mpiCheck(MPI_Send(packed, bytes, MPI_BYTE, last, 411, comm));
    }
    const float *result = nullptr;
    if (rank == last) {
      if (last)
        mpiCheck(
            MPI_Recv(packed, bytes, MPI_BYTE, 0, 411, comm, MPI_STATUS_IGNORE));
      result = stage.draftHidden(packed, halfEmbed, hidden, pos, rows, valid) +
               (valid - 1) * width;
    }
    return result;
  }

  Draft propose(const Options &opt, const float *hidden, int nextPos,
                int limit) {
    Draft draft;
    double cumulative = 1.;
    while (draft.count < limit) {
      int decision[2] = {}; // top token and confidence gate, one reply
      if (rank == last) {
        const float *logits = stage.linear(hidden, 1);
        auto top = std::max_element(logits, logits + vocab);
        decision[0] = int(top - logits);
        decision[1] = 1;
        if (opt.mode == Mode::DSpark) {
          double sum = 0.;
          for (int i = 0; i < vocab; ++i)
            sum += std::exp(double(logits[i]) - *top);
          const double probability = 1. / sum;
          cumulative *= probability;
          decision[1] = probability >= opt.minP && cumulative >= opt.tau;
        }
      }
      mpiCheck(MPI_Bcast(decision, 2, MPI_INT, last, comm));
      // Always verify one candidate. Discard a failed deeper candidate.
      if (decision[1] || !draft.count)
        draft.tokens[draft.count++] = decision[0];
      if (!decision[1] || draft.count == limit)
        break;
      int64_t shifted = decision[0];
      hidden = advance(&shifted, hidden, nextPos++, 1, 1);
    }
    return draft;
  }
};

inline void generate(const DistributedGenerateOptions &opts, MtpStage &stage,
                     MPI_Comm comm, int rank, int ranks, bool halfEmbed,
                     bool serve, const std::vector<std::string> &prompts,
                     const std::string &ranges, const Options &opt) {
  const auto &cfg = stage.configuration();
  const int slots = stage.maxSlots(), chunk = cfg.prefillChunk;
  Link link(stage, comm, rank, ranks - 1, halfEmbed,
            opts.wireDtype == WireDtype::F16);
  std::vector<Stream> streams(slots);
  std::deque<int> free, ready;
  for (int i = 0; i < slots; ++i)
    free.push_back(i);
  detail::ServeInbox inbox;
  std::thread reader;
  if (serve) {
    std::vector<int> pids(ranks);
    int pid = getpid();
    mpiCheck(MPI_Gather(&pid, 1, MPI_INT, pids.data(), 1, MPI_INT, 0, comm));
    if (!rank) {
      detail::emitServe(
          {{"event", "ready"},
           {"slots", slots},
           {"context", cfg.context},
           {"vocab", opts.vocabPath},
           {"wire", opts.wireDtype == WireDtype::F16 ? "f16" : "f32"},
           {"ranges", ranges},
           {"pids", opts.servePids.empty() ? pids : opts.servePids}});
      reader = std::thread([&] { inbox.read(opts.readServeLine); });
    }
  }
  size_t next = 0;
  std::vector<Json> results(prompts.size());
  std::vector<int64_t> shifted(chunk), ids(chunk), promptIds;
  for (;;) {
    // slot, prompt rows (zero for a verification step), token limit.
    int cmd[3] = {-1, 0, 0};
    if (!rank) {
      while (!free.empty()) {
        if (!serve && next >= prompts.size())
          break;
        std::string request;
        try {
          int value = opts.maxTokens;
          auto mode = opts.tokenLimitMode;
          bool structured = false;
          std::string prompt;
          auto arrival = BatchClock::now();
          if (serve) {
            bool finished;
            auto input = inbox.next(ready.empty(), finished);
            if (!input)
              break;
            const auto &j = input->command;
            arrival = input->arrival;
            request = j.value("req_id", std::string{});
            if (request.empty() || j.at("cmd") != "submit")
              throw std::invalid_argument("invalid submit");
            for (auto &s : streams)
              if (s.request == request)
                throw std::invalid_argument("duplicate active req_id");
            bool newTokens = j.contains("max_new_tokens");
            if (newTokens && j.contains("total_tokens"))
              throw std::invalid_argument("choose one token limit");
            auto number =
                j.value(newTokens ? "max_new_tokens" : "total_tokens", Json(0));
            if (!number.is_number_integer() || number < 0 ||
                number > cfg.context)
              throw std::invalid_argument("invalid token limit");
            value = number.get<int>();
            mode = newTokens ? TokenLimitMode::NewTokens
                             : TokenLimitMode::TotalSequenceLength;
            structured = j.contains("chat");
            prompt = structured ? j.at("chat").dump()
                                : j.at("prompt").get<std::string>();
          } else
            prompt = prompts[next++];
          int n =
              encodePrompt(opts, cfg, prompt, promptIds, nullptr, structured);
          int limit =
              resolveGeneratedTokenLimit(value, mode, true, n, cfg.context);
          cmd[0] = free.front();
          free.pop_front();
          cmd[1] = n;
          cmd[2] = limit;
          auto &s = streams[cmd[0]];
          s = Stream{};
          s.request = request;
          s.arrival = arrival;
          s.ordinal = serve ? 0 : next - 1;
          if (serve)
            detail::emitServe({{"event", "accepted"},
                               {"req_id", request},
                               {"slot", cmd[0]},
                               {"prompt_tokens", n}});
          break;
        } catch (const std::bad_alloc &) {
          throw;
        } catch (const std::exception &e) {
          if (!serve)
            throw;
          detail::emitServe(
              {{"event", "error"}, {"req_id", request}, {"message", e.what()}});
        }
      }
      if (cmd[0] < 0 && !ready.empty()) {
        cmd[0] = ready.front();
        ready.pop_front();
      }
    }
    mpiCheck(MPI_Bcast(cmd, 3, MPI_INT, 0, comm));
    int slot = cmd[0];
    if (slot < 0)
      break;
    auto &s = streams[slot];
    stage.setActiveSlot(slot);
    bool done = false;
    auto start = BatchClock::now();
    auto emit = [&](int token) {
      bool first = s.generated.empty();
      s.generated.push_back(token);
      if (opt.mode != Mode::Mtp)
        s.history.push_back(token);
      if (first) {
        s.first = BatchClock::now();
        if (!rank)
          s.ttft = detail::batchElapsedUs(s.arrival);
      }
      if (!rank && serve)
        detail::emitServe({{"event", "token"},
                           {"req_id", s.request},
                           {"slot", slot},
                           {"step", s.generated.size() - 1},
                           {"token_id", token},
                           {"is_prefill", first},
                           {"ttft_us", s.ttft},
                           {"step_us", detail::batchElapsedUs(start)}});
      done = int(s.generated.size()) == s.limit ||
             isStopToken(token, cfg.eosId, opts.stopTokenIds) ||
             int(s.generated.size()) == opts.testEosAtStep;
    };
    // Replay the committed rows with target hidden states. Tentative predictor
    // KV at nextPos and beyond is overwritten before any later causal read.
    auto prepareDraft = [&](const float *h, int pos, int rows, int valid,
                            bool final) {
      const int nextPos = pos + valid;
      int limit = final ? std::min({opt.maxDraft, cfg.context - nextPos - 1,
                                    s.limit - int(s.generated.size()) - 1})
                        : 0;
      s.draft = {};
      if (limit > 0 && opt.mode != Mode::Mtp) {
        s.draft = lookup(s.history, opt.mode == Mode::DSpark ? 3 : 2, limit);
        if (s.draft.count)
          ++s.lookupSteps;
      }
      if (opt.mode == Mode::Ngram || (final && limit <= 0))
        return;
      if (s.draft.count) {
        if (!s.deferredRows)
          s.deferredPos = pos;
        s.deferredRows += valid;
        if (!rank)
          s.deferredIds.insert(s.deferredIds.end(), shifted.begin(),
                               shifted.begin() + valid);
        if (rank == ranks - 1)
          s.deferredHidden.insert(s.deferredHidden.end(), h,
                                  h + size_t(valid) * cfg.activationWidth);
        return;
      }
      for (int offset = 0; offset < s.deferredRows; offset += chunk) {
        int n = std::min(chunk, s.deferredRows - offset);
        link.advance(!rank ? s.deferredIds.data() + offset : nullptr,
                     rank == ranks - 1
                         ? s.deferredHidden.data() +
                               size_t(offset) * cfg.activationWidth
                         : nullptr,
                     s.deferredPos + offset, n <= 4 ? n : chunk, n);
      }
      s.deferredIds.clear();
      s.deferredHidden.clear();
      s.deferredRows = 0;
      const float *lastHidden =
          link.advance(shifted.data(), h, pos, rows, valid);
      if (final) {
        s.draft = link.propose(opt, lastHidden, nextPos, limit);
        ++s.predictorWidths[s.draft.count - 1];
      }
    };
    if (cmd[1]) {
      if (rank)
        s = Stream{};
      stage.clearSlot(slot);
      s.prompt = s.position = cmd[1];
      s.limit = cmd[2];
      if (opt.mode != Mode::Mtp) {
        s.history.resize(s.prompt);
        if (!rank)
          std::copy_n(promptIds.begin(), s.prompt, s.history.begin());
        mpiCheck(MPI_Bcast(s.history.data(), s.prompt, MPI_INT64_T, 0, comm));
      }
      for (int pos = 0; pos < s.prompt; pos += chunk) {
        int valid = std::min(chunk, s.prompt - pos);
        bool final = pos + valid == s.prompt;
        if (!rank)
          for (int i = 0; i < chunk; ++i)
            ids[i] = promptIds[pos + std::min(i, valid - 1)];
        auto h = link.backbone(ids.data(), pos, chunk, valid);
        int token = 0;
        if (final) {
          link.greedy(rank == ranks - 1 ? h + (valid - 1) * cfg.activationWidth
                                        : nullptr,
                      1, &token);
          emit(token);
        }
        if (done || s.limit == 1)
          continue;
        if (!rank)
          for (int i = 0; i < valid; ++i)
            shifted[i] =
                (final && i == valid - 1) ? token : promptIds[pos + i + 1];
        prepareDraft(h, pos, chunk, valid, final);
      }
    } else {
      const int k = s.draft.count, rows = k + 1;
      ids[0] = s.generated.back();
      std::copy_n(s.draft.tokens.begin(), k, ids.begin() + 1);
      auto h = link.backbone(ids.data(), s.position, rows, rows);
      const auto [prefix, correction] = link.verify(h, s.draft);
      ++s.steps;
      s.proposed += k;
      if (k)
        ++s.chosen[k - 1];
      int accepted = 0, committed = 0;
      for (int i = 0; i < k; ++i)
        ++s.offered[i];
      for (int i = 0; i <= prefix; ++i) {
        ++committed;
        shifted[i] = i == prefix ? correction : s.draft.tokens[i];
        emit(static_cast<int>(shifted[i]));
        if (i == prefix)
          break;
        ++accepted;
        ++s.matched[i];
        if (done)
          break;
      }
      s.accepted += accepted;
      if (k)
        ++s.prefixes[accepted];
      stage.commitPrefix(s.position, committed, rows);
      if (!done) {
        prepareDraft(h, s.position, committed, committed, true);
      }
      s.position += committed;
    }
    if (!rank && done) {
      auto j = s.stats();
      if (serve) {
        int tokens = int(s.generated.size()) - 1;
        auto us = detail::batchElapsedUs(s.first);
        bool eos =
            isStopToken(s.generated.back(), cfg.eosId, opts.stopTokenIds) ||
            int(s.generated.size()) == opts.testEosAtStep;
        j.update(Json{{"event", "done"},
                      {"req_id", s.request},
                      {"slot", slot},
                      {"reason", eos ? "eos" : "length"},
                      {"decode_steps", tokens},
                      {"decode_us", us},
                      {"decode_tok_s", us ? tokens * 1e6 / us : 0}});
        detail::emitServe(j);
      } else {
        j["token_ids"] = s.generated;
        j["prompt_tokens"] = s.prompt;
        results[s.ordinal] = j;
      }
      s.request.clear();
      free.push_front(slot);
    } else if (!rank)
      ready.push_back(slot);
  }
  if (reader.joinable())
    reader.join();
  if (!rank && !serve)
    std::cout << (results.size() == 1 ? results.front()
                                      : Json{{"streams", results}})
                     .dump()
              << std::endl;
}
} // namespace mtp
} // namespace buddy::runtime::pipeline
