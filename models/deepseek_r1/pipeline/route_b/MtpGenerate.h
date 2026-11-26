// Synchronous speculative steps. All ranks commit the same number of rows.
// Only the last stage owns normalized hidden rows, predictor KV and logits.
#pragma once
#include "../ResidentGenerate.h"
#include "../TemplateBundle.h"

namespace buddy::runtime::pipeline {
namespace mtp {
using detail::BatchClock;
using Json = nlohmann::json;
struct Stream {
  int prompt = 0, position = 0, limit = 0, draft = 0;
  size_t ordinal = 0;
  int proposed = 0, accepted = 0, steps = 0;
  std::string request;
  std::vector<int> generated;
  BatchClock::time_point arrival, first;
  int64_t ttft = 0;
  Json stats() const {
    return {{"mtp_proposed", proposed},
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
  int advance(const int64_t *shifted, const float *hidden, int pos, int rows,
              int valid, bool propose) {
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
    int draft = 0;
    if (propose)
      greedy(result, 1, &draft);
    return draft;
  }
};

inline void generate(const DistributedGenerateOptions &opts, MtpStage &stage,
                     MPI_Comm comm, int rank, int ranks, bool halfEmbed,
                     bool serve, const std::vector<std::string> &prompts,
                     const std::string &ranges) {
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
           {"pids", pids}});
      reader = std::thread([&] { inbox.read(); });
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
    if (cmd[1]) {
      if (rank)
        s = Stream{};
      stage.clearSlot(slot);
      s.prompt = s.position = cmd[1];
      s.limit = cmd[2];
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
        s.draft = link.advance(shifted.data(), h, pos, chunk, valid, final);
      }
    } else {
      int rows = s.position + 1 < cfg.context ? 2 : 1;
      ids[0] = s.generated.back();
      ids[1] = s.draft;
      auto h = link.backbone(ids.data(), s.position, rows, rows);
      int target[2] = {};
      link.greedy(h, rows, target);
      ++s.steps;
      bool accept = rows == 2 && target[0] == s.draft;
      if (rows == 2) {
        ++s.proposed;
        if (accept)
          ++s.accepted;
        else
          stage.rewindOne();
      }
      emit(target[0]);
      if (accept && !done)
        emit(target[1]);
      int committed = accept ? 2 : 1;
      if (!done) {
        shifted[0] = target[0];
        shifted[1] = target[1];
        s.draft = link.advance(shifted.data(), h, s.position, committed,
                               committed, true);
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
