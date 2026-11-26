#pragma once

#include "DistributedGenerate.h"

#include <condition_variable>
#include <mutex>
#include <nlohmann/json.hpp>
#include <thread>
#include <unistd.h>

namespace buddy::runtime::pipeline {
namespace detail {
using ServeJson = nlohmann::json;

struct ServeInput {
  ServeJson command;
  BatchClock::time_point arrival;
};

class ServeInbox {
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<ServeInput> pending;
  std::exception_ptr failure;
  bool closed = false;

public:
  void read() {
    try {
      for (std::string line; std::getline(std::cin, line);) {
        auto arrival = BatchClock::now();
        auto command = ServeJson::parse(line, nullptr, false);
        if (command.is_object() && command.contains("cmd") &&
            command["cmd"] == "shutdown")
          break;
        {
          std::lock_guard<std::mutex> guard(mutex);
          pending.push_back({std::move(command), arrival});
        }
        changed.notify_one();
      }
    } catch (...) {
      std::lock_guard<std::mutex> guard(mutex);
      failure = std::current_exception();
    }
    {
      std::lock_guard<std::mutex> guard(mutex);
      closed = true;
    }
    changed.notify_one();
  }

  std::optional<ServeInput> next(bool wait, bool &finished) {
    std::unique_lock<std::mutex> guard(mutex);
    if (wait)
      changed.wait(guard, [&] { return closed || !pending.empty(); });
    if (failure)
      std::rethrow_exception(failure);
    finished = closed && pending.empty();
    if (pending.empty())
      return std::nullopt;
    auto input = std::move(pending.front());
    pending.pop_front();
    return input;
  }
};

struct ResidentStream {
  BatchStream batch;
  std::string request;
  int step = 0;
  BatchClock::time_point arrival, firstToken, sent;
  int64_t ttft = 0;
};

inline void emitServe(ServeJson event) {
  event["ts_us"] = std::chrono::duration_cast<std::chrono::microseconds>(
                       BatchClock::now().time_since_epoch())
                       .count();
  std::cout << event.dump() << '\n' << std::flush;
  if (!std::cout)
    throw std::runtime_error("cannot write serve event");
}

inline void validateReply(const StageConfig &config,
                          const BatchCtrlReply &reply, size_t slots) {
  if (reply.slot < 0 || size_t(reply.slot) >= slots || reply.token < 0 ||
      reply.token >= config.vocabSize || (reply.stop != 0 && reply.stop != 1) ||
      reply.reserved)
    throw std::runtime_error("unexpected resident control reply");
}

template <typename WireElement>
BatchCtrlReply residentPrefill(const DistributedGenerateOptions &opts,
                               Stage &stage, BatchLink<WireElement> &link,
                               int slot, BatchStream &stream, int rank,
                               int last, size_t slots, int chunks,
                               const std::vector<int64_t> &tokens) {
  const auto &config = stage.configuration();
  const PromptFramePlan plan(
      stream.promptRows, config.prefillChunk, config.context,
      resolveWireSpec(config.dtype, opts.wireDtype).version,
      opts.appendOnlyPrefill);
  if (chunks != 0 && chunks != plan.size())
    throw std::runtime_error("unexpected resident prefill chunk count");

  link.drain();
  stage.clearSlot(slot);
  stage.setActiveSlot(slot);
  link.startPrefill(slot, stream.promptRows, stream.limit, plan.size());
  if (rank == last) {
    stream.prefill.sampler.emplace(opts.samplerConfig);
    stream.generated.reserve(stream.limit);
  }

  std::vector<int64_t> padded(rank == 0 ? config.prefillChunk : 0);
  const void *sampledRow = nullptr;
  for (int i = 0; i < plan.size(); ++i) {
    const auto frame = plan[i];
    void *input;
    if (rank == 0) {
      if (frame.prefill) {
        for (int row = 0; row < frame.header.rows; ++row)
          padded[row] = tokens[std::min(frame.header.pos + row,
                                        plan.prefillTokens() - 1)];
      } else {
        padded.front() = tokens[frame.header.pos];
      }
      input = padded.data();
    } else {
      const auto incoming = link.receiveActivation(slots);
      if (incoming.kind != BatchActKind::PrefillChunk ||
          incoming.slot != slot || incoming.pos != frame.header.pos ||
          incoming.rows != frame.header.rows)
        throw std::runtime_error("unexpected resident prefill frame");
      input = link.activationData(incoming.rows);
    }
    const auto *output =
        runPromptFrame(stage, input, frame, opts.appendOnlyPrefill);
    if (rank < last)
      link.sendActivation(slot, frame.header.pos, output, frame.header.rows,
                          BatchActKind::PrefillChunk);
    else if (frame.needLogits)
      sampledRow = output;
  }

  if (rank == last) {
    const auto reply = sampleBatchToken(opts, config, slot, stream, sampledRow);
    stream.done = reply.stop;
    if (rank != 0)
      link.sendReply(reply);
    return reply;
  }
  if (rank == 0)
    return link.receiveReply();
  return {};
}

template <typename WireElement>
void runResident(const DistributedGenerateOptions &opts, Stage &stage,
                 MPI_Comm comm, int rank, int ranks, int slots,
                 const std::string &ranges) {
  const auto &config = stage.configuration();
  const int last = ranks - 1;
  std::vector<ResidentStream> streams(slots);
  BatchLink<WireElement> link(config.activationWidth, slots, comm, rank, last,
                              config.prefillChunk);
  ServeInbox inbox;
  std::thread reader;
  try {
    std::optional<buddy::Text<size_t, 2>> tokenizer;
    if (rank == 0) {
      tokenizer.emplace("");
      tokenizer->loadVocab(opts.vocabPath);
    }
    // The ready event follows all model, KV and wire-buffer allocations.
    std::vector<int> pids(rank == 0 ? ranks : 0);
    const int pid = getpid();
    mpiCheck(MPI_Gather(&pid, 1, MPI_INT, pids.data(), 1, MPI_INT, 0, comm));
    if (rank != 0) {
      for (;;) {
        const auto frame = link.receiveActivation(slots);
        if (frame.kind == BatchActKind::Shutdown)
          break;
        auto &stream = streams[frame.slot].batch;
        if (frame.kind == BatchActKind::PrefillStart) {
          if (frame.pos < 1 || frame.pos > config.context || frame.limit < 1 ||
              frame.limit > config.context - frame.pos + 1 || frame.rows < 1)
            throw std::runtime_error("invalid resident prefill start");
          stream = BatchStream{};
          stream.pos = stream.promptRows = frame.pos;
          stream.limit = frame.limit;
          residentPrefill(opts, stage, link, frame.slot, stream, rank, last,
                          slots, frame.rows, {});
          continue;
        }

        if (frame.kind != BatchActKind::Decode || stream.promptRows == 0 ||
            stream.done || frame.pos != stream.pos ||
            stream.pos >= config.context)
          throw std::runtime_error("unexpected resident decode position");
        const auto *output = decodeBatchSlot(opts, stage, rank, frame.slot,
                                             stream, link.activationData());
        if (rank == last) {
          const auto reply =
              sampleBatchToken(opts, config, frame.slot, stream, output);
          stream.done = reply.stop;
          link.sendReply(reply);
        } else {
          link.sendActivation(frame.slot, stream.pos, output);
        }
        ++stream.pos;
      }
      link.drainAndShutdown();
      return;
    }

    emitServe({{"event", "ready"},
               {"slots", slots},
               {"context", config.context},
               {"vocab", opts.vocabPath},
               {"wire", opts.wireDtype == WireDtype::F16 ? "f16" : "f32"},
               {"ranges", ranges},
               {"pids", pids}});
    reader = std::thread([&] { inbox.read(); });

    std::deque<int> free, ready;
    for (int slot = 0; slot < slots; ++slot)
      free.push_back(slot);
    int inFlight = 0;
    auto complete = [&](const BatchCtrlReply &reply, bool prefill) {
      validateReply(config, reply, streams.size());
      auto &stream = streams[reply.slot];
      if (stream.request.empty() || stream.batch.inFlight == prefill)
        throw std::runtime_error("unexpected resident reply slot");
      if (prefill) {
        stream.firstToken = BatchClock::now();
        stream.ttft = batchElapsedUs(stream.arrival);
      } else {
        --inFlight;
        ++stream.batch.pos;
        stream.batch.inFlight = false;
      }
      stream.batch.token = reply.token;
      const int step = stream.step++;
      emitServe({{"event", "token"},
                 {"req_id", stream.request},
                 {"slot", reply.slot},
                 {"token_id", reply.token},
                 {"step", step},
                 {"is_prefill", prefill},
                 {"ttft_us", stream.ttft},
                 {"step_us", batchElapsedUs(stream.sent)}});
      if (reply.stop) {
        const auto time = step ? batchElapsedUs(stream.firstToken) : 0;
        const bool eos =
            isStopToken(reply.token, config.eosId, opts.stopTokenIds) ||
            (opts.testEosAtStep > 0 && step + 1 == opts.testEosAtStep);
        emitServe({{"event", "done"},
                   {"req_id", stream.request},
                   {"slot", reply.slot},
                   {"reason", eos ? "eos" : "length"},
                   {"decode_steps", step},
                   {"decode_us", time},
                   {"decode_tok_s", time ? 1e6 * step / time : 0.0}});
        stream = ResidentStream{};
        free.push_front(reply.slot);
      } else {
        ready.push_back(reply.slot);
      }
    };
    auto receive = [&] { complete(link.receiveReply(), false); };

    auto admit = [&](const ServeInput &input) {
      const auto &command = input.command;
      std::string request;
      std::vector<int64_t> tokens;
      int rows, limit;
      try {
        if (!command.is_object())
          throw std::invalid_argument("expected a JSON command object");
        request = command.value("req_id", std::string{});
        if (command.value("cmd", std::string{}) != "submit" || request.empty())
          throw std::invalid_argument(
              "expected submit with a non-empty req_id");
        for (const auto &stream : streams)
          if (stream.request == request)
            throw std::invalid_argument("duplicate active req_id");
        const bool newTokens = command.contains("max_new_tokens");
        if (newTokens && command.contains("total_tokens"))
          throw std::invalid_argument("choose one token limit");
        const auto field = newTokens ? "max_new_tokens" : "total_tokens";
        const auto value = command.value(field, ServeJson(0));
        if (!value.is_number_integer() || value < 0 || value > config.context)
          throw std::invalid_argument("invalid token limit");
        const auto prompt = command.at("prompt").get<std::string>();
        rows = tokenizePrompt(prompt, opts.vocabPath, config.context,
                              opts.tokenizer, tokens, &*tokenizer);
        limit = resolveGeneratedTokenLimit(
            value.get<int>(),
            newTokens ? TokenLimitMode::NewTokens
                      : TokenLimitMode::TotalSequenceLength,
            true, rows, config.context);
      } catch (const std::bad_alloc &) {
        throw;
      } catch (const std::exception &e) {
        emitServe(
            {{"event", "error"}, {"req_id", request}, {"message", e.what()}});
        return;
      }
      const int slot = free.front();
      free.pop_front();
      auto &stream = streams[slot];
      stream.request = std::move(request);
      stream.batch.promptRows = stream.batch.pos = rows;
      stream.batch.limit = limit;
      stream.arrival = input.arrival;
      emitServe({{"event", "accepted"},
                 {"req_id", stream.request},
                 {"slot", slot},
                 {"prompt_tokens", rows}});
      stream.sent = BatchClock::now();
      const auto reply = residentPrefill(opts, stage, link, slot, stream.batch,
                                         rank, last, slots, 0, tokens);
      if (reply.slot != slot)
        throw std::runtime_error("unexpected resident prefill reply slot");
      complete(reply, true);
    };

    for (;;) {
      while (inFlight && link.replyAvailable())
        receive();
      if (!free.empty()) {
        const bool idle = ready.empty() && inFlight == 0;
        bool finished;
        if (auto input = inbox.next(idle, finished)) {
          // Stop issuing Decode frames at admission. All existing replies
          // are consumed before any rank enters the new prefill sequence.
          while (inFlight)
            receive();
          admit(*input);
          continue;
        }
        if (idle && finished)
          break;
      }

      if (ready.empty()) {
        if (!inFlight)
          throw std::logic_error("resident slots have no work");
        receive();
        continue;
      }
      const int slot = ready.front();
      ready.pop_front();
      auto &stream = streams[slot];
      stream.sent = BatchClock::now();
      const auto *output = decodeBatchSlot(opts, stage, rank, slot,
                                           stream.batch, &stream.batch.token);
      stream.batch.inFlight = true;
      ++inFlight;
      if (ranks == 1)
        complete(sampleBatchToken(opts, config, slot, stream.batch, output),
                 false);
      else
        link.sendActivation(slot, stream.batch.pos, output);
    }
    reader.join();
    link.drainAndShutdown();
  } catch (const std::exception &e) {
    abortBatch(comm, rank, "resident serve", e.what());
  } catch (...) {
    abortBatch(comm, rank, "resident serve", "unknown exception");
  }
}
} // namespace detail

inline void generateResident(const DistributedGenerateOptions &opts,
                             Stage &stage, MPI_Comm comm, int rank, int ranks,
                             int slots, const std::string &ranges) {
  const auto &config = stage.configuration();
  if (slots < 1 || slots > stage.maxSlots() ||
      config.dtype != ActivationType::F32 || config.prefillChunk < 1)
    throw std::invalid_argument(
        "serve requires a chunked f32 stage with KV slots");
  if (ranks > 1 && opts.wireDtype == WireDtype::F16)
    detail::runResident<uint16_t>(opts, stage, comm, rank, ranks, slots,
                                  ranges);
  else
    detail::runResident<float>(opts, stage, comm, rank, ranks, slots, ranges);
}
} // namespace buddy::runtime::pipeline
