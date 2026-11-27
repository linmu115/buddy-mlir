#pragma once

#include "../TemplateBundle.h"
#include "buddy/runtime/pipeline/ActivationWire.h"
#include <climits>
#include <fstream>
#include <mpi.h>
#include <nlohmann/json.hpp>

namespace buddy::runtime {
// The endpoint runs the existing generation scheduler on MPI_COMM_SELF. Its
// peer receives body work only. Both ranks enter identical layer collectives.
class TensorParallel {
  enum Operation { Stop, Forward, Clear, Rewind };
  struct Command {
    int op, slot, position, capacity, valid;
  };
  MPI_Comm comm;
  int rank, peer, width, chunk;
  bool half;
  std::vector<uint16_t> sendHalf, recvHalf;
  std::vector<float> remote;

  static void check(int rc) {
    if (rc != MPI_SUCCESS)
      throw std::runtime_error("TP MPI operation failed");
  }
  void send(Command c) {
    check(MPI_Send(&c, sizeof(c), MPI_BYTE, peer, 530, comm));
  }
  static int32_t reduce(void *ctx, float *x, int64_t n) {
    auto &self = *static_cast<TensorParallel *>(ctx);
    if (n < 1 || n > INT_MAX)
      return -1;
    if (!self.half)
      return MPI_Allreduce(MPI_IN_PLACE, x, int(n), MPI_FLOAT, MPI_SUM,
                           self.comm);
    // Two ranks need one full-duplex exchange. Round both operands to f16
    // then sum in f32, so both replicas enter the next norm with equal values.
    pipeline::encodeInto(x, n, self.sendHalf.data(), self.sendHalf.size());
    int rc = MPI_Sendrecv(self.sendHalf.data(), n, MPI_UINT16_T, self.peer, 532,
                          self.recvHalf.data(), n, MPI_UINT16_T, self.peer, 532,
                          self.comm, MPI_STATUS_IGNORE);
    if (rc != MPI_SUCCESS)
      return rc;
    pipeline::decodeInto(self.sendHalf.data(), n, x, n);
    pipeline::decodeInto(self.recvHalf.data(), n, self.remote.data(),
                         self.remote.size());
    for (int i = 0; i < n; ++i)
      x[i] += self.remote[i];
    return 0;
  }

public:
  TensorHooks hooks{};
  TensorParallel(MPI_Comm c, int r, const TemplateBundle &b, bool fp16)
      : comm(c), rank(r), peer(1 - r), width(b.config.activationWidth),
        chunk(b.config.prefillChunk), half(fp16) {
    if (int64_t(width) * chunk > INT_MAX)
      throw std::length_error("TP frame exceeds MPI count");
    if (half) {
      sendHalf.resize(width * chunk);
      recvHalf.resize(width * chunk);
      remote.resize(width * chunk);
    }
    hooks.reduce = reduce;
    hooks.context = this;
    if (rank == b.tpHost) {
      hooks.dispatch = [this](const float *x, int slot, int pos, int rows,
                              int valid) {
        send({Forward, slot, pos, rows, valid});
        check(MPI_Send(x, valid * width, MPI_FLOAT, peer, 531, comm));
      };
      hooks.clear = [this](int slot) { send({Clear, slot, 0, 0, 0}); };
      hooks.rewind = [this](int slot) { send({Rewind, slot, 0, 0, 0}); };
    }
  }
  void reportMemory() const {
    if (!std::getenv("BUDDY_PP_OBSERVE"))
      return;
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line))
      if (line.rfind("VmHWM:", 0) == 0 || line.rfind("VmRSS:", 0) == 0 ||
          line.rfind("VmSwap:", 0) == 0)
        std::fprintf(stderr, "tp_memory rank=%d %s\n", rank, line.c_str());
  }
  void stop() { send({Stop, 0, 0, 0, 0}); }
  void worker(MtpStage &stage) {
    std::vector<float> frame(size_t(width) * chunk);
    for (;;) {
      Command cmd{};
      check(MPI_Recv(&cmd, sizeof(cmd), MPI_BYTE, peer, 530, comm,
                     MPI_STATUS_IGNORE));
      if (cmd.op == Stop)
        return;
      stage.setActiveSlot(cmd.slot);
      switch (cmd.op) {
      case Clear:
        stage.clearSlot(cmd.slot);
        break;
      case Rewind:
        stage.rewindOne();
        break;
      case Forward:
        if (cmd.valid < 1 || cmd.valid > cmd.capacity || cmd.capacity > chunk)
          throw std::runtime_error("invalid TP frame");
        check(MPI_Recv(frame.data(), width * cmd.valid, MPI_FLOAT, peer, 531,
                       comm, MPI_STATUS_IGNORE));
        stage.backbone(frame.data(), cmd.position, cmd.capacity, cmd.valid);
        break;
      default:
        throw std::runtime_error("invalid TP command");
      }
    }
  }
  static TemplateBundle load(const std::filesystem::path &path, int rank,
                             int size, const std::string &options,
                             MPI_Comm comm) {
    if (size != 2)
      throw std::invalid_argument("TP requires two MPI ranks");
    std::ifstream stream(path / "tp.json");
    auto index = nlohmann::json::parse(stream);
    auto ids = index.at("ranks").get<std::vector<std::string>>();
    if (index.at("format") != "bj47.tp/1" || ids.size() != 2)
      throw std::invalid_argument("invalid TP index");
    auto b = TemplateBundle::load(path / ("rank" + std::to_string(rank)) /
                                  "route-b.json");
    if (b.tpRank != rank || ids.at(rank) != b.id)
      throw std::invalid_argument("invalid TP member");
    auto plan = index.dump() + b.tpSource + std::to_string(b.tpHost) +
                nlohmann::json(b.tpHeads).dump() + options;
    auto hash = contentDigest(plan.data(), plan.size());
    char other[64];
    check(MPI_Sendrecv(hash.data(), 64, MPI_CHAR, 1 - rank, 529, other, 64,
                       MPI_CHAR, 1 - rank, 529, comm, MPI_STATUS_IGNORE));
    if (hash.compare(0, 64, other, 64))
      throw std::invalid_argument(
          "TP ranks disagree on bundle or launch options");
    return b;
  }
};
} // namespace buddy::runtime
