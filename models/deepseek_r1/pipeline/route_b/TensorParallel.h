#pragma once

#include "../TemplateBundle.h"
#include "TcpSpinAllReduce.h"
#include "buddy/runtime/pipeline/ActivationWire.h"
#include <climits>
#include <fstream>
#include <iostream>
#include <mpi.h>
#include <nlohmann/json.hpp>

namespace buddy::runtime {
// The endpoint runs the existing generation scheduler on MPI_COMM_SELF. Its
// peer receives body work only. Both ranks enter identical layer collectives.
class TensorParallel {
  enum Operation { Stop, Forward, Clear, Commit };
  struct Command {
    int op, slot, position, capacity, valid;
    // Verifier input IDs (current token followed by up to three candidates).
    int64_t ids[4]{};
  };
  MPI_Comm comm;
  int rank, peer, width, chunk;
  bool half;
  std::unique_ptr<TpSocket> socket;
  std::vector<uint16_t> sendHalf, recvHalf;
  std::vector<float> remote;

  static void check(int rc) {
    if (rc != MPI_SUCCESS)
      throw std::runtime_error("TP MPI operation failed");
  }
  void send(Command c) {
    if (socket) {
      socket->transfer(&c, sizeof(c), nullptr, 0);
      return;
    }
    check(MPI_Send(&c, sizeof(c), MPI_BYTE, peer, 530, comm));
  }
  static int32_t reduce(void *ctx, float *x, int64_t n) {
    auto &self = *static_cast<TensorParallel *>(ctx);
    if (n < 1 || n > int64_t(self.width) * self.chunk)
      return -1;
    if (self.half) {
      pipeline::encodeInto(x, n, self.sendHalf.data(), self.sendHalf.size());
      if (self.socket) {
        size_t bytes = size_t(n) * sizeof(uint16_t);
        self.socket->transfer(self.sendHalf.data(), bytes, self.recvHalf.data(),
                              bytes);
      } else {
        int rc = MPI_Sendrecv(self.sendHalf.data(), n, MPI_UINT16_T, self.peer,
                              532, self.recvHalf.data(), n, MPI_UINT16_T,
                              self.peer, 532, self.comm, MPI_STATUS_IGNORE);
        if (rc != MPI_SUCCESS)
          return rc;
      }
      pipeline::decodeInto(self.sendHalf.data(), n, x, n);
      pipeline::decodeInto(self.recvHalf.data(), n, self.remote.data(),
                           self.remote.size());
    } else if (self.socket) {
      size_t bytes = size_t(n) * sizeof(float);
      self.socket->transfer(x, bytes, self.remote.data(), bytes);
    } else {
      return MPI_Allreduce(MPI_IN_PLACE, x, int(n), MPI_FLOAT, MPI_SUM,
                           self.comm);
    }
    for (int i = 0; i < n; ++i)
      x[i] += self.remote[i];
    return 0;
  }

public:
  TensorHooks hooks{};
  TensorParallel(MPI_Comm c, int r, const TemplateBundle &b, bool fp16,
                 TpTransport mode = TpTransport::Mpi,
                 const std::string &iface = {})
      : comm(c), rank(r), peer(1 - r), width(b.config.activationWidth),
        chunk(b.config.prefillChunk), half(fp16) {
    if (width <= 0 || chunk <= 0 || int64_t(width) * chunk > INT_MAX)
      throw std::length_error("invalid TP frame capacity");
    if (mode != TpTransport::Mpi)
      socket = std::make_unique<TpSocket>(comm, rank,
                                          mode == TpTransport::TcpSpin, iface);
    if (half || socket)
      remote.resize(width * chunk);
    if (half) {
      sendHalf.resize(width * chunk);
      recvHalf.resize(width * chunk);
    }
    hooks.reduce = reduce;
    hooks.context = this;
    if (rank == b.tpHost) {
      hooks.dispatch = [this](const float *x, const int64_t *ids, int slot,
                              int pos, int rows, int valid) {
        Command c{Forward, slot, pos, rows, valid};
        if (rows > 1 && rows != chunk)
          std::copy_n(ids, rows, c.ids);
        send(c);
        if (socket)
          socket->transfer(x, size_t(valid) * width * sizeof(float), nullptr,
                           0);
        else
          check(MPI_Send(x, valid * width, MPI_FLOAT, peer, 531, comm));
      };
      hooks.clear = [this](int slot) { send({Clear, slot, 0, 0, 0}); };
      hooks.commit = [this](int slot, int pos, int kept, int rows) {
        send({Commit, slot, pos, rows, kept});
      };
    }
  }
  // Hydra sends stdin to world rank zero. Only the optional gateway relay
  // and its matching endpoint reader use MPI from a second thread.
  void relayInput() {
    for (std::string line; std::getline(std::cin, line);) {
      if (line.size() >= INT_MAX)
        throw std::length_error("TP request exceeds MPI count");
      check(MPI_Send(line.c_str(), int(line.size()) + 1, MPI_CHAR, peer, 533,
                     comm));
      auto request = nlohmann::json::parse(line, nullptr, false);
      if (request.is_object() && request.contains("cmd") &&
          request["cmd"] == "shutdown")
        return;
    }
    check(MPI_Send(nullptr, 0, MPI_CHAR, peer, 533, comm));
  }
  bool receiveInput(std::string &line) {
    MPI_Status status;
    check(MPI_Probe(peer, 533, comm, &status));
    int bytes;
    check(MPI_Get_count(&status, MPI_CHAR, &bytes));
    line.resize(bytes);
    check(MPI_Recv(line.data(), bytes, MPI_CHAR, peer, 533, comm,
                   MPI_STATUS_IGNORE));
    if (!bytes)
      return false;
    line.pop_back();
    return true;
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
    Command pending{};
    for (;;) {
      Command cmd{};
      if (socket)
        socket->transfer(nullptr, 0, &cmd, sizeof(cmd));
      else
        check(MPI_Recv(&cmd, sizeof(cmd), MPI_BYTE, peer, 530, comm,
                       MPI_STATUS_IGNORE));
      if (cmd.op == Stop)
        return;
      stage.setActiveSlot(cmd.slot);
      switch (cmd.op) {
      case Clear:
        stage.clearSlot(cmd.slot);
        break;
      case Commit:
        if (pending.op != Forward || pending.slot != cmd.slot ||
            pending.position != cmd.position ||
            pending.capacity != cmd.capacity)
          throw std::runtime_error("TP commit does not match verifier frame");
        stage.commitPrefix(cmd.position, cmd.valid, cmd.capacity);
        pending = {};
        break;
      case Forward:
        if (cmd.valid < 1 || cmd.valid > cmd.capacity || cmd.capacity > chunk)
          throw std::runtime_error("invalid TP frame");
        if (cmd.capacity > 1 && cmd.capacity != chunk) {
          if (cmd.capacity > int(std::size(cmd.ids)))
            throw std::runtime_error("invalid TP verifier capacity");
          for (int i = 0; i < cmd.valid; ++i)
            if (cmd.ids[i] < 0 || cmd.ids[i] >= stage.configuration().vocabSize)
              throw std::runtime_error("invalid TP verifier token");
          pending = cmd;
        }
        if (socket)
          socket->transfer(nullptr, 0, frame.data(),
                           size_t(width) * cmd.valid * sizeof(float));
        else
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
                             MPI_Comm comm,
                             TpTransport mode = TpTransport::Mpi) {
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
                nlohmann::json(b.tpHeads).dump() +
                nlohmann::json::array({options, int(mode)}).dump();
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
