//===----------------------------------------------------------------------===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#ifndef BUDDY_RUNTIME_PIPELINE_MPI_ASYNC_MPI_H
#define BUDDY_RUNTIME_PIPELINE_MPI_ASYNC_MPI_H

#include "buddy/runtime/pipeline/mpi/ActivationProtocol.h"

#include <cstdio>
#include <vector>

class AsyncActivation : private ActivationProtocol {
  using WireDtype = buddy::runtime::pipeline::WireDtype;
  using WireSpec = buddy::runtime::pipeline::WireSpec;

  struct Slot {
    ActivationHeader frame;
    std::array<int32_t, 4> header{};
    std::vector<float> floats;
    std::vector<uint16_t> halves;
    MPI_Request head = MPI_REQUEST_NULL, body = MPI_REQUEST_NULL;
    bool occupied = false;
    void *data() { return floats.empty() ? static_cast<void *>(halves.data())
                                         : static_cast<void *>(floats.data()); }
  };

public:
  // All ranks enter together. Only allocate the directions this rank uses.
  AsyncActivation(const StageConfig &cfg, MPI_Comm control, WireDtype wire,
                  bool hasRx, bool hasTx, int window, bool synchronous = false)
      : ActivationProtocol(cfg, control, resolveWireSpec(cfg.dtype, wire).version),
        spec(resolveWireSpec(cfg.dtype, wire)),
        type(spec.dtype == ActivationType::F32 ? MPI_FLOAT : MPI_UINT16_T),
        capacity(checkedElementCount(prefillChunk ? prefillChunk : context)),
        window(window), synchronous(synchronous) {
    if (window != 1 && window != 2)
      throw std::invalid_argument("invalid activation window");
    checkedWireBytes(spec, capacity);
    for (int i = 0; i < window; ++i) {
      if (hasRx)
        allocate(rx[i]);
      if (hasTx)
        allocate(tx[i]);
    }
    if (hasRx && spec.compressed())
      decoded.resize(capacity);
    mpiCheck(MPI_Comm_dup(control, &comm));
    mpiCheck(MPI_Comm_set_errhandler(comm, MPI_ERRORS_RETURN));
  }
  AsyncActivation(const AsyncActivation &) = delete;
  AsyncActivation &operator=(const AsyncActivation &) = delete;

  ~AsyncActivation() {
    // Never unwind through buffers still owned by MPI, or wait on an
    // unmatched receive during exception handling. Normal callers drain.
    for (int i = 0; i < window; ++i)
      if (rx[i].occupied || tx[i].occupied)
        abortProtocol(rank, rank, {}, nullptr, "undrained activation session");
    MPI_Comm_free(&comm);
  }

  MPI_Comm communicator() const { return comm; }

  void postReceive(const ActivationHeader &frame, int peer) {
    const int n = checkFrame(frame, peer);
    Slot &s = rx[frame.step % window];
    if (s.occupied || !s.data())
      abortProtocol(peer, peer, frame, nullptr, "unavailable receive slot");
    s.occupied = true;
    s.frame = frame;
    check(MPI_Irecv(s.header.data(), 4, MPI_INT32_T, peer, kHeaderTag, comm,
                    &s.head), frame, peer);
    check(MPI_Irecv(s.data(), n, type, peer, kActivationTag, comm, &s.body),
          frame, peer);
  }

  // Native data lives until the same RX slot is posted again. The one shared
  // f32 decode buffer lives until the next finishReceive. Stage calls serialize.
  void *finishReceive(const ActivationHeader &frame, int peer) {
    Slot &s = rx.at(static_cast<size_t>(frame.step) % window);
    if (!s.occupied || s.frame != frame)
      abortProtocol(peer, peer, frame, nullptr, "unexpected receive slot");
    waitReceive(s.head, MPI_INT32_T, kHeaderTag, 4, peer, frame, nullptr);
    const ActivationHeader got{s.header[0], s.header[1], s.header[2], s.header[3]};
    const int n = validateReceived(got, frame, capacity, peer, peer);
    waitReceive(s.body, type, kActivationTag, n, peer, frame, &got);
    s.occupied = false;
    if (spec.compressed()) {
      buddy::runtime::pipeline::decodeInto(s.halves.data(), n, decoded.data(),
                                           decoded.size());
      return decoded.data();
    }
    return s.data();
  }

  void postSend(const ActivationHeader &frame, const void *data, int peer) {
    const int n = checkFrame(frame, peer);
    Slot &s = tx[frame.step % window];
    waitSend(s);
    if (!data || !s.data())
      abortProtocol(peer, peer, frame, nullptr, "missing activation send buffer");
    if (spec.compressed()) {
      try {
        buddy::runtime::pipeline::encodeInto(static_cast<const float *>(data),
                                             n, s.halves.data(), capacity);
      } catch (const std::exception &e) {
        abortProtocol(rank, rank, frame, nullptr, e.what());
      }
    } else {
      std::memcpy(s.data(), data, checkedWireBytes(spec, n));
    }
    s.frame = frame;
    s.header = {frame.version, frame.step, frame.pos, frame.rows};
    s.occupied = true;
    auto send = synchronous ? MPI_Issend : MPI_Isend;
    check(send(s.header.data(), 4, MPI_INT32_T, peer, kHeaderTag, comm, &s.head),
          frame, peer);
    check(send(s.data(), n, type, peer, kActivationTag, comm, &s.body), frame, peer);
  }

  void drain() {
    for (int i = 0; i < window; ++i) {
      if (rx[i].occupied)
        abortProtocol(rank, rank, rx[i].frame, nullptr, "unconsumed receive");
      waitSend(tx[i]);
    }
  }

private:
  int checkFrame(const ActivationHeader &h, int peer) const {
    std::string reason;
    if (!validateHeaderInvariants(h, reason))
      abortProtocol(peer, peer, h, nullptr, "invalid local header: " + reason);
    const int n = checkedElementCount(h.rows);
    if (n > capacity)
      abortProtocol(peer, peer, h, nullptr, "frame exceeds slot capacity");
    return n;
  }

  void allocate(Slot &s) {
    if (spec.dtype == ActivationType::F32)
      s.floats.resize(capacity);
    else
      s.halves.resize(capacity);
  }

  void check(int rc, const ActivationHeader &h, int peer) const {
    if (rc != MPI_SUCCESS)
      abortProtocol(peer, peer, h, nullptr, mpiErrorString(rc));
  }

  void waitSend(Slot &s) {
    if (!s.occupied)
      return;
    check(MPI_Wait(&s.head, MPI_STATUS_IGNORE), s.frame, rank + 1);
    check(MPI_Wait(&s.body, MPI_STATUS_IGNORE), s.frame, rank + 1);
    s.occupied = false;
  }

  void waitReceive(MPI_Request &request, MPI_Datatype dtype, int tag, int count,
                   int peer, const ActivationHeader &frame,
                   const ActivationHeader *got) {
    MPI_Status status;
    const int rc = MPI_Wait(&request, &status);
    const char *part = got ? "payload" : "header";
    const auto mismatch = [&](const std::string &actual) {
      abortProtocol(peer, peer, frame, got,
                    std::string(part) + " element count mismatch (got " + actual +
                    ", expected " + std::to_string(count) + ")");
    };
    if (rc != MPI_SUCCESS) {
      int cls;
      if (MPI_Error_class(rc, &cls) == MPI_SUCCESS && cls == MPI_ERR_TRUNCATE)
        mismatch("truncated");
      abortProtocol(peer, peer, frame, got, mpiErrorString(rc));
    }
    if (status.MPI_SOURCE != peer || status.MPI_TAG != tag)
      abortProtocol(peer, status.MPI_SOURCE, frame, got,
                    std::string(part) + " receive source or tag mismatch");
    int actual;
    check(MPI_Get_count(&status, dtype, &actual), frame, peer);
    if (actual != count)
      mismatch(actual == MPI_UNDEFINED ? "MPI_UNDEFINED" : std::to_string(actual));
  }

  WireSpec spec;
  MPI_Datatype type;
  int capacity, window;
  bool synchronous;
  std::array<Slot, 2> rx, tx;
  std::vector<float> decoded;
};

#endif
