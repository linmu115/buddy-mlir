//===- BlockingMPIProtocolTest.cpp - BlockingMPI protocol unit tests ------===//
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

#include "buddy/runtime/pipeline/mpi/BlockingMPI.h"

#include <fcntl.h>
#include <mpi.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static_assert(kActivationProtocolVersion == 1,
              "unexpected activation protocol version");
static_assert(kActivationHeaderCount == 4,
              "unexpected activation header count");
static_assert(kActivationTag == 0, "unexpected activation payload tag");
static_assert(kTokenTag == 1, "unexpected token return tag");
static_assert(kHeaderTag == 2, "unexpected activation header tag");

namespace {

constexpr int kContext = 64;
constexpr int kWidth = 1536;
constexpr int kPromptRows = 5;

StageConfig makeTestConfig() {
  StageConfig cfg{};
  cfg.startLayer = 0;
  cfg.endLayer = 14;
  cfg.totalLayers = 28;
  cfg.context = kContext;
  cfg.activationWidth = kWidth;
  cfg.vocabSize = 151936;
  cfg.eosId = 151643;
  cfg.dtype = ActivationType::F16;
  return cfg;
}

uint16_t makePattern(int step, size_t index) {
  // Include edge-case FP16 bit patterns (0x0000, -0.0, 1.0, max finite, NaN).
  static constexpr std::array<uint16_t, 6> special = {0x0000, 0x8000, 0x3c00,
                                                      0x7bff, 0xfbff, 0x7e01};
  if (index < special.size())
    return static_cast<uint16_t>(special[index] ^
                                 static_cast<uint16_t>(step * 0x0101u));
  return static_cast<uint16_t>(((step + 1) * 4099u + index * 131u) & 0xffffu);
}

std::vector<uint16_t> fillPayload(int step, int rows) {
  const size_t total = static_cast<size_t>(rows) * kWidth;
  std::vector<uint16_t> buf(total);
  for (size_t i = 0; i < total; ++i)
    buf[i] = makePattern(step, i);
  return buf;
}

void verifyMakeActivationHeaderErrors() {
  auto expectInvalid = [](int step, int promptRows, int context,
                          const char *label) {
    try {
      (void)makeActivationHeader(step, promptRows, context);
    } catch (const std::invalid_argument &) {
      return;
    }
    throw std::runtime_error(std::string("makeActivationHeader(") + label +
                             ") did not throw");
  };
  expectInvalid(0, 5, 0, "zero-context");
  expectInvalid(-1, 5, 64, "negative-step");
  expectInvalid(1, 64, 64, "decode-pos-oob");
}

void writeCounterFile(const std::string &path, int totalCalls, int prefillCalls,
                      int decodeCalls, int lastStep, int lastPos) {
  if (path.empty())
    return;
  std::ostringstream os;
  os << "compute_calls=" << totalCalls << "\n"
     << "prefill_calls=" << prefillCalls << "\n"
     << "decode_calls=" << decodeCalls << "\n"
     << "last_step=" << lastStep << "\n"
     << "last_pos=" << lastPos << "\n";
  const std::string content = os.str();
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if (fd < 0)
    throw std::runtime_error("failed to open counter file: " + path);
  const char *ptr = content.data();
  size_t remaining = content.size();
  while (remaining > 0) {
    ssize_t written = ::write(fd, ptr, remaining);
    if (written <= 0) {
      ::close(fd);
      throw std::runtime_error("failed to write counter file: " + path);
    }
    ptr += written;
    remaining -= static_cast<size_t>(written);
  }
  ::fsync(fd);
  ::close(fd);
}

void sendRawFrame(int peer, const std::array<int32_t, 4> &wireHeader,
                  const uint16_t *payload, int payloadElements) {
  mpiCheck(MPI_Send(wireHeader.data(), 4, MPI_INT32_T, peer, kHeaderTag,
                    MPI_COMM_WORLD));
  mpiCheck(MPI_Send(payload, payloadElements, MPI_UINT16_T, peer,
                    kActivationTag, MPI_COMM_WORLD));
}

int runNormal2Rank(int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error("normal mode requires exactly 2 ranks");
  if (rank == 0)
    verifyMakeActivationHeaderErrors();

  const StageConfig cfg = makeTestConfig();
  BlockingActivation activation(cfg, MPI_COMM_WORLD);
  std::vector<uint16_t> recvBuf(static_cast<size_t>(kContext) * kWidth, 0);

  int computeCalls = 0;
  int prefillCalls = 0;
  int decodeCalls = 0;
  std::vector<int> decodePositions;

  for (int step = 0; step < 3; ++step) {
    const ActivationHeader expected =
        makeActivationHeader(step, kPromptRows, kContext);
    if (rank == 0) {
      auto payload = fillPayload(step, expected.rows);
      activation.send(expected, payload.data(), 1);
      int token = -1;
      mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
      if (token != 1000 + step)
        throw std::runtime_error("rank 0 received unexpected token");
    } else {
      const ActivationHeader got =
          activation.receive(recvBuf.data(), recvBuf.size(), 0, expected);
      if (got != expected)
        throw std::runtime_error("received header does not match expected");
      const size_t n = static_cast<size_t>(got.rows) * kWidth;
      for (size_t i = 0; i < n; ++i) {
        const uint16_t want = makePattern(step, i);
        if (recvBuf[i] != want) {
          std::ostringstream os;
          os << "bit pattern mismatch at step=" << step << " idx=" << i
             << " got=" << recvBuf[i] << " want=" << want;
          throw std::runtime_error(os.str());
        }
      }
      ++computeCalls;
      if (got.step == 0) {
        ++prefillCalls;
      } else {
        ++decodeCalls;
        decodePositions.push_back(got.pos);
      }
      std::cout << "rank 1 verified step=" << got.step
                << " header=" << formatActivationHeader(got)
                << " elements=" << n << "/" << n << " bit-exact" << std::endl;
      const int token = 1000 + step;
      mpiCheck(MPI_Send(&token, 1, MPI_INT, 0, kTokenTag, MPI_COMM_WORLD));
    }
  }

  if (rank == 1) {
    if (computeCalls != 3 || prefillCalls != 1 || decodeCalls != 2 ||
        decodePositions.size() != 2 || decodePositions[0] != 5 ||
        decodePositions[1] != 6)
      throw std::runtime_error("unexpected compute call summary");
    std::cout << "PASS normal-2rank: prefill=1 decode=2 decode_pos=[5,6]"
              << std::endl;
  }
  return 0;
}

int runForward3Rank(int rank, int ranks) {
  if (ranks != 3)
    throw std::runtime_error("forward3 mode requires exactly 3 ranks");
  const StageConfig cfg = makeTestConfig();
  BlockingActivation activation(cfg, MPI_COMM_WORLD);
  std::vector<uint16_t> recvBuf(static_cast<size_t>(kContext) * kWidth, 0);

  for (int step = 0; step < 3; ++step) {
    const ActivationHeader expected =
        makeActivationHeader(step, kPromptRows, kContext);
    if (rank == 0) {
      auto payload = fillPayload(step, expected.rows);
      activation.send(expected, payload.data(), 1);
      int token = -1;
      mpiCheck(MPI_Recv(&token, 1, MPI_INT, 2, kTokenTag, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
      if (token != 2000 + step)
        throw std::runtime_error("rank 0 received unexpected token");
    } else if (rank == 1) {
      const ActivationHeader forwarded =
          activation.receive(recvBuf.data(), recvBuf.size(), 0, expected);
      // Middle rank forwards the validated header verbatim.
      activation.send(forwarded, recvBuf.data(), 2);
    } else {
      const ActivationHeader got =
          activation.receive(recvBuf.data(), recvBuf.size(), 1, expected);
      if (got != expected)
        throw std::runtime_error("rank 2 header mismatch");
      const size_t n = static_cast<size_t>(got.rows) * kWidth;
      for (size_t i = 0; i < n; ++i) {
        if (recvBuf[i] != makePattern(step, i))
          throw std::runtime_error("rank 2 bit pattern mismatch");
      }
      const int token = 2000 + step;
      mpiCheck(MPI_Send(&token, 1, MPI_INT, 0, kTokenTag, MPI_COMM_WORLD));
    }
  }
  if (rank == 2) {
    std::cout << "PASS forward-3rank: middle stage forwarded validated headers "
                 "verbatim across 3 steps"
              << std::endl;
  }
  return 0;
}

int runBadInjection(const std::string &mode, const std::string &counterFile,
                    int rank, int ranks) {
  if (ranks != 2)
    throw std::runtime_error(mode + " mode requires exactly 2 ranks");
  const StageConfig cfg = makeTestConfig();
  BlockingActivation activation(cfg, MPI_COMM_WORLD);
  std::vector<uint16_t> recvBuf(static_cast<size_t>(kContext) * kWidth, 0);

  int totalCalls = 0;
  int prefillCalls = 0;
  int decodeCalls = 0;
  if (rank == 1)
    writeCounterFile(counterFile, 0, 0, 0, -1, -1);

  int badStep = 1;
  if (mode == "bad-duplicate-step")
    badStep = 2;
  else if (mode == "bad-short-prefill")
    badStep = 0;

  for (int step = 0; step <= badStep; ++step) {
    const ActivationHeader expected =
        makeActivationHeader(step, kPromptRows, kContext);
    if (rank == 0) {
      if (step < badStep) {
        auto payload = fillPayload(step, expected.rows);
        activation.send(expected, payload.data(), 1);
        int token = -1;
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      } else if (mode == "bad-duplicate-step") {
        auto payload = fillPayload(1, 1);
        const std::array<int32_t, 4> dupHeader = {kActivationProtocolVersion, 1,
                                                  kPromptRows, 1};
        sendRawFrame(1, dupHeader, payload.data(), kWidth);
        int token = -1;
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      } else if (mode == "bad-wrong-pos") {
        auto payload = fillPayload(1, 1);
        const std::array<int32_t, 4> badPosHeader = {kActivationProtocolVersion,
                                                     1, kPromptRows + 2, 1};
        sendRawFrame(1, badPosHeader, payload.data(), kWidth);
        int token = -1;
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      } else if (mode == "bad-oob-pos") {
        auto payload = fillPayload(1, 1);
        const std::array<int32_t, 4> oobHeader = {kActivationProtocolVersion, 1,
                                                  kContext, 1};
        sendRawFrame(1, oobHeader, payload.data(), kWidth);
        int token = -1;
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      } else if (mode == "bad-short-payload") {
        auto payload = fillPayload(1, 1);
        const std::array<int32_t, 4> validHeader = {kActivationProtocolVersion,
                                                    1, kPromptRows, 1};
        sendRawFrame(1, validHeader, payload.data(), kWidth - 1);
        int token = -1;
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      } else if (mode == "bad-short-prefill") {
        auto payload = fillPayload(0, kContext);
        const std::array<int32_t, 4> validHeader = {kActivationProtocolVersion,
                                                    0, -1, kContext};
        sendRawFrame(1, validHeader, payload.data(), kContext * kWidth - 1);
        int token = -1;
        mpiCheck(MPI_Recv(&token, 1, MPI_INT, 1, kTokenTag, MPI_COMM_WORLD,
                          MPI_STATUS_IGNORE));
      } else {
        throw std::runtime_error("unknown bad injection mode: " + mode);
      }
    } else {
      const ActivationHeader got =
          activation.receive(recvBuf.data(), recvBuf.size(), 0, expected);
      // Simulated compute entry point: must NOT be reached on badStep.
      ++totalCalls;
      if (got.step == 0)
        ++prefillCalls;
      else
        ++decodeCalls;
      writeCounterFile(counterFile, totalCalls, prefillCalls, decodeCalls,
                       got.step, got.pos);
      std::cerr << "rank 1 compute_entry step=" << got.step
                << " pos=" << got.pos << " compute_calls=" << totalCalls
                << std::endl;
      const int token = 1000 + step;
      mpiCheck(MPI_Send(&token, 1, MPI_INT, 0, kTokenTag, MPI_COMM_WORLD));
    }
  }
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    return 1;
  int rank = 0;
  try {
    int ranks = 0;
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (argc < 2)
      throw std::runtime_error("usage: buddy-pipeline-protocol-test MODE "
                               "[COUNTER_FILE]");
    const std::string mode = argv[1];
    const std::string counterFile = argc >= 3 ? argv[2] : "";
    int rc = 0;
    if (mode == "normal")
      rc = runNormal2Rank(rank, ranks);
    else if (mode == "forward3")
      rc = runForward3Rank(rank, ranks);
    else
      rc = runBadInjection(mode, counterFile, rank, ranks);
    if (rc != 0) {
      MPI_Abort(MPI_COMM_WORLD, rc);
      return rc;
    }
  } catch (const std::exception &err) {
    std::cerr << "rank " << rank << ": " << err.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
}
