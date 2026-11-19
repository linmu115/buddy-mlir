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

#ifndef BUDDY_RUNTIME_PIPELINE_MPI_BLOCKING_MPI_H
#define BUDDY_RUNTIME_PIPELINE_MPI_BLOCKING_MPI_H

#include "buddy/runtime/pipeline/Stage.h"
#include <mpi.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

inline constexpr int32_t kActivationProtocolVersion = 1;
inline constexpr int kActivationHeaderCount = 4;
inline constexpr int kActivationTag = 0;
inline constexpr int kTokenTag = 1;
inline constexpr int kHeaderTag = 2;

inline std::string mpiErrorString(int status) {
  char message[MPI_MAX_ERROR_STRING];
  int length = 0;
  MPI_Error_string(status, message, &length);
  return std::string(message, length);
}

inline void mpiCheck(int status) {
  if (status == MPI_SUCCESS)
    return;
  throw std::runtime_error(mpiErrorString(status));
}

struct ActivationHeader {
  int32_t version = kActivationProtocolVersion;
  int32_t step = 0;
  int32_t pos = -1;
  int32_t rows = 0;

  bool operator==(const ActivationHeader &other) const {
    return version == other.version && step == other.step && pos == other.pos &&
           rows == other.rows;
  }

  bool operator!=(const ActivationHeader &other) const {
    return !(*this == other);
  }
};

inline std::string formatActivationHeader(const ActivationHeader &header) {
  std::ostringstream os;
  os << "{version=" << header.version << ", step=" << header.step
     << ", pos=" << header.pos << ", rows=" << header.rows << "}";
  return os.str();
}

inline ActivationHeader makeActivationHeader(int step, int promptRows,
                                             int context) {
  if (context <= 0)
    throw std::invalid_argument("activation context must be positive");
  if (step < 0)
    throw std::invalid_argument("activation step must be non-negative");
  if (step == 0)
    return ActivationHeader{kActivationProtocolVersion, 0, -1, context};
  const int64_t pos64 = static_cast<int64_t>(promptRows) + step - 1;
  if (pos64 < 0 || pos64 >= static_cast<int64_t>(context))
    throw std::invalid_argument("decode pos out of context range");
  return ActivationHeader{kActivationProtocolVersion,
                          static_cast<int32_t>(step),
                          static_cast<int32_t>(pos64), 1};
}

inline ActivationHeader makeChunkPrefillHeader(int step, int pos,
                                               int prefillChunk, int context) {
  if (context <= 0)
    throw std::invalid_argument("activation context must be positive");
  if (prefillChunk <= 0 || prefillChunk > context)
    throw std::invalid_argument(
        "activation prefillChunk must satisfy 0 < prefillChunk <= context");
  if (step < 0)
    throw std::invalid_argument("activation step must be non-negative");
  if (pos < 0 || pos >= context)
    throw std::invalid_argument("prefill chunk pos out of context range");
  return ActivationHeader{kActivationProtocolVersion,
                          static_cast<int32_t>(step),
                          static_cast<int32_t>(pos),
                          static_cast<int32_t>(prefillChunk)};
}

inline ActivationHeader makeChunkDecodeHeader(int step, int pos, int context) {
  if (context <= 0)
    throw std::invalid_argument("activation context must be positive");
  if (step < 0)
    throw std::invalid_argument("activation step must be non-negative");
  if (pos < 0 || pos >= context)
    throw std::invalid_argument("decode pos out of context range");
  return ActivationHeader{kActivationProtocolVersion,
                          static_cast<int32_t>(step),
                          static_cast<int32_t>(pos), 1};
}

class BlockingActivation {
public:
  BlockingActivation(const StageConfig &config, MPI_Comm comm)
      : width(config.activationWidth), context(config.context),
        prefillChunk(config.prefillChunk),
        type(config.dtype == ActivationType::F16 ? MPI_UINT16_T : MPI_FLOAT),
        comm(comm) {
    if (width <= 0 || context <= 0)
      throw std::invalid_argument("invalid activation width or context");
    if (prefillChunk < 0 || prefillChunk > context)
      throw std::invalid_argument("invalid activation prefillChunk");
    if (config.dtype != ActivationType::F16 &&
        config.dtype != ActivationType::F32)
      throw std::invalid_argument("unsupported activation dtype");
    mpiCheck(MPI_Comm_rank(comm, &rank));
    checkedElementCount(1);
  }

  void send(const ActivationHeader &header, const void *data, int peer) const {
    if (!data)
      throw std::invalid_argument("activation send buffer is null");
    std::string reason;
    if (!validateHeaderInvariants(header, reason))
      throw std::invalid_argument("invalid activation header for send: " +
                                  reason);
    const int elements = checkedElementCount(header.rows);
    const std::array<int32_t, kActivationHeaderCount> wire = {
        header.version, header.step, header.pos, header.rows};
    mpiCheck(MPI_Send(wire.data(), kActivationHeaderCount, MPI_INT32_T, peer,
                      kHeaderTag, comm));
    mpiCheck(MPI_Send(data, elements, type, peer, kActivationTag, comm));
  }

  ActivationHeader receive(void *data, size_t capacityElements, int peer,
                           const ActivationHeader &expected) const {
    if (!data)
      abortProtocol(peer, peer, expected, nullptr,
                    "activation receive buffer is null");
    std::string reason;
    if (!validateHeaderInvariants(expected, reason))
      abortProtocol(peer, peer, expected, nullptr,
                    "invalid expected header: " + reason);

    MPI_Status headerProbeStatus;
    int rc = MPI_Probe(peer, kHeaderTag, comm, &headerProbeStatus);
    if (rc != MPI_SUCCESS)
      abortProtocol(peer, peer, expected, nullptr,
                    "MPI_Probe(header) failed: " + mpiErrorString(rc));
    if (headerProbeStatus.MPI_SOURCE != peer ||
        headerProbeStatus.MPI_TAG != kHeaderTag)
      abortProtocol(peer, headerProbeStatus.MPI_SOURCE, expected, nullptr,
                    "unexpected header source or tag (tag=" +
                        std::to_string(headerProbeStatus.MPI_TAG) + ")");

    int headerCount = 0;
    rc = MPI_Get_count(&headerProbeStatus, MPI_INT32_T, &headerCount);
    if (rc != MPI_SUCCESS || headerCount == MPI_UNDEFINED ||
        headerCount != kActivationHeaderCount)
      abortProtocol(peer, headerProbeStatus.MPI_SOURCE, expected, nullptr,
                    "header element count mismatch (got " +
                        (headerCount == MPI_UNDEFINED
                             ? std::string("MPI_UNDEFINED")
                             : std::to_string(headerCount)) +
                        ", expected " + std::to_string(kActivationHeaderCount) +
                        ")");

    std::array<int32_t, kActivationHeaderCount> wire{};
    MPI_Status headerRecvStatus;
    rc = MPI_Recv(wire.data(), kActivationHeaderCount, MPI_INT32_T, peer,
                  kHeaderTag, comm, &headerRecvStatus);
    if (rc != MPI_SUCCESS)
      abortProtocol(peer, headerProbeStatus.MPI_SOURCE, expected, nullptr,
                    "MPI_Recv(header) failed: " + mpiErrorString(rc));
    if (headerRecvStatus.MPI_SOURCE != peer ||
        headerRecvStatus.MPI_TAG != kHeaderTag)
      abortProtocol(peer, headerRecvStatus.MPI_SOURCE, expected, nullptr,
                    "header receive source or tag mismatch (tag=" +
                        std::to_string(headerRecvStatus.MPI_TAG) + ")");

    int receivedHeaderCount = 0;
    rc = MPI_Get_count(&headerRecvStatus, MPI_INT32_T, &receivedHeaderCount);
    if (rc != MPI_SUCCESS || receivedHeaderCount == MPI_UNDEFINED ||
        receivedHeaderCount != kActivationHeaderCount)
      abortProtocol(peer, headerRecvStatus.MPI_SOURCE, expected, nullptr,
                    "header receive element count mismatch");

    const ActivationHeader received{wire[0], wire[1], wire[2], wire[3]};
    const int source = headerRecvStatus.MPI_SOURCE;

    if (received.version != kActivationProtocolVersion)
      abortProtocol(peer, source, expected, &received,
                    "header version mismatch");
    if (received.step != expected.step)
      abortProtocol(peer, source, expected, &received, "header step mismatch");
    if (!validateHeaderInvariants(received, reason))
      abortProtocol(peer, source, expected, &received,
                    "header phase/pos/rows invalid: " + reason);
    if (received.pos != expected.pos)
      abortProtocol(peer, source, expected, &received, "header pos mismatch");
    if (received.rows != expected.rows)
      abortProtocol(peer, source, expected, &received, "header rows mismatch");

    int expectedElements = 0;
    if (!computeElementCount(received.rows, expectedElements))
      abortProtocol(peer, source, expected, &received,
                    "expected payload element count overflows MPI count");
    if (static_cast<size_t>(expectedElements) > capacityElements)
      abortProtocol(peer, source, expected, &received,
                    "expected payload element count (" +
                        std::to_string(expectedElements) +
                        ") exceeds buffer capacity (" +
                        std::to_string(capacityElements) + ")");

    MPI_Status payloadProbeStatus;
    rc = MPI_Probe(peer, kActivationTag, comm, &payloadProbeStatus);
    if (rc != MPI_SUCCESS)
      abortProtocol(peer, source, expected, &received,
                    "MPI_Probe(payload) failed: " + mpiErrorString(rc));
    if (payloadProbeStatus.MPI_SOURCE != peer ||
        payloadProbeStatus.MPI_TAG != kActivationTag)
      abortProtocol(peer, payloadProbeStatus.MPI_SOURCE, expected, &received,
                    "unexpected payload source or tag (tag=" +
                        std::to_string(payloadProbeStatus.MPI_TAG) + ")");

    int actualElements = 0;
    rc = MPI_Get_count(&payloadProbeStatus, type, &actualElements);
    if (rc != MPI_SUCCESS || actualElements == MPI_UNDEFINED ||
        static_cast<int64_t>(actualElements) !=
            static_cast<int64_t>(expectedElements))
      abortProtocol(peer, payloadProbeStatus.MPI_SOURCE, expected, &received,
                    "payload element count mismatch (got " +
                        (actualElements == MPI_UNDEFINED
                             ? std::string("MPI_UNDEFINED")
                             : std::to_string(actualElements)) +
                        ", expected " + std::to_string(expectedElements) + ")");

    MPI_Status payloadRecvStatus;
    rc = MPI_Recv(data, expectedElements, type, peer, kActivationTag, comm,
                  &payloadRecvStatus);
    if (rc != MPI_SUCCESS)
      abortProtocol(peer, payloadProbeStatus.MPI_SOURCE, expected, &received,
                    "MPI_Recv(payload) failed: " + mpiErrorString(rc));
    if (payloadRecvStatus.MPI_SOURCE != peer ||
        payloadRecvStatus.MPI_TAG != kActivationTag)
      abortProtocol(peer, payloadRecvStatus.MPI_SOURCE, expected, &received,
                    "payload receive source or tag mismatch (tag=" +
                        std::to_string(payloadRecvStatus.MPI_TAG) + ")");

    int receivedElements = 0;
    rc = MPI_Get_count(&payloadRecvStatus, type, &receivedElements);
    if (rc != MPI_SUCCESS || receivedElements == MPI_UNDEFINED ||
        receivedElements != expectedElements)
      abortProtocol(peer, payloadRecvStatus.MPI_SOURCE, expected, &received,
                    "payload receive element count mismatch");

    return received;
  }

private:
  bool validateHeaderInvariants(const ActivationHeader &header,
                                std::string &reason) const {
    if (header.version != kActivationProtocolVersion) {
      reason = "version must be " + std::to_string(kActivationProtocolVersion);
      return false;
    }
    if (header.step < 0) {
      reason = "step must be non-negative";
      return false;
    }
    if (prefillChunk > 0) {
      if (header.pos < 0 || header.pos >= context) {
        reason = "chunked pos must satisfy 0 <= pos < " +
                 std::to_string(context);
        return false;
      }
      if (header.rows != prefillChunk && header.rows != 1) {
        reason = "chunked rows must be 1 or prefillChunk (" +
                 std::to_string(prefillChunk) + ")";
        return false;
      }
      return true;
    }
    if (header.step == 0) {
      if (header.pos != -1) {
        reason = "prefill pos must be -1";
        return false;
      }
      if (header.rows != context) {
        reason = "prefill rows must equal context (" + std::to_string(context) +
                 ")";
        return false;
      }
      return true;
    }
    if (header.pos < 0 || header.pos >= context) {
      reason = "decode pos must satisfy 0 <= pos < " + std::to_string(context);
      return false;
    }
    if (header.rows != 1) {
      reason = "decode rows must be 1";
      return false;
    }
    return true;
  }

  bool computeElementCount(int32_t rows, int &outElements) const {
    if (width <= 0 || rows <= 0)
      return false;
    const int64_t elements =
        static_cast<int64_t>(rows) * static_cast<int64_t>(width);
    if (elements <= 0 || elements > std::numeric_limits<int>::max())
      return false;
    outElements = static_cast<int>(elements);
    return true;
  }

  int checkedElementCount(int32_t rows) const {
    int elements = 0;
    if (!computeElementCount(rows, elements))
      throw std::runtime_error("activation shape exceeds MPI element count");
    return elements;
  }

  [[noreturn]] void abortProtocol(int expectedPeer, int actualSource,
                                  const ActivationHeader &expected,
                                  const ActivationHeader *received,
                                  const std::string &detail) const {
    std::ostringstream os;
    os << "rank " << rank << " protocol error (source=" << actualSource
       << ", expected_source=" << expectedPeer << "): " << detail
       << "; expected=" << formatActivationHeader(expected) << "; received=";
    if (received)
      os << formatActivationHeader(*received);
    else
      os << "<unavailable>";
    const std::string message = os.str() + '\n';
    std::cerr.write(message.data(),
                    static_cast<std::streamsize>(message.size()));
    std::cerr.flush();
    // Give the MPI launcher a moment to drain the redirected stderr pipe;
    // otherwise MPI_Abort can race the diagnostic and drop it.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    MPI_Abort(comm, 1);
    std::_Exit(1);
  }

  int width;
  int context;
  int prefillChunk;
  MPI_Datatype type;
  MPI_Comm comm;
  int rank = 0;
};

#endif
