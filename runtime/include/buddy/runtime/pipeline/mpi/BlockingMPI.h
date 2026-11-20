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

#include "buddy/runtime/pipeline/mpi/ActivationProtocol.h"

#include <vector>

class BlockingActivation : private ActivationProtocol {
  using WireDtype = buddy::runtime::pipeline::WireDtype;
  using WireSpec = buddy::runtime::pipeline::WireSpec;

public:
  BlockingActivation(const StageConfig &config, MPI_Comm comm,
                     WireDtype requested = WireDtype::Native)
      : BlockingActivation(config, comm,
                           resolveWireSpec(config.dtype, requested)) {}

  void send(const ActivationHeader &header, const void *data, int peer) const {
    if (!data)
      throw std::invalid_argument("activation send buffer is null");
    std::string reason;
    if (!validateHeaderInvariants(header, reason))
      throw std::invalid_argument("invalid activation header for send: " +
                                  reason);
    const int elements = checkedElementCount(header.rows);
    checkedWireBytes(spec, elements);
    if (spec.compressed()) {
      try {
        scratch.resize(elements);
        buddy::runtime::pipeline::encodeInto(static_cast<const float *>(data),
                                             elements, scratch.data(),
                                             scratch.size());
      } catch (const std::exception &e) {
        abortProtocol(rank, rank, header, nullptr, e.what());
      }
      data = scratch.data();
    }
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

    const int expectedElements =
        validateReceived(received, expected, capacityElements, peer, source);

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
    void *wireData = data;
    try {
      checkedWireBytes(spec, expectedElements);
      if (spec.compressed()) {
        scratch.resize(expectedElements);
        wireData = scratch.data();
      }
    } catch (const std::exception &e) {
      abortProtocol(peer, source, expected, &received, e.what());
    }
    rc = MPI_Recv(wireData, expectedElements, type, peer, kActivationTag, comm,
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

    if (spec.compressed()) {
      try {
        buddy::runtime::pipeline::decodeInto(scratch.data(), receivedElements,
                                             static_cast<float *>(data),
                                             capacityElements);
      } catch (const std::exception &e) {
        abortProtocol(peer, source, expected, &received, e.what());
      }
    }
    return received;
  }

private:
  BlockingActivation(const StageConfig &config, MPI_Comm comm, WireSpec spec)
      : ActivationProtocol(config, comm, spec.version), spec(spec),
        type(spec.dtype == ActivationType::F16 ? MPI_UINT16_T : MPI_FLOAT) {}

  WireSpec spec;
  mutable std::vector<uint16_t> scratch;
  MPI_Datatype type;
};

#endif
