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

#ifndef BUDDY_DEEPSEEK_PIPELINE_BLOCKING_MPI_H
#define BUDDY_DEEPSEEK_PIPELINE_BLOCKING_MPI_H

#include "Stage.h"
#include <mpi.h>

#include <limits>
#include <stdexcept>
#include <string>

inline void mpiCheck(int status) {
  if (status == MPI_SUCCESS)
    return;
  char message[MPI_MAX_ERROR_STRING];
  int length = 0;
  MPI_Error_string(status, message, &length);
  throw std::runtime_error(std::string(message, length));
}

class BlockingActivation {
public:
  BlockingActivation(const StageConfig &config, MPI_Comm comm)
      : width(config.activationWidth),
        type(config.dtype == ActivationType::F16 ? MPI_UINT16_T : MPI_FLOAT),
        comm(comm) {
    count(1);
  }

  void send(const void *data, int rows, int peer, int tag) const {
    mpiCheck(MPI_Send(data, count(rows), type, peer, tag, comm));
  }

  int receive(void *data, int maxRows, int peer, int tag) const {
    MPI_Status status;
    mpiCheck(MPI_Probe(peer, tag, comm, &status));
    int elements = 0;
    mpiCheck(MPI_Get_count(&status, type, &elements));
    if (elements <= 0 || elements % width != 0 || elements > count(maxRows))
      throw std::runtime_error("activation message has invalid row count");
    mpiCheck(MPI_Recv(data, elements, type, status.MPI_SOURCE, status.MPI_TAG,
                      comm, MPI_STATUS_IGNORE));
    return elements / width;
  }

private:
  int count(int rows) const {
    if (width <= 0 || rows <= 0 ||
        rows > std::numeric_limits<int>::max() / width)
      throw std::runtime_error("activation shape exceeds MPI element count");
    return rows * width;
  }

  int width;
  MPI_Datatype type;
  MPI_Comm comm;
};

#endif
