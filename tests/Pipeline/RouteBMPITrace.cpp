#include "buddy/runtime/pipeline/mpi/ActivationProtocol.h"
#include <cstdlib>
#include <fstream>
#include <mpi.h>
#include <string>

// Test-only PMPI observer. Record real MPI destinations and stop values
// independently of the Stage trace emitted by the generator.
static void record(MPI_Comm comm, const char *event, int peer, int value) {
  const char *prefix = std::getenv("ROUTE_B_MPI_TRACE");
  if (!prefix)
    return;
  int rank;
  PMPI_Comm_rank(comm, &rank);
  std::ofstream out(std::string(prefix) + "." + std::to_string(rank),
                    std::ios::app);
  out << event << ' ' << peer << ' ' << value << '\n';
}
extern "C" int MPI_Send(const void *data, int count, MPI_Datatype type,
                        int dest, int tag, MPI_Comm comm) {
  const char *event = tag == kActivationTag ? "hidden"
                      : tag == kHeaderTag   ? "header"
                      : tag == kTokenTag    ? "token"
                                            : "unexpected";
  record(comm, event, dest, count);
  return PMPI_Send(data, count, type, dest, tag, comm);
}
extern "C" int MPI_Bcast(void *data, int count, MPI_Datatype type, int root,
                         MPI_Comm comm) {
  int rc = PMPI_Bcast(data, count, type, root, comm);
  if (rc == MPI_SUCCESS && count == 1 && type == MPI_INT)
    record(comm, "stop", root, *static_cast<int *>(data));
  return rc;
}
