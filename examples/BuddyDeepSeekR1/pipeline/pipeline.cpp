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

#include "../../../models/deepseek_r1/pipeline/DistributedGenerate.h"
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/mpi/BlockingMPI.h"

#include <dlfcn.h>

#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
class StageLibrary {
public:
  explicit StageLibrary(const std::filesystem::path &directory) {
    const auto path = directory / "stage.so";
    handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      throw std::runtime_error("cannot load " + path.string() + ": " +
                               dlerror());
  }
  ~StageLibrary() { dlclose(handle); }
  StageLibrary(const StageLibrary &) = delete;
  StageLibrary &operator=(const StageLibrary &) = delete;

  std::unique_ptr<Stage> create(const char *directory) const {
    dlerror();
    auto createStage =
        reinterpret_cast<CreateStage>(dlsym(handle, "createPipelineStage"));
    if (const char *error = dlerror())
      throw std::runtime_error(std::string("createPipelineStage: ") + error);
    if (!createStage)
      throw std::runtime_error("createPipelineStage symbol is null");
    std::unique_ptr<Stage> stage(createStage(directory));
    if (!stage)
      throw std::runtime_error("createPipelineStage returned null");
    return stage;
  }

private:
  void *handle;
};

void generate(int count, const std::string &prompt, const char *vocab,
              const std::string &logitsFile, Stage &stage, int rank,
              int ranks) {
  buddy::runtime::pipeline::checkAllStages(MPI_COMM_WORLD, rank, ranks,
                                           stage.configuration());
  buddy::runtime::pipeline::DistributedGenerateOptions opts;
  opts.prompt = prompt;
  opts.vocabPath = vocab ? vocab : "";
  opts.maxNewTokens = count;
  opts.allowZeroMaxNewTokens = false;
  opts.samplerConfig = buddy::SamplerConfig{};
  opts.logitsDumpPath = logitsFile;
  buddy::runtime::pipeline::generateDistributed(opts, stage, MPI_COMM_WORLD,
                                                rank, ranks);
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
    if (argc != 5 + ranks)
      throw std::runtime_error("usage: pipeline VOCAB MAX_NEW_TOKENS PROMPT "
                               "LOGITS_FILE_OR_DASH STAGE_DIR...");
    {
      StageLibrary library(std::filesystem::absolute(argv[5 + rank]));
      auto stage = library.create(argv[5 + rank]);
      generate(std::stoi(argv[2]), argv[3], argv[1], argv[4], *stage, rank,
               ranks);
    }
  } catch (const std::exception &err) {
    std::cerr << "rank " << rank << ": " << err.what() << '\n';
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  return MPI_Finalize() == MPI_SUCCESS ? 0 : 1;
}
