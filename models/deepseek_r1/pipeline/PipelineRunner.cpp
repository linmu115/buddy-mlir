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

#include "PipelineRunner.h"
#include "PipelineGenerate.h"

#include "buddy/runtime/core/ModelManifest.h"
#include "buddy/runtime/pipeline/Stage.h"
#include "buddy/runtime/pipeline/StageManifest.h"
#include "buddy/runtime/pipeline/mpi/BlockingMPI.h"

#include <dlfcn.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace buddy::runtime {
namespace {

void requireActiveMPI() {
  int initialized = 0;
  int finalized = 0;
  mpiCheck(MPI_Initialized(&initialized));
  if (!initialized)
    throw std::runtime_error(
        "PipelineRunner requires an active MPI session (MPI_Init not called)");
  mpiCheck(MPI_Finalized(&finalized));
  if (finalized)
    throw std::runtime_error("PipelineRunner requires an active MPI session "
                             "(MPI_Finalize already called)");
}

void requireExternalWeights(const ModelManifest &m, const StageManifest &spec) {
  if (m.soPath.empty())
    throw std::runtime_error("missing HostSharedLib code object (soPath)");
  if (spec.activationType == "f16") {
    if (m.weightPaths.size() != 1)
      throw std::runtime_error(
          "field constants (external weights) must contain exactly 1 weight "
          "for f16 stage (got " +
          std::to_string(m.weightPaths.size()) + ")");
  } else {
    if (m.weightPaths.empty() || m.weightPaths.size() > 2)
      throw std::runtime_error(
          "field constants (external weights) must contain 1 or 2 weights "
          "for f32 stage (got " +
          std::to_string(m.weightPaths.size()) + ")");
  }
  for (size_t i = 0; i < m.weightPaths.size(); ++i) {
    if (m.weightPaths[i].empty())
      throw std::runtime_error("field constants[" + std::to_string(i) +
                               "] (external weight path) must be non-empty");
  }
}

struct StageDeleter {
  DestroyStageV1 destroy = nullptr;
  void operator()(Stage *stage) const {
    if (stage && destroy)
      destroy(stage);
  }
};

using StagePtr = std::unique_ptr<Stage, StageDeleter>;

class StageLibrary {
public:
  explicit StageLibrary(const std::string &soPath) {
    if (soPath.empty())
      throw std::runtime_error("field soPath must be non-empty");
    dlerror();
    handle = dlopen(soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      const char *err = dlerror();
      throw std::runtime_error("cannot load stage library " + soPath + ": " +
                               (err ? err : "unknown error"));
    }
  }

  ~StageLibrary() {
    if (handle)
      dlclose(handle);
  }

  StageLibrary(const StageLibrary &) = delete;
  StageLibrary &operator=(const StageLibrary &) = delete;

  StagePtr create(const std::vector<std::string> &weightPaths) const {
    dlerror();
    auto *destroySym = dlsym(handle, "buddy_destroy_pipeline_stage_v1");
    if (const char *err = dlerror())
      throw std::runtime_error(
          std::string("missing buddy_destroy_pipeline_stage_v1: ") + err);
    if (!destroySym)
      throw std::runtime_error(
          "buddy_destroy_pipeline_stage_v1 symbol is null");
    auto destroyFn = reinterpret_cast<DestroyStageV1>(destroySym);

    if (weightPaths.size() > 1) {
      dlerror();
      auto *createV2Sym = dlsym(handle, "buddy_create_pipeline_stage_v2");
      if (!dlerror() && createV2Sym) {
        std::vector<const char *> ptrs;
        ptrs.reserve(weightPaths.size());
        for (const auto &p : weightPaths)
          ptrs.push_back(p.c_str());
        auto createV2Fn = reinterpret_cast<CreateStageV2>(createV2Sym);
        Stage *raw = createV2Fn(ptrs.data(), ptrs.size());
        if (!raw)
          throw std::runtime_error(
              "buddy_create_pipeline_stage_v2 returned null");
        return StagePtr(raw, StageDeleter{destroyFn});
      }
    }

    dlerror();
    auto *createSym = dlsym(handle, "buddy_create_pipeline_stage_v1");
    if (const char *err = dlerror())
      throw std::runtime_error(
          std::string("missing buddy_create_pipeline_stage_v1: ") + err);
    if (!createSym)
      throw std::runtime_error("buddy_create_pipeline_stage_v1 symbol is null");

    auto createFn = reinterpret_cast<CreateStageV1>(createSym);
    Stage *raw = createFn(weightPaths.at(0).c_str());
    if (!raw)
      throw std::runtime_error("buddy_create_pipeline_stage_v1 returned null");
    return StagePtr(raw, StageDeleter{destroyFn});
  }

private:
  void *handle = nullptr;
};

void validateRunConfig(const RunConfig &cfg) {
  if (cfg.raxPath.empty())
    throw std::runtime_error("field raxPath must be non-empty");
  if (cfg.batchSize != 0 && cfg.batchSize != 1)
    throw std::runtime_error("field batchSize must be 0 or 1 in pipeline mode "
                             "(got " +
                             std::to_string(cfg.batchSize) + ")");
  if (cfg.interactive)
    throw std::runtime_error("field interactive is not supported in pipeline "
                             "mode");
  if (cfg.promptLength != 0)
    throw std::runtime_error("field promptLength is not supported in pipeline "
                             "mode");
  if (!cfg.prompts.empty())
    throw std::runtime_error("field prompts (batch prompt-file) is not "
                             "supported in pipeline mode");
  if (!cfg.perplexityIdsPath.empty())
    throw std::runtime_error("field perplexityIdsPath is not supported in "
                             "pipeline mode");
  if (!cfg.chatTemplatePath.empty())
    throw std::runtime_error("field chatTemplatePath is not supported in "
                             "pipeline mode");
  if (cfg.maxNewTokens < 0)
    throw std::runtime_error("field maxNewTokens must be non-negative (got " +
                             std::to_string(cfg.maxNewTokens) + ")");
}

[[noreturn]] void abortPipelineJob(MPI_Comm comm, int rank,
                                   const std::string &raxPath,
                                   const std::string &modelName,
                                   const std::string &detail) {
  std::ostringstream os;
  os << "rank " << rank
     << " [package=" << (raxPath.empty() ? "<unset>" : raxPath);
  if (!modelName.empty())
    os << ", model=" << modelName;
  os << "]: " << detail << '\n';
  const std::string msg = os.str();
  std::cerr.write(msg.data(), static_cast<std::streamsize>(msg.size()));
  std::cerr.flush();
  // Give the MPI process manager a brief window to drain redirected stderr
  // pipes before MPI_Abort tears down the job.
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  MPI_Abort(comm, 1);
  std::_Exit(1);
}

} // namespace

void PipelineRunner::run(const RunConfig &cfg) {
  requireActiveMPI();
  const MPI_Comm comm = MPI_COMM_WORLD;
  int rank = 0;
  int size = 0;
  mpiCheck(MPI_Comm_rank(comm, &rank));
  mpiCheck(MPI_Comm_size(comm, &size));

  std::string modelName;
  try {
    validateRunConfig(cfg);
    auto m = ModelManifest::loadFromRax(cfg.raxPath);
    modelName = m.modelName;

    auto spec = parseStageV1(m.moduleAttrs);
    requireExternalWeights(m, spec);

    StageLibrary lib(m.soPath);
    auto stage = lib.create(m.weightPaths);
    runDistributedStage(*stage, spec, m.modelName, m.vocabPath, cfg, comm, rank,
                        size);
  } catch (const std::exception &err) {
    abortPipelineJob(comm, rank, cfg.raxPath, modelName, err.what());
  } catch (...) {
    abortPipelineJob(comm, rank, cfg.raxPath, modelName, "unknown exception");
  }
}

} // namespace buddy::runtime
