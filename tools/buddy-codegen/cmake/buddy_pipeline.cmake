# ===- buddy_pipeline.cmake - Shared pipeline-parallel stage RAX build ------===
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ===----------------------------------------------------------------------===
#
# Registers the MPI pipeline runner plugin and optional per-stage .rax build
# targets (<NAME>_pipeline_rax) for any model whose stages follow the shared
# Stage.h / StageManifest v1 ABI.
#
# Usage:
#   include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_pipeline.cmake)
#   buddy_add_pipeline_stages(
#     NAME qwen3_vl_moe
#     VAR_PREFIX QWEN3_VL_MOE
#     FALLBACK_VOCAB "${CMAKE_SOURCE_DIR}/examples/BuddyQwen3/vocab.txt"
#   )
#
# ===----------------------------------------------------------------------===

function(buddy_add_pipeline_stages)
  cmake_parse_arguments(_PP
    ""
    "NAME;VAR_PREFIX;FALLBACK_VOCAB;PIPELINE_SRC_DIR"
    ""
    ${ARGN}
  )

  if(NOT _PP_NAME)
    message(FATAL_ERROR "buddy_add_pipeline_stages requires NAME")
  endif()
  if(NOT _PP_VAR_PREFIX)
    string(TOUPPER "${_PP_NAME}" _PP_VAR_PREFIX)
  endif()
  if(NOT _PP_FALLBACK_VOCAB)
    message(FATAL_ERROR
      "buddy_add_pipeline_stages (${_PP_NAME}) requires FALLBACK_VOCAB")
  endif()
  if(NOT _PP_PIPELINE_SRC_DIR)
    set(_PP_PIPELINE_SRC_DIR "${CMAKE_SOURCE_DIR}/models/deepseek_r1/pipeline")
  endif()

  if(NOT Python3_EXECUTABLE)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
  endif()

  # Optional MPI pipeline-parallel runner plugin (skipped when MPI is unavailable).
  find_package(MPI COMPONENTS CXX QUIET)
  set(_PP_RUNNER_TARGET "buddy_models_${_PP_NAME}_pipeline_runner")
  if(MPI_CXX_FOUND)
    add_library(${_PP_RUNNER_TARGET} SHARED
      "${_PP_PIPELINE_SRC_DIR}/PipelineRunner.cpp"
      "${_PP_PIPELINE_SRC_DIR}/PipelineRunnerPlugin.cpp"
    )
    set_target_properties(${_PP_RUNNER_TARGET} PROPERTIES
      LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      OUTPUT_NAME "${_PP_NAME}_pipeline_runner"
      PREFIX ""
    )
    target_include_directories(${_PP_RUNNER_TARGET} PRIVATE
      "${_PP_PIPELINE_SRC_DIR}"
      "${CMAKE_SOURCE_DIR}/frontend/Interfaces"
      "${CMAKE_BINARY_DIR}/frontend/Interfaces"
    )
    target_link_libraries(${_PP_RUNNER_TARGET} PRIVATE
      buddy_runtime_core
      MPI::MPI_CXX
      ${CMAKE_DL_LIBS}
    )
    target_compile_features(${_PP_RUNNER_TARGET} PRIVATE
      cxx_std_17
    )
  else()
    message(STATUS
      "[${_PP_NAME}] MPI not found; skipping ${_PP_NAME}_pipeline_runner plugin.")
  endif()

  # ════════════════════════════════════════════════════════════════════════════
  # Pipeline-parallel stage RAX targets (<NAME>_pipeline_rax)
  #
  # Enabled only when <VAR_PREFIX>_PIPELINE_STAGE_DIRS is set (ordered list of
  # stage directories corresponding to MPI ranks 0, 1, ...). Unset by default so
  # normal whole-model builds are unaffected.
  # ════════════════════════════════════════════════════════════════════════════
  set(_PP_VAR_STAGE_DIRS "${_PP_VAR_PREFIX}_PIPELINE_STAGE_DIRS")
  set(_PP_VAR_EMBED_PAYLOAD "${_PP_VAR_PREFIX}_PIPELINE_EMBED_PAYLOAD")
  set(_PP_VAR_OUTPUT_DIR "${_PP_VAR_PREFIX}_PIPELINE_OUTPUT_DIR")
  set(_PP_VAR_RAX_PACK "${_PP_VAR_PREFIX}_PIPELINE_RAX_PACK")
  set(_PP_VAR_VOCAB_FILE "${_PP_VAR_PREFIX}_PIPELINE_VOCAB_FILE")
  set(_PP_VAR_RUNNER_LIB "${_PP_VAR_PREFIX}_PIPELINE_RUNNER_LIBRARY")

  set(${_PP_VAR_STAGE_DIRS} "" CACHE STRING
    "Ordered semicolon-separated list of ${_PP_NAME} pipeline stage directories (rank 0, 1, ...)")
  option(${_PP_VAR_EMBED_PAYLOAD}
    "Embed weights, stage.so, runner, deps, and vocab into stage<i>.rax (OFF = manifest-only)"
    ON)
  set(${_PP_VAR_OUTPUT_DIR} "${CMAKE_CURRENT_BINARY_DIR}/pipeline_rax" CACHE PATH
    "Directory where published ${_PP_NAME} stage<i>.rax artifacts are placed")
  set(${_PP_VAR_RAX_PACK} "" CACHE FILEPATH
    "Optional pre-built rax-pack executable (defaults to in-tree rax-pack target)")
  set(${_PP_VAR_VOCAB_FILE} "${_PP_FALLBACK_VOCAB}" CACHE FILEPATH
    "Fallback vocab.txt for the first ${_PP_NAME} pipeline stage when absent from its stage dir")
  set(RISCV_MPI_DIR "" CACHE PATH
    "Prefix of target riscv64 MPI installation (include/mpi.h, lib/libmpi.so) for IS_RVV_CROSSCOMPILE")
  set(${_PP_VAR_RUNNER_LIB} "" CACHE FILEPATH
    "Optional pre-built target ${_PP_NAME}_pipeline_runner.so for IS_RVV_CROSSCOMPILE")

  set(_PP_STAGE_DIRS "${${_PP_VAR_STAGE_DIRS}}")
  if(NOT _PP_STAGE_DIRS)
    return()
  endif()

  set(_PP_EMBED_PAYLOAD "${${_PP_VAR_EMBED_PAYLOAD}}")
  set(_PP_OUTPUT_DIR "${${_PP_VAR_OUTPUT_DIR}}")
  set(_PP_RAX_PACK "${${_PP_VAR_RAX_PACK}}")
  set(_PP_VOCAB_FALLBACK "${${_PP_VAR_VOCAB_FILE}}")
  set(_PP_RUNNER_LIBRARY "${${_PP_VAR_RUNNER_LIB}}")

  if(NOT IS_RVV_CROSSCOMPILE AND NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
      "[${_PP_NAME}] ${_PP_VAR_STAGE_DIRS} is set, but MPI (CXX) was not found. "
      "Building ${_PP_NAME}_pipeline_rax requires MPI for ${_PP_NAME}_pipeline_runner.so.")
  endif()

  set(_PP_VERIFY_SCRIPT
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/verify_pipeline_stage.py")
  set(_PP_GEN_MANIFEST_SCRIPT
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_manifest.py")
  foreach(_req_script "${_PP_VERIFY_SCRIPT}" "${_PP_GEN_MANIFEST_SCRIPT}")
    if(NOT EXISTS "${_req_script}")
      message(FATAL_ERROR
        "[${_PP_NAME}] Missing required pipeline script: ${_req_script}")
    endif()
  endforeach()

  if(_PP_RAX_PACK)
    if(NOT EXISTS "${_PP_RAX_PACK}")
      message(FATAL_ERROR
        "[${_PP_NAME}] ${_PP_VAR_RAX_PACK} not found: ${_PP_RAX_PACK}")
    endif()
    set(_PP_RAX_PACK_BIN "${_PP_RAX_PACK}")
    set(_PP_RAX_PACK_DEP "${_PP_RAX_PACK}")
  else()
    set(_PP_RAX_PACK_BIN "$<TARGET_FILE:rax-pack>")
    set(_PP_RAX_PACK_DEP rax-pack)
  endif()

  # Resolve compiler, symbol checker, runner plugin, and MLIR/OpenMP runtime
  # library directories for either host or RISC-V cross-compilation.
  set(_PP_RUNNER_NAME "${_PP_NAME}_pipeline_runner${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(_PP_RUNNER_FILE "")
  set(_PP_RUNNER_DEPS)
  set(_PP_MLIR_RUNTIME_DIR "")
  set(_PP_CRUNNER_CAND "")
  set(_PP_OMP_LIB "")

  if(IS_RVV_CROSSCOMPILE)
    if(NOT RISCV_GNU_TOOLCHAIN)
      message(FATAL_ERROR
        "[${_PP_NAME}] IS_RVV_CROSSCOMPILE=ON requires RISCV_GNU_TOOLCHAIN for ${_PP_NAME}_pipeline_rax.")
    endif()
    if(NOT RISCV_MLIR_C_RUNNER_UTILS OR NOT EXISTS "${RISCV_MLIR_C_RUNNER_UTILS}")
      message(FATAL_ERROR
        "[${_PP_NAME}] IS_RVV_CROSSCOMPILE=ON requires RISCV_MLIR_C_RUNNER_UTILS (got '${RISCV_MLIR_C_RUNNER_UTILS}').")
    endif()

    set(_PP_CRUNNER_CAND "${RISCV_MLIR_C_RUNNER_UTILS}")
    get_filename_component(_PP_MLIR_RUNTIME_DIR "${RISCV_MLIR_C_RUNNER_UTILS}" DIRECTORY)
    if(RISCV_OMP_SHARED AND EXISTS "${RISCV_OMP_SHARED}")
      set(_PP_OMP_LIB "${RISCV_OMP_SHARED}")
    endif()

    # Locate cross-capable clang/clang++.
    set(_PP_CROSS_CC "")
    set(_PP_CROSS_CXX "")
    foreach(_clang_dir "${LLVM_TOOLS_BINARY_DIR}" "${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin")
      if(NOT _PP_CROSS_CC AND EXISTS "${_clang_dir}/clang")
        set(_PP_CROSS_CC "${_clang_dir}/clang")
      endif()
      if(NOT _PP_CROSS_CXX AND EXISTS "${_clang_dir}/clang++")
        set(_PP_CROSS_CXX "${_clang_dir}/clang++")
      endif()
    endforeach()
    if(NOT _PP_CROSS_CC)
      set(_PP_CROSS_CC "${CMAKE_C_COMPILER}")
    endif()
    if(NOT _PP_CROSS_CXX)
      set(_PP_CROSS_CXX "${CMAKE_CXX_COMPILER}")
    endif()

    find_program(_PP_NM
      NAMES riscv64-linux-gnu-nm llvm-nm nm
      HINTS "${LLVM_TOOLS_BINARY_DIR}")
    if(NOT _PP_NM)
      set(_PP_NM "nm")
    endif()

    set(_PP_RV_MARCH "rv64gcv")
    if(BUDDY_RISCV_ENABLE_ZFH_ZVFH)
      string(APPEND _PP_RV_MARCH "_zfh_zvfh")
    endif()
    set(_PP_RV_TARGET_OPTS
      --target=riscv64-unknown-linux-gnu
      "--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot"
      "--gcc-toolchain=${RISCV_GNU_TOOLCHAIN}")

    if(_PP_RUNNER_LIBRARY)
      if(NOT EXISTS "${_PP_RUNNER_LIBRARY}")
        message(FATAL_ERROR
          "[${_PP_NAME}] ${_PP_VAR_RUNNER_LIB} not found: ${_PP_RUNNER_LIBRARY}")
      endif()
      set(_PP_RUNNER_FILE "${_PP_RUNNER_LIBRARY}")
      set(_PP_RUNNER_DEPS "${_PP_RUNNER_LIBRARY}")
    elseif(RISCV_MPI_DIR)
      if(NOT EXISTS "${RISCV_MPI_DIR}/include/mpi.h" OR
         NOT EXISTS "${RISCV_MPI_DIR}/lib/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}")
        message(FATAL_ERROR
          "[${_PP_NAME}] RISCV_MPI_DIR='${RISCV_MPI_DIR}' must contain include/mpi.h and lib/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}.")
      endif()
      set(_PP_RV_RUNNER_DIR "${CMAKE_CURRENT_BINARY_DIR}/pipeline_stages/rv64")
      set(_PP_RV_FB_DIR "${_PP_RV_RUNNER_DIR}/flatbuffers-include")
      file(MAKE_DIRECTORY "${_PP_RV_FB_DIR}")
      file(CREATE_LINK "${FLATBUFFERS_INCLUDE_DIR}/flatbuffers"
           "${_PP_RV_FB_DIR}/flatbuffers" SYMBOLIC)

      set(_PP_RV_RUNNER_INCS
        "-I${_PP_PIPELINE_SRC_DIR}"
        "-I${CMAKE_SOURCE_DIR}/runtime/include"
        "-I${CMAKE_BINARY_DIR}/runtime/include"
        "-I${_PP_RV_FB_DIR}"
        "-I${CMAKE_SOURCE_DIR}/frontend/Interfaces"
        "-I${CMAKE_BINARY_DIR}/frontend/Interfaces"
        "-I${RISCV_MPI_DIR}/include")
      set(_PP_RV_RUNNER_OBJS)
      foreach(_rsrc PipelineRunner.cpp PipelineRunnerPlugin.cpp)
        get_filename_component(_rname "${_rsrc}" NAME_WE)
        set(_robj "${_PP_RV_RUNNER_DIR}/${_rname}.o")
        add_custom_command(
          OUTPUT "${_robj}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${_PP_RV_RUNNER_DIR}"
          COMMAND "${_PP_CROSS_CXX}" ${_PP_RV_TARGET_OPTS}
            -std=c++17 -O2 -DNDEBUG -fPIC "-march=${_PP_RV_MARCH}"
            ${_PP_RV_RUNNER_INCS}
            -MD -MF "${_robj}.d"
            -c "${_PP_PIPELINE_SRC_DIR}/${_rsrc}" -o "${_robj}"
          DEPENDS "${_PP_PIPELINE_SRC_DIR}/${_rsrc}" buddy-rax-gen
          DEPFILE "${_robj}.d"
          COMMENT "[${_PP_NAME}_pipeline] riscv64: compiling ${_rname}.cpp"
          VERBATIM
        )
        list(APPEND _PP_RV_RUNNER_OBJS "${_robj}")
      endforeach()
      set(_PP_RV_RUNNER_SO "${_PP_RV_RUNNER_DIR}/${_PP_RUNNER_NAME}")
      add_custom_command(
        OUTPUT "${_PP_RV_RUNNER_SO}"
        COMMAND "${_PP_CROSS_CXX}" ${_PP_RV_TARGET_OPTS}
          -shared -fPIC
          "-Wl,-soname,${_PP_RUNNER_NAME}"
          "-Wl,-rpath,\$ORIGIN"
          "-Wl,-z,defs"
          -o "${_PP_RV_RUNNER_SO}"
          ${_PP_RV_RUNNER_OBJS}
          "-L${RISCV_MPI_DIR}/lib" -lmpi -ldl -lpthread
        DEPENDS ${_PP_RV_RUNNER_OBJS}
        COMMENT "[${_PP_NAME}_pipeline] riscv64: linking ${_PP_RUNNER_NAME}"
        VERBATIM
      )
      set(_PP_RV_RUNNER_TARGET "${_PP_NAME}_pipeline_runner_rv64")
      add_custom_target(${_PP_RV_RUNNER_TARGET}
        DEPENDS "${_PP_RV_RUNNER_SO}"
      )
      set(_PP_RUNNER_FILE "${_PP_RV_RUNNER_SO}")
      set(_PP_RUNNER_DEPS ${_PP_RV_RUNNER_TARGET} "${_PP_RV_RUNNER_SO}")
    elseif(_PP_EMBED_PAYLOAD)
      message(FATAL_ERROR
        "[${_PP_NAME}] IS_RVV_CROSSCOMPILE=ON with ${_PP_VAR_EMBED_PAYLOAD}=ON requires "
        "RISCV_MPI_DIR (to cross-compile ${_PP_NAME}_pipeline_runner.so) or "
        "${_PP_VAR_RUNNER_LIB} (or set ${_PP_VAR_EMBED_PAYLOAD}=OFF).")
    else()
      message(STATUS
        "[${_PP_NAME}] IS_RVV_CROSSCOMPILE=ON without RISCV_MPI_DIR: packing manifest-only "
        "stage .rax without copying ${_PP_NAME}_pipeline_runner.so.")
    endif()
  else()
    set(_PP_NM "nm")
    set(_PP_RUNNER_NAME "$<TARGET_FILE_NAME:${_PP_RUNNER_TARGET}>")
    set(_PP_RUNNER_FILE "$<TARGET_FILE:${_PP_RUNNER_TARGET}>")
    set(_PP_RUNNER_DEPS
      ${_PP_RUNNER_TARGET}
      "$<TARGET_FILE:${_PP_RUNNER_TARGET}>")

    set(_PP_MLIR_RUNTIME_DIR "${LLVM_LIBRARY_DIR}")
    set(_PP_CRUNNER_CAND
      "${LLVM_LIBRARY_DIR}/libmlir_c_runner_utils${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_PP_CRUNNER_CAND}")
      message(FATAL_ERROR
        "[${_PP_NAME}] Missing required MLIR runtime library: ${_PP_CRUNNER_CAND}")
    endif()

    if(BUDDY_OPENMP_RUNTIME_LIBRARY AND EXISTS "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
      set(_PP_OMP_LIB "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
    else()
      get_filename_component(_PP_LLVM_ROOT "${LLVM_LIBRARY_DIR}" DIRECTORY)
      foreach(_omp_cand
        "${_PP_LLVM_ROOT}/runtimes/runtimes-bins/openmp/runtime/src/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${LLVM_LIBRARY_DIR}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
      )
        if(EXISTS "${_omp_cand}")
          set(_PP_OMP_LIB "${_omp_cand}")
          break()
        endif()
      endforeach()
    endif()
    if(NOT _PP_OMP_LIB)
      find_library(_PP_OMP_LIB NAMES omp)
    endif()
  endif()

  file(REAL_PATH "${_PP_CRUNNER_CAND}" _PP_CRUNNER_REAL)
  get_filename_component(_PP_CRUNNER_NAME "${_PP_CRUNNER_REAL}" NAME)

  set(_PP_OMP_REAL "")
  set(_PP_OMP_NAME "")
  if(_PP_OMP_LIB AND EXISTS "${_PP_OMP_LIB}")
    file(REAL_PATH "${_PP_OMP_LIB}" _PP_OMP_REAL)
    get_filename_component(_PP_OMP_NAME "${_PP_OMP_REAL}" NAME)
  endif()

  # Base MLIR runtime shared library dependencies (in load order).
  set(_PP_BASE_DEP_PATHS)
  set(_PP_BASE_DEP_NAMES)
  foreach(_mlir_dep_base libmlir_apfloat_wrappers libmlir_float16_utils)
    set(_mlir_dep_cand
      "${_PP_MLIR_RUNTIME_DIR}/${_mlir_dep_base}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_mlir_dep_cand}")
      file(GLOB _mlir_dep_globs
        "${_PP_MLIR_RUNTIME_DIR}/${_mlir_dep_base}${CMAKE_SHARED_LIBRARY_SUFFIX}.*")
      if(_mlir_dep_globs)
        list(GET _mlir_dep_globs 0 _mlir_dep_cand)
      endif()
    endif()
    if(EXISTS "${_mlir_dep_cand}")
      file(REAL_PATH "${_mlir_dep_cand}" _mlir_dep_real)
      get_filename_component(_mlir_dep_name "${_mlir_dep_real}" NAME)
      list(APPEND _PP_BASE_DEP_PATHS "${_mlir_dep_real}")
      list(APPEND _PP_BASE_DEP_NAMES "${_mlir_dep_name}")
    endif()
  endforeach()
  list(APPEND _PP_BASE_DEP_PATHS "${_PP_CRUNNER_REAL}")
  list(APPEND _PP_BASE_DEP_NAMES "${_PP_CRUNNER_NAME}")

  get_filename_component(_PP_OUT_DIR "${_PP_OUTPUT_DIR}" ABSOLUTE)

  set(_PP_SYMCHECK_PY
    "import os, subprocess, sys; so = sys.argv[1]; nm_bin = sys.argv[2] if len(sys.argv) > 2 else 'nm'; req = {'buddy_create_pipeline_stage_v1', 'buddy_create_pipeline_stage_v2', 'buddy_destroy_pipeline_stage_v1', '_mlir_ciface_forward_prefill', '_mlir_ciface_forward_decode'}; out = subprocess.check_output([nm_bin, '-D', '--defined-only', so], text=True); defs = {l.split()[-1].split('@')[0] for l in out.splitlines() if l.strip()}; miss = sorted(req - defs); miss and (os.remove(so), sys.exit(f'stage.so missing required symbols: {miss}'))")
  # Hardlink large weight blobs into staging (fall back to copy across filesystems).
  # Avoid semicolons and literal newlines so the snippet survives CMake list(APPEND) and Ninja.
  set(_PP_HARDLINK_PY
    "exec('import os, shutil, sys\\ns, d = sys.argv[1], sys.argv[2]\\nif os.path.lexists(d):\\n  try:\\n    if not os.path.islink(d) and os.path.samefile(s, d): sys.exit(0)\\n  except OSError: pass\\n  os.unlink(d)\\ntry: os.link(s, d)\\nexcept OSError: shutil.copy2(s, d)')")

  set(_PP_STAGE_TARGETS)
  set(_PP_RANK 0)
  foreach(_PP_RAW_DIR IN LISTS _PP_STAGE_DIRS)
    get_filename_component(_PP_STAGE_DIR "${_PP_RAW_DIR}" ABSOLUTE)
    if(NOT IS_DIRECTORY "${_PP_STAGE_DIR}")
      message(FATAL_ERROR
        "[${_PP_NAME}] stage ${_PP_RANK}: stage directory does not exist: ${_PP_STAGE_DIR}")
    endif()

    set(_PP_CONFIG "${_PP_STAGE_DIR}/config.json")
    if(NOT EXISTS "${_PP_CONFIG}")
      message(FATAL_ERROR
        "[${_PP_NAME}] stage ${_PP_RANK}: missing required input file: ${_PP_CONFIG}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_PP_CONFIG}")

    file(READ "${_PP_CONFIG}" _PP_CFG_JSON)
    string(JSON _PP_START_LAYER ERROR_VARIABLE _PP_JSON_ERR
      GET "${_PP_CFG_JSON}" pipeline_stage start_layer)
    if(_PP_JSON_ERR)
      message(FATAL_ERROR
        "[${_PP_NAME}] stage ${_PP_RANK}: failed to read pipeline_stage.start_layer from ${_PP_CONFIG}: ${_PP_JSON_ERR}")
    endif()
    string(JSON _PP_VOCAB_FILE_NAME ERROR_VARIABLE _PP_JSON_ERR
      GET "${_PP_CFG_JSON}" tokens vocab_file)
    if(_PP_JSON_ERR)
      message(FATAL_ERROR
        "[${_PP_NAME}] stage ${_PP_RANK}: failed to read tokens.vocab_file from ${_PP_CONFIG}: ${_PP_JSON_ERR}")
    endif()

    string(JSON _PP_VARIANT ERROR_VARIABLE _PP_VAR_ERR
      GET "${_PP_CFG_JSON}" variant)
    if(_PP_VAR_ERR)
      set(_PP_VARIANT "f16")
    endif()
    set(_PP_IS_W4G32 FALSE)
    if(_PP_VARIANT STREQUAL "w4g32")
      set(_PP_IS_W4G32 TRUE)
    endif()

    set(_PP_USE_ARENA FALSE)
    string(JSON _PP_ARENA_VAL ERROR_VARIABLE _PP_OPT_ERR
      GET "${_PP_CFG_JSON}" arena)
    if(NOT _PP_OPT_ERR AND _PP_ARENA_VAL)
      set(_PP_USE_ARENA TRUE)
    endif()

    set(_PP_USE_THREAD_POOL FALSE)
    string(JSON _PP_TP_VAL ERROR_VARIABLE _PP_OPT_ERR
      GET "${_PP_CFG_JSON}" thread_pool)
    if(NOT _PP_OPT_ERR AND _PP_TP_VAL)
      set(_PP_USE_THREAD_POOL TRUE)
    endif()

    set(_PP_USE_PREFILL_IME FALSE)
    string(JSON _PP_IME_VAL ERROR_VARIABLE _PP_OPT_ERR
      GET "${_PP_CFG_JSON}" prefill_ime)
    if(NOT _PP_OPT_ERR AND _PP_IME_VAL)
      set(_PP_USE_PREFILL_IME TRUE)
    endif()

    set(_PP_ARENA_PREFAULT_MB 0)
    string(JSON _PP_PREFAULT_VAL ERROR_VARIABLE _PP_OPT_ERR
      GET "${_PP_CFG_JSON}" arena_prefault_mb)
    if(NOT _PP_OPT_ERR AND _PP_PREFAULT_VAL)
      set(_PP_ARENA_PREFAULT_MB "${_PP_PREFAULT_VAL}")
    endif()

    string(JSON _PP_WEIGHT_COUNT ERROR_VARIABLE _PP_JSON_ERR
      LENGTH "${_PP_CFG_JSON}" weights)
    if(_PP_JSON_ERR OR _PP_WEIGHT_COUNT LESS 1)
      message(FATAL_ERROR
        "[${_PP_NAME}] stage ${_PP_RANK}: failed to read weights array from ${_PP_CONFIG}: ${_PP_JSON_ERR}")
    endif()
    set(_PP_WEIGHT_FILES)
    set(_PP_WEIGHT_PATHS)
    math(EXPR _PP_WEIGHT_LAST "${_PP_WEIGHT_COUNT} - 1")
    foreach(_widx RANGE 0 ${_PP_WEIGHT_LAST})
      string(JSON _wfile ERROR_VARIABLE _PP_JSON_ERR
        GET "${_PP_CFG_JSON}" weights ${_widx} file)
      if(_PP_JSON_ERR)
        message(FATAL_ERROR
          "[${_PP_NAME}] stage ${_PP_RANK}: failed to read weights[${_widx}].file from ${_PP_CONFIG}: ${_PP_JSON_ERR}")
      endif()
      list(APPEND _PP_WEIGHT_FILES "${_wfile}")
      list(APPEND _PP_WEIGHT_PATHS "${_PP_STAGE_DIR}/${_wfile}")
    endforeach()

    set(_PP_IS_FIRST FALSE)
    if(_PP_START_LAYER EQUAL 0)
      set(_PP_IS_FIRST TRUE)
    endif()

    if(_PP_IS_W4G32)
      set(_PP_MLIR_SUFFIX "-w4g32")
    else()
      set(_PP_MLIR_SUFFIX "-f16")
    endif()

    set(_PP_PREFILL_MLIR "${_PP_STAGE_DIR}/forward_prefill${_PP_MLIR_SUFFIX}.mlir")
    set(_PP_DECODE_MLIR "${_PP_STAGE_DIR}/forward_decode${_PP_MLIR_SUFFIX}.mlir")
    set(_PP_PREFILL_O "${_PP_STAGE_DIR}/forward_prefill.o")
    set(_PP_DECODE_O "${_PP_STAGE_DIR}/forward_decode.o")
    set(_PP_SUB_PREFILL_O "${_PP_STAGE_DIR}/subgraph_prefill.o")
    set(_PP_SUB_DECODE_O "${_PP_STAGE_DIR}/subgraph_decode.o")

    set(_PP_VERIFY_DEPS
      "${_PP_VERIFY_SCRIPT}"
      "${_PP_CONFIG}"
      "${_PP_PREFILL_MLIR}"
      "${_PP_DECODE_MLIR}"
      ${_PP_WEIGHT_PATHS}
    )
    set(_PP_OBJS
      "${_PP_PREFILL_O}"
      "${_PP_DECODE_O}"
      "${_PP_SUB_PREFILL_O}"
      "${_PP_SUB_DECODE_O}"
    )
    set(_PP_REQ_FILES
      "${_PP_PREFILL_MLIR}"
      "${_PP_DECODE_MLIR}"
      ${_PP_OBJS}
      ${_PP_WEIGHT_PATHS}
    )
    if(_PP_IS_W4G32)
      set(_PP_K3_MLIR "${_PP_STAGE_DIR}/k3_kernels-w4g32.mlir")
      set(_PP_K3_O "${_PP_STAGE_DIR}/k3_kernels.o")
      list(APPEND _PP_VERIFY_DEPS "${_PP_K3_MLIR}")
      list(APPEND _PP_OBJS "${_PP_K3_O}")
      list(APPEND _PP_REQ_FILES "${_PP_K3_MLIR}" "${_PP_K3_O}")
      if(_PP_USE_PREFILL_IME)
        set(_PP_K3_IME_MLIR "${_PP_STAGE_DIR}/k3_kernels_ime-w4g32.mlir")
        set(_PP_K3_IME_O "${_PP_STAGE_DIR}/k3_kernels_ime.o")
        list(APPEND _PP_VERIFY_DEPS "${_PP_K3_IME_MLIR}")
        list(APPEND _PP_OBJS "${_PP_K3_IME_O}")
        list(APPEND _PP_REQ_FILES "${_PP_K3_IME_MLIR}" "${_PP_K3_IME_O}")
      endif()
    endif()

    foreach(_req_file IN LISTS _PP_REQ_FILES)
      if(NOT EXISTS "${_req_file}")
        message(FATAL_ERROR
          "[${_PP_NAME}] stage ${_PP_RANK}: missing required input file: ${_req_file}")
      endif()
    endforeach()

    # Per-stage runtime shared libraries (skip libomp when thread_pool=true).
    set(_PP_DEP_LIB_PATHS ${_PP_BASE_DEP_PATHS})
    set(_PP_DEP_LIB_NAMES ${_PP_BASE_DEP_NAMES})
    if(NOT _PP_USE_THREAD_POOL)
      if(NOT _PP_OMP_REAL)
        message(FATAL_ERROR
          "[${_PP_NAME}] stage ${_PP_RANK}: OpenMP runtime library required when thread_pool=false "
          "(set BUDDY_OPENMP_RUNTIME_LIBRARY or RISCV_OMP_SHARED).")
      endif()
      list(APPEND _PP_DEP_LIB_PATHS "${_PP_OMP_REAL}")
      list(APPEND _PP_DEP_LIB_NAMES "${_PP_OMP_NAME}")
    endif()

    set(_PP_DEP_MANIFEST_ARGS)
    foreach(_dep_name IN LISTS _PP_DEP_LIB_NAMES)
      list(APPEND _PP_DEP_MANIFEST_ARGS --dep-shared-lib "${_dep_name}")
    endforeach()
    list(LENGTH _PP_DEP_LIB_PATHS _PP_DEP_COUNT)
    math(EXPR _PP_DEP_LAST "${_PP_DEP_COUNT} - 1")

    set(_PP_VOCAB_SRC "")
    if(_PP_IS_FIRST)
      if(EXISTS "${_PP_STAGE_DIR}/${_PP_VOCAB_FILE_NAME}")
        set(_PP_VOCAB_SRC "${_PP_STAGE_DIR}/${_PP_VOCAB_FILE_NAME}")
      else()
        set(_PP_VOCAB_SRC "${_PP_VOCAB_FALLBACK}")
      endif()
      if(NOT EXISTS "${_PP_VOCAB_SRC}")
        message(FATAL_ERROR
          "[${_PP_NAME}] stage ${_PP_RANK}: missing required vocab file: ${_PP_VOCAB_SRC}")
      endif()
    endif()

    set(_PP_WORK_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/pipeline_stages/stage${_PP_RANK}")
    set(_PP_CONFIG_INC "${_PP_WORK_DIR}/stage-config.inc")
    set(_PP_MANIFEST "${_PP_WORK_DIR}/stage.mlir")

    # 1. Verify stage MLIR ABI & weights, emit stage-config.inc.
    add_custom_command(
      OUTPUT "${_PP_CONFIG_INC}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_PP_WORK_DIR}"
      COMMAND "${Python3_EXECUTABLE}" "${_PP_VERIFY_SCRIPT}"
        --stage-dir "${_PP_STAGE_DIR}"
        --header "${_PP_CONFIG_INC}"
      DEPENDS ${_PP_VERIFY_DEPS}
      COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK}: verifying ABI and generating stage-config.inc"
      VERBATIM
    )

    # Runtime C support sources (arena / thread_pool / SpacemiT TCM).
    set(_PP_RT_C_SRCS)
    if(_PP_USE_ARENA)
      list(APPEND _PP_RT_C_SRCS "${CMAKE_SOURCE_DIR}/runtime/arena/BuddyArena.c")
    endif()
    if(_PP_USE_PREFILL_IME)
      list(APPEND _PP_RT_C_SRCS "${CMAKE_SOURCE_DIR}/runtime/spacemit/BuddySpacemitTcm.c")
    endif()
    if(_PP_USE_THREAD_POOL)
      list(APPEND _PP_RT_C_SRCS "${CMAKE_SOURCE_DIR}/runtime/threadpool/BuddyThreadPool.c")
    endif()

    # 2. Compile Stage.cpp (+ runtime support) and link stage.so with symbol closure & export check.
    set(_PP_SO_TARGET "${_PP_NAME}_pipeline_stage_${_PP_RANK}_so")
    if(IS_RVV_CROSSCOMPILE)
      set(_PP_STAGE_SO_FILE "${_PP_WORK_DIR}/stage.so")
      set(_PP_RT_C_OBJS)
      foreach(_csrc IN LISTS _PP_RT_C_SRCS)
        get_filename_component(_cname "${_csrc}" NAME_WE)
        set(_cobj "${_PP_WORK_DIR}/${_cname}.o")
        set(_copts ${_PP_RV_TARGET_OPTS} -O2 -fPIC)
        if(_cname STREQUAL "BuddyArena" AND _PP_ARENA_PREFAULT_MB GREATER 0)
          list(APPEND _copts "-DBUDDY_ARENA_DEFAULT_PREFAULT_MB=${_PP_ARENA_PREFAULT_MB}")
        endif()
        add_custom_command(
          OUTPUT "${_cobj}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${_PP_WORK_DIR}"
          COMMAND "${_PP_CROSS_CC}" ${_copts} -c "${_csrc}" -o "${_cobj}"
          DEPENDS "${_csrc}" "${_PP_CONFIG}"
          COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK} riscv64: compiling ${_cname}.c"
          VERBATIM
        )
        list(APPEND _PP_RT_C_OBJS "${_cobj}")
      endforeach()

      set(_PP_MLIR_INC_ARGS)
      foreach(_inc IN LISTS MLIR_INCLUDE_DIRS)
        if(NOT _inc MATCHES "^/usr(/local)?/include/?$")
          list(APPEND _PP_MLIR_INC_ARGS "-I${_inc}")
        endif()
      endforeach()

      set(_PP_STAGE_OBJ "${_PP_WORK_DIR}/Stage.o")
      add_custom_command(
        OUTPUT "${_PP_STAGE_OBJ}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_PP_WORK_DIR}"
        COMMAND "${_PP_CROSS_CXX}" ${_PP_RV_TARGET_OPTS}
          -std=c++17 -O2 -DNDEBUG -fPIC "-march=${_PP_RV_MARCH}"
          "-I${_PP_WORK_DIR}"
          ${_PP_MLIR_INC_ARGS}
          "-I${CMAKE_SOURCE_DIR}/runtime/include"
          -c "${_PP_PIPELINE_SRC_DIR}/Stage.cpp"
          -o "${_PP_STAGE_OBJ}"
        DEPENDS
          "${_PP_PIPELINE_SRC_DIR}/Stage.cpp"
          "${_PP_CONFIG_INC}"
        COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK} riscv64: compiling Stage.cpp"
        VERBATIM
      )

      set(_PP_RV_LINK_LIBS "${_PP_CRUNNER_REAL}" -lm -lpthread -ldl)
      if(NOT _PP_USE_THREAD_POOL)
        list(APPEND _PP_RV_LINK_LIBS "${_PP_OMP_REAL}")
      endif()

      add_custom_command(
        OUTPUT "${_PP_STAGE_SO_FILE}"
        COMMAND "${_PP_CROSS_CXX}" ${_PP_RV_TARGET_OPTS}
          -shared -fPIC
          -Wl,-soname,stage.so
          -Wl,--allow-multiple-definition
          -Wl,-z,defs
          "-Wl,-rpath,\$ORIGIN"
          -o "${_PP_STAGE_SO_FILE}"
          "${_PP_STAGE_OBJ}"
          ${_PP_OBJS}
          ${_PP_RT_C_OBJS}
          ${_PP_RV_LINK_LIBS}
        COMMAND "${Python3_EXECUTABLE}" -c "${_PP_SYMCHECK_PY}"
          "${_PP_STAGE_SO_FILE}" "${_PP_NM}"
        DEPENDS
          "${_PP_STAGE_OBJ}"
          ${_PP_OBJS}
          ${_PP_RT_C_OBJS}
        COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK} riscv64: linking and verifying stage.so"
        VERBATIM
      )
      add_custom_target(${_PP_SO_TARGET} DEPENDS "${_PP_STAGE_SO_FILE}")
    else()
      set(_PP_STAGE_SO_FILE "$<TARGET_FILE:${_PP_SO_TARGET}>")
      set_source_files_properties(${_PP_OBJS} PROPERTIES
        EXTERNAL_OBJECT TRUE
      )
      add_library(${_PP_SO_TARGET} SHARED EXCLUDE_FROM_ALL
        "${_PP_PIPELINE_SRC_DIR}/Stage.cpp"
        "${_PP_CONFIG_INC}"
        ${_PP_RT_C_SRCS}
        ${_PP_OBJS}
      )
      set_target_properties(${_PP_SO_TARGET} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_PP_WORK_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${_PP_WORK_DIR}"
        OUTPUT_NAME "stage"
        PREFIX ""
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "\$ORIGIN"
      )
      target_include_directories(${_PP_SO_TARGET} PRIVATE
        "${_PP_WORK_DIR}"
        ${MLIR_INCLUDE_DIRS}
        "${CMAKE_SOURCE_DIR}/runtime/include"
      )
      if(_PP_USE_ARENA AND _PP_ARENA_PREFAULT_MB GREATER 0)
        target_compile_definitions(${_PP_SO_TARGET} PRIVATE
          "BUDDY_ARENA_DEFAULT_PREFAULT_MB=${_PP_ARENA_PREFAULT_MB}")
      endif()
      set(_PP_HOST_LINK_LIBS "${_PP_CRUNNER_REAL}")
      if(_PP_USE_THREAD_POOL)
        list(APPEND _PP_HOST_LINK_LIBS -lpthread ${CMAKE_DL_LIBS})
      else()
        list(APPEND _PP_HOST_LINK_LIBS "${_PP_OMP_REAL}")
      endif()
      target_link_libraries(${_PP_SO_TARGET} PRIVATE ${_PP_HOST_LINK_LIBS})
      if(NOT APPLE)
        target_link_options(${_PP_SO_TARGET} PRIVATE "-Wl,-z,defs")
        if(_PP_IS_W4G32)
          target_link_options(${_PP_SO_TARGET} PRIVATE
            "-Wl,--allow-multiple-definition")
        endif()
      endif()
      target_compile_features(${_PP_SO_TARGET} PRIVATE cxx_std_17)
      add_custom_command(TARGET ${_PP_SO_TARGET} POST_BUILD
        COMMAND "${Python3_EXECUTABLE}" -c "${_PP_SYMCHECK_PY}"
          "$<TARGET_FILE:${_PP_SO_TARGET}>" "${_PP_NM}"
        COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK}: verifying stage.so factory and MLIR entry symbols"
        VERBATIM
      )
    endif()

    # 3. Generate per-stage RHAL manifest (stage.mlir).
    set(_PP_VERSION_ARGS)
    if(DEFINED BUDDY_VERSION AND NOT "${BUDDY_VERSION}" STREQUAL "")
      set(_PP_VERSION_ARGS --version "${BUDDY_VERSION}")
    endif()
    add_custom_command(
      OUTPUT "${_PP_MANIFEST}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_PP_WORK_DIR}"
      COMMAND "${Python3_EXECUTABLE}" "${_PP_GEN_MANIFEST_SCRIPT}"
        --config "${_PP_CONFIG}"
        ${_PP_DEP_MANIFEST_ARGS}
        --runner-library "${_PP_RUNNER_NAME}"
        ${_PP_VERSION_ARGS}
        -o "${_PP_MANIFEST}"
      DEPENDS
        "${_PP_GEN_MANIFEST_SCRIPT}"
        "${_PP_CONFIG}"
      COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK}: generating RHAL manifest"
      VERBATIM
    )

    # 4. Stage referenced files, run rax-pack, and publish stage<i>.rax.
    if(_PP_EMBED_PAYLOAD)
      set(_PP_STAGE_OUT_DIR "${_PP_WORK_DIR}")
      set(_PP_RAX_TMP "${_PP_WORK_DIR}/stage${_PP_RANK}.rax.tmp")
      set(_PP_RAX_OUT "${_PP_OUT_DIR}/stage${_PP_RANK}.rax")
      set(_PP_RAX_PACK_EXTRA_ARGS --embed-payload)
      set(_PP_STAGE_SO_CMD)
    else()
      set(_PP_STAGE_OUT_DIR "${_PP_OUT_DIR}/stage${_PP_RANK}")
      set(_PP_RAX_TMP "${_PP_STAGE_OUT_DIR}/stage${_PP_RANK}.rax.tmp")
      set(_PP_RAX_OUT "${_PP_STAGE_OUT_DIR}/stage${_PP_RANK}.rax")
      set(_PP_RAX_PACK_EXTRA_ARGS)
      set(_PP_STAGE_SO_CMD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_PP_STAGE_SO_FILE}"
          "${_PP_STAGE_OUT_DIR}/stage.so")
    endif()

    set(_PP_STAGE_RUNNER_CMD)
    if(_PP_RUNNER_FILE)
      set(_PP_STAGE_RUNNER_CMD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_PP_RUNNER_FILE}"
          "${_PP_STAGE_OUT_DIR}/${_PP_RUNNER_NAME}")
    endif()

    set(_PP_STAGE_DEP_CMDS)
    foreach(_dep_idx RANGE 0 ${_PP_DEP_LAST})
      list(GET _PP_DEP_LIB_PATHS ${_dep_idx} _dep_path)
      list(GET _PP_DEP_LIB_NAMES ${_dep_idx} _dep_name)
      list(APPEND _PP_STAGE_DEP_CMDS
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_dep_path}" "${_PP_STAGE_OUT_DIR}/${_dep_name}")
    endforeach()

    set(_PP_STAGE_WEIGHT_CMDS)
    foreach(_widx RANGE 0 ${_PP_WEIGHT_LAST})
      list(GET _PP_WEIGHT_PATHS ${_widx} _wpath)
      list(GET _PP_WEIGHT_FILES ${_widx} _wfile)
      list(APPEND _PP_STAGE_WEIGHT_CMDS
        COMMAND "${Python3_EXECUTABLE}" -c "${_PP_HARDLINK_PY}"
          "${_wpath}" "${_PP_STAGE_OUT_DIR}/${_wfile}")
    endforeach()

    set(_PP_STAGE_VOCAB_CMD)
    set(_PP_VOCAB_DEP)
    if(_PP_IS_FIRST)
      set(_PP_STAGE_VOCAB_CMD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_PP_VOCAB_SRC}" "${_PP_STAGE_OUT_DIR}/${_PP_VOCAB_FILE_NAME}")
      set(_PP_VOCAB_DEP "${_PP_VOCAB_SRC}")
    endif()

    add_custom_command(
      OUTPUT "${_PP_RAX_OUT}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${_PP_STAGE_OUT_DIR}" "${_PP_OUT_DIR}"
      ${_PP_STAGE_SO_CMD}
      ${_PP_STAGE_RUNNER_CMD}
      ${_PP_STAGE_DEP_CMDS}
      ${_PP_STAGE_VOCAB_CMD}
      ${_PP_STAGE_WEIGHT_CMDS}
      COMMAND "${_PP_RAX_PACK_BIN}" "${_PP_MANIFEST}"
        -o "${_PP_RAX_TMP}" ${_PP_RAX_PACK_EXTRA_ARGS}
      COMMAND "${CMAKE_COMMAND}" -E rename
        "${_PP_RAX_TMP}" "${_PP_RAX_OUT}"
      DEPENDS
        ${_PP_RAX_PACK_DEP}
        "${_PP_MANIFEST}"
        ${_PP_SO_TARGET}
        "${_PP_STAGE_SO_FILE}"
        ${_PP_RUNNER_DEPS}
        ${_PP_WEIGHT_PATHS}
        ${_PP_DEP_LIB_PATHS}
        ${_PP_VOCAB_DEP}
      COMMENT "[${_PP_NAME}_pipeline] stage${_PP_RANK}: packing ${_PP_RAX_OUT}"
      VERBATIM
    )

    set(_PP_STAGE_TARGET "${_PP_NAME}_pipeline_stage_${_PP_RANK}_rax")
    add_custom_target(${_PP_STAGE_TARGET} DEPENDS "${_PP_RAX_OUT}")
    list(APPEND _PP_STAGE_TARGETS ${_PP_STAGE_TARGET})

    math(EXPR _PP_RANK "${_PP_RANK} + 1")
  endforeach()

  add_custom_target(${_PP_NAME}_pipeline_rax
    DEPENDS ${_PP_STAGE_TARGETS}
  )
  message(STATUS
    "[${_PP_NAME}] Registered ${_PP_NAME}_pipeline_rax for ${_PP_RANK} stage(s) "
    "(cross=${IS_RVV_CROSSCOMPILE}, embed=${_PP_EMBED_PAYLOAD}, out=${_PP_OUT_DIR}).")
endfunction()
