# ══════════════════════════════════════════════════════════════════════════════
# DeepSeek R1 — single build path (buddy-codegen)
#
# Variant JSON → gen_config + ModelSession + RHAL manifest + import_model +
# compile_pipeline + .so + .rax  (see tools/buddy-codegen/)
#
# Part 1 (shape / compile-time constants): values come from the chosen
# specs/<variant>.json via gen_config.py (not hand-edited CMake cache vars).
# Default --max-tokens / RunConfig::maxNewTokens should stay aligned with
# buddy::runtime::RunConfig in InferenceRunner.h and buddy-cli defaults.
#
# Part 2 (linking the static model library): buddy_model.cmake links
#   buddy_runtime_core + buddy_runtime_llm + CMAKE_DL_LIBS (-ldl) + LLVMSupport
# Part 3 (MLIR pipeline): Pipe is roughly
#   buddy-opt → mlir-opt (TOSA) → buddy-opt (bufferize + vectorize + lower) →
#   mlir-translate → llvm-as → llc → .o → link .so → gen_manifest → rax-pack → .rax
#   Mode A: pre-built .o | Mode B: pre-generated MLIR | Mode C: import_model.py
# Output .rax and vocab.txt live next to the .so so manifest URIs like
# file:model.so and file:vocab.txt resolve relative to the .rax directory.
# ══════════════════════════════════════════════════════════════════════════════

include(${CMAKE_SOURCE_DIR}/tools/buddy-codegen/cmake/buddy_model.cmake)
include(ProcessorCount)

ProcessorCount(_BUDDY_DSR1_DEFAULT_COMPILE_JOBS)
if(_BUDDY_DSR1_DEFAULT_COMPILE_JOBS EQUAL 0)
  set(_BUDDY_DSR1_DEFAULT_COMPILE_JOBS 1)
endif()

set(BUDDY_DSR1_SPEC
  "${CMAKE_CURRENT_SOURCE_DIR}/specs/f32.json"
  CACHE FILEPATH "Variant spec JSON, e.g. specs/w8a16.json")
set(BUDDY_DSR1_HF_CONFIG ""
  CACHE FILEPATH "Optional: HuggingFace config.json for gen_config.py (auto-detect if empty)")
set(BUDDY_DSR1_LOCAL_MODEL ""
  CACHE PATH "Optional: local HF model directory (weights import; sets DEEPSEEKR1_MODEL_PATH)")
set(BUDDY_DSR1_COMPILE_JOBS "${_BUDDY_DSR1_DEFAULT_COMPILE_JOBS}"
  CACHE STRING "Parallel MLIR→.o jobs when using compile_pipeline.py")
set(BUDDY_DSR1_TIERED_KV_CACHE ""
  CACHE STRING "Auto/ON/OFF: build DeepSeek R1 with tiered KV cache entrypoints")
set(BUDDY_DSR1_TIERED_CACHE_SIZES ""
  CACHE STRING "Semicolon-separated tiered KV cache sizes; auto-detected from spec if empty")

# Optional: skip PyTorch import (Mode B) or use pre-built .o (Mode A)
set(DEEPSEEKR1_BUILD_DIR "" CACHE PATH
  "Mode A: dir containing pre-built forward/subgraph .o files")
set(DEEPSEEKR1_MLIR_DIR "" CACHE PATH
  "Mode B: dir containing pre-generated forward_prefill.mlir etc.")

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c
          "import json,sys; s=json.load(open(sys.argv[1])); raw=s.get('tiered_kv_cache', False); enabled=False; sizes=[]\nif isinstance(raw, dict):\n  enabled=bool(raw.get('enabled', True)); sizes=raw.get('cache_sizes', s.get('cache_sizes', []))\nelse:\n  enabled=bool(raw); sizes=s.get('cache_sizes', [])\nif isinstance(sizes, str): sizes=[x.strip() for x in sizes.split(',') if x.strip()]\nif enabled and not sizes: sizes=[32,64,128,256,512,1024]\nprint(('ON' if enabled else 'OFF') + '|' + ','.join(str(x) for x in sizes))"
          "${BUDDY_DSR1_SPEC}"
  OUTPUT_VARIABLE _BUDDY_DSR1_TIERED_FROM_SPEC
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _BUDDY_DSR1_TIERED_PARSE_RESULT)

set(_BUDDY_DSR1_TIERED_ENABLED_FROM_SPEC OFF)
set(_BUDDY_DSR1_TIERED_SIZES_FROM_SPEC "")
if(_BUDDY_DSR1_TIERED_PARSE_RESULT EQUAL 0)
  string(REPLACE "|" ";" _BUDDY_DSR1_TIERED_FIELDS
         "${_BUDDY_DSR1_TIERED_FROM_SPEC}")
  list(GET _BUDDY_DSR1_TIERED_FIELDS 0 _BUDDY_DSR1_TIERED_ENABLED_FROM_SPEC)
  list(LENGTH _BUDDY_DSR1_TIERED_FIELDS _BUDDY_DSR1_TIERED_FIELD_COUNT)
  if(_BUDDY_DSR1_TIERED_FIELD_COUNT GREATER 1)
    list(GET _BUDDY_DSR1_TIERED_FIELDS 1 _BUDDY_DSR1_TIERED_SIZES_CSV)
    string(REPLACE "," ";" _BUDDY_DSR1_TIERED_SIZES_FROM_SPEC
           "${_BUDDY_DSR1_TIERED_SIZES_CSV}")
  endif()
endif()

if(BUDDY_DSR1_TIERED_KV_CACHE STREQUAL "")
  set(_BUDDY_DSR1_TIERED_ENABLED "${_BUDDY_DSR1_TIERED_ENABLED_FROM_SPEC}")
else()
  set(_BUDDY_DSR1_TIERED_ENABLED "${BUDDY_DSR1_TIERED_KV_CACHE}")
endif()

if(BUDDY_DSR1_TIERED_CACHE_SIZES STREQUAL "")
  set(_BUDDY_DSR1_TIERED_CACHE_SIZES
      "${_BUDDY_DSR1_TIERED_SIZES_FROM_SPEC}")
else()
  set(_BUDDY_DSR1_TIERED_CACHE_SIZES "${BUDDY_DSR1_TIERED_CACHE_SIZES}")
endif()

buddy_add_model(
  NAME deepseek_r1
  SPEC "${BUDDY_DSR1_SPEC}"
  RUNNER_SRC DeepSeekR1Runner.cpp
  RUNNER_PLUGIN_SRC DeepSeekR1RunnerPlugin.cpp
  MODEL_PLUGIN_SRC DeepSeekR1ResidentModelPlugin.cpp
  EXTRA_SRCS DeepSeekR1ResidentModel.cpp
  HF_CONFIG "${BUDDY_DSR1_HF_CONFIG}"
  LOCAL_MODEL "${BUDDY_DSR1_LOCAL_MODEL}"
  BUILD_DIR "${DEEPSEEKR1_BUILD_DIR}"
  MLIR_DIR "${DEEPSEEKR1_MLIR_DIR}"
  COMPILE_JOBS "${BUDDY_DSR1_COMPILE_JOBS}"
  TIERED_KV_CACHE "${_BUDDY_DSR1_TIERED_ENABLED}"
  TIERED_CACHE_SIZES "${_BUDDY_DSR1_TIERED_CACHE_SIZES}"
)

# Optional MPI pipeline-parallel runner plugin (skipped when MPI is unavailable).
find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND)
  add_library(buddy_models_deepseek_r1_pipeline_runner SHARED
    pipeline/PipelineRunner.cpp
    pipeline/PipelineRunnerPlugin.cpp
  )
  set_target_properties(buddy_models_deepseek_r1_pipeline_runner PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    OUTPUT_NAME "deepseek_r1_pipeline_runner"
    PREFIX ""
  )
  target_include_directories(buddy_models_deepseek_r1_pipeline_runner PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/pipeline"
    "${CMAKE_SOURCE_DIR}/frontend/Interfaces"
    "${CMAKE_BINARY_DIR}/frontend/Interfaces"
  )
  target_link_libraries(buddy_models_deepseek_r1_pipeline_runner PRIVATE
    buddy_runtime_core
    MPI::MPI_CXX
    ${CMAKE_DL_LIBS}
  )
  target_compile_features(buddy_models_deepseek_r1_pipeline_runner PRIVATE
    cxx_std_17
  )
else()
  message(STATUS
    "[deepseek_r1] MPI not found; skipping deepseek_r1_pipeline_runner plugin.")
endif()

# ══════════════════════════════════════════════════════════════════════════════
# Pipeline-parallel stage RAX targets (deepseek_r1_pipeline_rax)
#
# Enabled only when DEEPSEEKR1_PIPELINE_STAGE_DIRS is set (ordered list of stage
# directories corresponding to MPI ranks 0, 1, ...). Unset by default so normal
# whole-model builds are unaffected.
# ══════════════════════════════════════════════════════════════════════════════
set(DEEPSEEKR1_PIPELINE_STAGE_DIRS "" CACHE STRING
  "Ordered semicolon-separated list of pipeline stage directories (rank 0, 1, ...)")
option(DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD
  "Embed weights, stage.so, runner, deps, and vocab into stage<i>.rax (OFF = manifest-only)"
  ON)
set(DEEPSEEKR1_PIPELINE_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/pipeline_rax" CACHE PATH
  "Directory where published stage<i>.rax artifacts are placed")
set(DEEPSEEKR1_PIPELINE_RAX_PACK "" CACHE FILEPATH
  "Optional pre-built rax-pack executable (defaults to in-tree rax-pack target)")
set(DEEPSEEKR1_PIPELINE_VOCAB_FILE
  "${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt" CACHE FILEPATH
  "Fallback vocab.txt for the first pipeline stage when absent from its stage dir")
set(RISCV_MPI_DIR "" CACHE PATH
  "Prefix of target riscv64 MPI installation (include/mpi.h, lib/libmpi.so) for IS_RVV_CROSSCOMPILE")
set(DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY "" CACHE FILEPATH
  "Optional pre-built target deepseek_r1_pipeline_runner.so for IS_RVV_CROSSCOMPILE")

if(DEEPSEEKR1_PIPELINE_STAGE_DIRS)
  if(NOT IS_RVV_CROSSCOMPILE AND NOT MPI_CXX_FOUND)
    message(FATAL_ERROR
      "[deepseek_r1] DEEPSEEKR1_PIPELINE_STAGE_DIRS is set, but MPI (CXX) was not found. "
      "Building deepseek_r1_pipeline_rax requires MPI for deepseek_r1_pipeline_runner.so.")
  endif()

  set(_DSR1_PP_VERIFY_SCRIPT
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/verify_pipeline_stage.py")
  set(_DSR1_PP_GEN_MANIFEST_SCRIPT
    "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_manifest.py")
  foreach(_req_script "${_DSR1_PP_VERIFY_SCRIPT}" "${_DSR1_PP_GEN_MANIFEST_SCRIPT}")
    if(NOT EXISTS "${_req_script}")
      message(FATAL_ERROR
        "[deepseek_r1] Missing required pipeline script: ${_req_script}")
    endif()
  endforeach()

  if(DEEPSEEKR1_PIPELINE_RAX_PACK)
    if(NOT EXISTS "${DEEPSEEKR1_PIPELINE_RAX_PACK}")
      message(FATAL_ERROR
        "[deepseek_r1] DEEPSEEKR1_PIPELINE_RAX_PACK not found: ${DEEPSEEKR1_PIPELINE_RAX_PACK}")
    endif()
    set(_DSR1_PP_RAX_PACK_BIN "${DEEPSEEKR1_PIPELINE_RAX_PACK}")
    set(_DSR1_PP_RAX_PACK_DEP "${DEEPSEEKR1_PIPELINE_RAX_PACK}")
  else()
    set(_DSR1_PP_RAX_PACK_BIN "$<TARGET_FILE:rax-pack>")
    set(_DSR1_PP_RAX_PACK_DEP rax-pack)
  endif()

  # Resolve compiler, symbol checker, runner plugin, and MLIR/OpenMP runtime
  # library directories for either host or RISC-V cross-compilation.
  set(_DSR1_PP_RUNNER_NAME "deepseek_r1_pipeline_runner${CMAKE_SHARED_LIBRARY_SUFFIX}")
  set(_DSR1_PP_RUNNER_FILE "")
  set(_DSR1_PP_RUNNER_DEPS)
  set(_DSR1_PP_MLIR_RUNTIME_DIR "")
  set(_DSR1_PP_CRUNNER_CAND "")
  set(_DSR1_PP_OMP_LIB "")

  if(IS_RVV_CROSSCOMPILE)
    if(NOT RISCV_GNU_TOOLCHAIN)
      message(FATAL_ERROR
        "[deepseek_r1] IS_RVV_CROSSCOMPILE=ON requires RISCV_GNU_TOOLCHAIN for deepseek_r1_pipeline_rax.")
    endif()
    if(NOT RISCV_MLIR_C_RUNNER_UTILS OR NOT EXISTS "${RISCV_MLIR_C_RUNNER_UTILS}")
      message(FATAL_ERROR
        "[deepseek_r1] IS_RVV_CROSSCOMPILE=ON requires RISCV_MLIR_C_RUNNER_UTILS (got '${RISCV_MLIR_C_RUNNER_UTILS}').")
    endif()

    set(_DSR1_PP_CRUNNER_CAND "${RISCV_MLIR_C_RUNNER_UTILS}")
    get_filename_component(_DSR1_PP_MLIR_RUNTIME_DIR "${RISCV_MLIR_C_RUNNER_UTILS}" DIRECTORY)
    if(RISCV_OMP_SHARED AND EXISTS "${RISCV_OMP_SHARED}")
      set(_DSR1_PP_OMP_LIB "${RISCV_OMP_SHARED}")
    endif()

    # Locate cross-capable clang/clang++.
    set(_DSR1_PP_CROSS_CC "")
    set(_DSR1_PP_CROSS_CXX "")
    foreach(_clang_dir "${LLVM_TOOLS_BINARY_DIR}" "${BUDDY_MLIR_BUILD_DIR}/../llvm/build/bin")
      if(NOT _DSR1_PP_CROSS_CC AND EXISTS "${_clang_dir}/clang")
        set(_DSR1_PP_CROSS_CC "${_clang_dir}/clang")
      endif()
      if(NOT _DSR1_PP_CROSS_CXX AND EXISTS "${_clang_dir}/clang++")
        set(_DSR1_PP_CROSS_CXX "${_clang_dir}/clang++")
      endif()
    endforeach()
    if(NOT _DSR1_PP_CROSS_CC)
      set(_DSR1_PP_CROSS_CC "${CMAKE_C_COMPILER}")
    endif()
    if(NOT _DSR1_PP_CROSS_CXX)
      set(_DSR1_PP_CROSS_CXX "${CMAKE_CXX_COMPILER}")
    endif()

    find_program(_DSR1_PP_NM
      NAMES riscv64-linux-gnu-nm llvm-nm nm
      HINTS "${LLVM_TOOLS_BINARY_DIR}")
    if(NOT _DSR1_PP_NM)
      set(_DSR1_PP_NM "nm")
    endif()

    set(_DSR1_PP_RV_MARCH "rv64gcv")
    if(BUDDY_RISCV_ENABLE_ZFH_ZVFH)
      string(APPEND _DSR1_PP_RV_MARCH "_zfh_zvfh")
    endif()
    set(_DSR1_PP_RV_TARGET_OPTS
      --target=riscv64-unknown-linux-gnu
      "--sysroot=${RISCV_GNU_TOOLCHAIN}/sysroot"
      "--gcc-toolchain=${RISCV_GNU_TOOLCHAIN}")

    if(DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY)
      if(NOT EXISTS "${DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY}")
        message(FATAL_ERROR
          "[deepseek_r1] DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY not found: ${DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY}")
      endif()
      set(_DSR1_PP_RUNNER_FILE "${DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY}")
      set(_DSR1_PP_RUNNER_DEPS "${DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY}")
    elseif(RISCV_MPI_DIR)
      if(NOT EXISTS "${RISCV_MPI_DIR}/include/mpi.h" OR
         NOT EXISTS "${RISCV_MPI_DIR}/lib/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}")
        message(FATAL_ERROR
          "[deepseek_r1] RISCV_MPI_DIR='${RISCV_MPI_DIR}' must contain include/mpi.h and lib/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}.")
      endif()
      set(_DSR1_PP_RV_RUNNER_DIR "${CMAKE_CURRENT_BINARY_DIR}/pipeline_stages/rv64")
      set(_DSR1_PP_RV_FB_DIR "${_DSR1_PP_RV_RUNNER_DIR}/flatbuffers-include")
      file(MAKE_DIRECTORY "${_DSR1_PP_RV_FB_DIR}")
      file(CREATE_LINK "${FLATBUFFERS_INCLUDE_DIR}/flatbuffers"
           "${_DSR1_PP_RV_FB_DIR}/flatbuffers" SYMBOLIC)

      set(_DSR1_PP_RV_RUNNER_INCS
        "-I${CMAKE_CURRENT_SOURCE_DIR}/pipeline"
        "-I${CMAKE_SOURCE_DIR}/runtime/include"
        "-I${CMAKE_BINARY_DIR}/runtime/include"
        "-I${_DSR1_PP_RV_FB_DIR}"
        "-I${CMAKE_SOURCE_DIR}/frontend/Interfaces"
        "-I${CMAKE_BINARY_DIR}/frontend/Interfaces"
        "-I${RISCV_MPI_DIR}/include")
      set(_DSR1_PP_RV_RUNNER_OBJS)
      foreach(_rsrc pipeline/PipelineRunner.cpp pipeline/PipelineRunnerPlugin.cpp)
        get_filename_component(_rname "${_rsrc}" NAME_WE)
        set(_robj "${_DSR1_PP_RV_RUNNER_DIR}/${_rname}.o")
        add_custom_command(
          OUTPUT "${_robj}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DSR1_PP_RV_RUNNER_DIR}"
          COMMAND "${_DSR1_PP_CROSS_CXX}" ${_DSR1_PP_RV_TARGET_OPTS}
            -std=c++17 -O2 -DNDEBUG -fPIC "-march=${_DSR1_PP_RV_MARCH}"
            ${_DSR1_PP_RV_RUNNER_INCS}
            -MD -MF "${_robj}.d"
            -c "${CMAKE_CURRENT_SOURCE_DIR}/${_rsrc}" -o "${_robj}"
          DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${_rsrc}" buddy-rax-gen
          DEPFILE "${_robj}.d"
          COMMENT "[deepseek_r1_pipeline] riscv64: compiling ${_rname}.cpp"
          VERBATIM
        )
        list(APPEND _DSR1_PP_RV_RUNNER_OBJS "${_robj}")
      endforeach()
      set(_DSR1_PP_RV_RUNNER_SO "${_DSR1_PP_RV_RUNNER_DIR}/${_DSR1_PP_RUNNER_NAME}")
      add_custom_command(
        OUTPUT "${_DSR1_PP_RV_RUNNER_SO}"
        COMMAND "${_DSR1_PP_CROSS_CXX}" ${_DSR1_PP_RV_TARGET_OPTS}
          -shared -fPIC
          "-Wl,-soname,${_DSR1_PP_RUNNER_NAME}"
          "-Wl,-rpath,\$ORIGIN"
          "-Wl,-z,defs"
          -o "${_DSR1_PP_RV_RUNNER_SO}"
          ${_DSR1_PP_RV_RUNNER_OBJS}
          "-L${RISCV_MPI_DIR}/lib" -lmpi -ldl -lpthread
        DEPENDS ${_DSR1_PP_RV_RUNNER_OBJS}
        COMMENT "[deepseek_r1_pipeline] riscv64: linking ${_DSR1_PP_RUNNER_NAME}"
        VERBATIM
      )
      add_custom_target(deepseek_r1_pipeline_runner_rv64
        DEPENDS "${_DSR1_PP_RV_RUNNER_SO}"
      )
      set(_DSR1_PP_RUNNER_FILE "${_DSR1_PP_RV_RUNNER_SO}")
      set(_DSR1_PP_RUNNER_DEPS deepseek_r1_pipeline_runner_rv64 "${_DSR1_PP_RV_RUNNER_SO}")
    elseif(DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD)
      message(FATAL_ERROR
        "[deepseek_r1] IS_RVV_CROSSCOMPILE=ON with DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD=ON requires "
        "RISCV_MPI_DIR (to cross-compile deepseek_r1_pipeline_runner.so) or "
        "DEEPSEEKR1_PIPELINE_RUNNER_LIBRARY (or set DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD=OFF).")
    else()
      message(STATUS
        "[deepseek_r1] IS_RVV_CROSSCOMPILE=ON without RISCV_MPI_DIR: packing manifest-only "
        "stage .rax without copying deepseek_r1_pipeline_runner.so.")
    endif()
  else()
    set(_DSR1_PP_NM "nm")
    set(_DSR1_PP_RUNNER_NAME "$<TARGET_FILE_NAME:buddy_models_deepseek_r1_pipeline_runner>")
    set(_DSR1_PP_RUNNER_FILE "$<TARGET_FILE:buddy_models_deepseek_r1_pipeline_runner>")
    set(_DSR1_PP_RUNNER_DEPS
      buddy_models_deepseek_r1_pipeline_runner
      "$<TARGET_FILE:buddy_models_deepseek_r1_pipeline_runner>")

    set(_DSR1_PP_MLIR_RUNTIME_DIR "${LLVM_LIBRARY_DIR}")
    set(_DSR1_PP_CRUNNER_CAND
      "${LLVM_LIBRARY_DIR}/libmlir_c_runner_utils${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_DSR1_PP_CRUNNER_CAND}")
      message(FATAL_ERROR
        "[deepseek_r1] Missing required MLIR runtime library: ${_DSR1_PP_CRUNNER_CAND}")
    endif()

    if(BUDDY_OPENMP_RUNTIME_LIBRARY AND EXISTS "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
      set(_DSR1_PP_OMP_LIB "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
    else()
      get_filename_component(_DSR1_PP_LLVM_ROOT "${LLVM_LIBRARY_DIR}" DIRECTORY)
      foreach(_omp_cand
        "${_DSR1_PP_LLVM_ROOT}/runtimes/runtimes-bins/openmp/runtime/src/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${LLVM_LIBRARY_DIR}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
      )
        if(EXISTS "${_omp_cand}")
          set(_DSR1_PP_OMP_LIB "${_omp_cand}")
          break()
        endif()
      endforeach()
    endif()
    if(NOT _DSR1_PP_OMP_LIB)
      find_library(_DSR1_PP_OMP_LIB NAMES omp)
    endif()
  endif()

  file(REAL_PATH "${_DSR1_PP_CRUNNER_CAND}" _DSR1_PP_CRUNNER_REAL)
  get_filename_component(_DSR1_PP_CRUNNER_NAME "${_DSR1_PP_CRUNNER_REAL}" NAME)

  set(_DSR1_PP_OMP_REAL "")
  set(_DSR1_PP_OMP_NAME "")
  if(_DSR1_PP_OMP_LIB AND EXISTS "${_DSR1_PP_OMP_LIB}")
    file(REAL_PATH "${_DSR1_PP_OMP_LIB}" _DSR1_PP_OMP_REAL)
    get_filename_component(_DSR1_PP_OMP_NAME "${_DSR1_PP_OMP_REAL}" NAME)
  endif()

  # Base MLIR runtime shared library dependencies (in load order).
  set(_DSR1_PP_BASE_DEP_PATHS)
  set(_DSR1_PP_BASE_DEP_NAMES)
  foreach(_mlir_dep_base libmlir_apfloat_wrappers libmlir_float16_utils)
    set(_mlir_dep_cand
      "${_DSR1_PP_MLIR_RUNTIME_DIR}/${_mlir_dep_base}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_mlir_dep_cand}")
      file(GLOB _mlir_dep_globs
        "${_DSR1_PP_MLIR_RUNTIME_DIR}/${_mlir_dep_base}${CMAKE_SHARED_LIBRARY_SUFFIX}.*")
      if(_mlir_dep_globs)
        list(GET _mlir_dep_globs 0 _mlir_dep_cand)
      endif()
    endif()
    if(EXISTS "${_mlir_dep_cand}")
      file(REAL_PATH "${_mlir_dep_cand}" _mlir_dep_real)
      get_filename_component(_mlir_dep_name "${_mlir_dep_real}" NAME)
      list(APPEND _DSR1_PP_BASE_DEP_PATHS "${_mlir_dep_real}")
      list(APPEND _DSR1_PP_BASE_DEP_NAMES "${_mlir_dep_name}")
    endif()
  endforeach()
  list(APPEND _DSR1_PP_BASE_DEP_PATHS "${_DSR1_PP_CRUNNER_REAL}")
  list(APPEND _DSR1_PP_BASE_DEP_NAMES "${_DSR1_PP_CRUNNER_NAME}")

  get_filename_component(_DSR1_PP_OUT_DIR "${DEEPSEEKR1_PIPELINE_OUTPUT_DIR}" ABSOLUTE)

  set(_DSR1_PP_SYMCHECK_PY
    "import os, subprocess, sys; so = sys.argv[1]; nm_bin = sys.argv[2] if len(sys.argv) > 2 else 'nm'; req = {'buddy_create_pipeline_stage_v1', 'buddy_create_pipeline_stage_v2', 'buddy_destroy_pipeline_stage_v1', '_mlir_ciface_forward_prefill', '_mlir_ciface_forward_decode'}; out = subprocess.check_output([nm_bin, '-D', '--defined-only', so], text=True); defs = {l.split()[-1].split('@')[0] for l in out.splitlines() if l.strip()}; miss = sorted(req - defs); miss and (os.remove(so), sys.exit(f'stage.so missing required symbols: {miss}'))")
  # Hardlink large weight blobs into staging (fall back to copy across filesystems).
  # Avoid semicolons and literal newlines so the snippet survives CMake list(APPEND) and Ninja.
  set(_DSR1_PP_HARDLINK_PY
    "exec('import os, shutil, sys\\ns, d = sys.argv[1], sys.argv[2]\\nif os.path.lexists(d):\\n  try:\\n    if not os.path.islink(d) and os.path.samefile(s, d): sys.exit(0)\\n  except OSError: pass\\n  os.unlink(d)\\ntry: os.link(s, d)\\nexcept OSError: shutil.copy2(s, d)')")

  set(_DSR1_PP_STAGE_TARGETS)
  set(_DSR1_PP_RANK 0)
  foreach(_DSR1_PP_RAW_DIR IN LISTS DEEPSEEKR1_PIPELINE_STAGE_DIRS)
    get_filename_component(_DSR1_PP_STAGE_DIR "${_DSR1_PP_RAW_DIR}" ABSOLUTE)
    if(NOT IS_DIRECTORY "${_DSR1_PP_STAGE_DIR}")
      message(FATAL_ERROR
        "[deepseek_r1] stage ${_DSR1_PP_RANK}: stage directory does not exist: ${_DSR1_PP_STAGE_DIR}")
    endif()

    set(_DSR1_PP_CONFIG "${_DSR1_PP_STAGE_DIR}/config.json")
    if(NOT EXISTS "${_DSR1_PP_CONFIG}")
      message(FATAL_ERROR
        "[deepseek_r1] stage ${_DSR1_PP_RANK}: missing required input file: ${_DSR1_PP_CONFIG}")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_DSR1_PP_CONFIG}")

    file(READ "${_DSR1_PP_CONFIG}" _DSR1_PP_CFG_JSON)
    string(JSON _DSR1_PP_START_LAYER ERROR_VARIABLE _DSR1_PP_JSON_ERR
      GET "${_DSR1_PP_CFG_JSON}" pipeline_stage start_layer)
    if(_DSR1_PP_JSON_ERR)
      message(FATAL_ERROR
        "[deepseek_r1] stage ${_DSR1_PP_RANK}: failed to read pipeline_stage.start_layer from ${_DSR1_PP_CONFIG}: ${_DSR1_PP_JSON_ERR}")
    endif()
    string(JSON _DSR1_PP_VOCAB_FILE ERROR_VARIABLE _DSR1_PP_JSON_ERR
      GET "${_DSR1_PP_CFG_JSON}" tokens vocab_file)
    if(_DSR1_PP_JSON_ERR)
      message(FATAL_ERROR
        "[deepseek_r1] stage ${_DSR1_PP_RANK}: failed to read tokens.vocab_file from ${_DSR1_PP_CONFIG}: ${_DSR1_PP_JSON_ERR}")
    endif()

    string(JSON _DSR1_PP_VARIANT ERROR_VARIABLE _DSR1_PP_VAR_ERR
      GET "${_DSR1_PP_CFG_JSON}" variant)
    if(_DSR1_PP_VAR_ERR)
      set(_DSR1_PP_VARIANT "f16")
    endif()
    set(_DSR1_PP_IS_W4G32 FALSE)
    if(_DSR1_PP_VARIANT STREQUAL "w4g32")
      set(_DSR1_PP_IS_W4G32 TRUE)
    endif()

    set(_DSR1_PP_USE_ARENA FALSE)
    string(JSON _DSR1_PP_ARENA_VAL ERROR_VARIABLE _DSR1_PP_OPT_ERR
      GET "${_DSR1_PP_CFG_JSON}" arena)
    if(NOT _DSR1_PP_OPT_ERR AND _DSR1_PP_ARENA_VAL)
      set(_DSR1_PP_USE_ARENA TRUE)
    endif()

    set(_DSR1_PP_USE_THREAD_POOL FALSE)
    string(JSON _DSR1_PP_TP_VAL ERROR_VARIABLE _DSR1_PP_OPT_ERR
      GET "${_DSR1_PP_CFG_JSON}" thread_pool)
    if(NOT _DSR1_PP_OPT_ERR AND _DSR1_PP_TP_VAL)
      set(_DSR1_PP_USE_THREAD_POOL TRUE)
    endif()

    set(_DSR1_PP_USE_PREFILL_IME FALSE)
    string(JSON _DSR1_PP_IME_VAL ERROR_VARIABLE _DSR1_PP_OPT_ERR
      GET "${_DSR1_PP_CFG_JSON}" prefill_ime)
    if(NOT _DSR1_PP_OPT_ERR AND _DSR1_PP_IME_VAL)
      set(_DSR1_PP_USE_PREFILL_IME TRUE)
    endif()

    set(_DSR1_PP_ARENA_PREFAULT_MB 0)
    string(JSON _DSR1_PP_PREFAULT_VAL ERROR_VARIABLE _DSR1_PP_OPT_ERR
      GET "${_DSR1_PP_CFG_JSON}" arena_prefault_mb)
    if(NOT _DSR1_PP_OPT_ERR AND _DSR1_PP_PREFAULT_VAL)
      set(_DSR1_PP_ARENA_PREFAULT_MB "${_DSR1_PP_PREFAULT_VAL}")
    endif()

    string(JSON _DSR1_PP_WEIGHT_COUNT ERROR_VARIABLE _DSR1_PP_JSON_ERR
      LENGTH "${_DSR1_PP_CFG_JSON}" weights)
    if(_DSR1_PP_JSON_ERR OR _DSR1_PP_WEIGHT_COUNT LESS 1)
      message(FATAL_ERROR
        "[deepseek_r1] stage ${_DSR1_PP_RANK}: failed to read weights array from ${_DSR1_PP_CONFIG}: ${_DSR1_PP_JSON_ERR}")
    endif()
    set(_DSR1_PP_WEIGHT_FILES)
    set(_DSR1_PP_WEIGHT_PATHS)
    math(EXPR _DSR1_PP_WEIGHT_LAST "${_DSR1_PP_WEIGHT_COUNT} - 1")
    foreach(_widx RANGE 0 ${_DSR1_PP_WEIGHT_LAST})
      string(JSON _wfile ERROR_VARIABLE _DSR1_PP_JSON_ERR
        GET "${_DSR1_PP_CFG_JSON}" weights ${_widx} file)
      if(_DSR1_PP_JSON_ERR)
        message(FATAL_ERROR
          "[deepseek_r1] stage ${_DSR1_PP_RANK}: failed to read weights[${_widx}].file from ${_DSR1_PP_CONFIG}: ${_DSR1_PP_JSON_ERR}")
      endif()
      list(APPEND _DSR1_PP_WEIGHT_FILES "${_wfile}")
      list(APPEND _DSR1_PP_WEIGHT_PATHS "${_DSR1_PP_STAGE_DIR}/${_wfile}")
    endforeach()

    set(_DSR1_PP_IS_FIRST FALSE)
    if(_DSR1_PP_START_LAYER EQUAL 0)
      set(_DSR1_PP_IS_FIRST TRUE)
    endif()

    if(_DSR1_PP_IS_W4G32)
      set(_DSR1_PP_MLIR_SUFFIX "-w4g32")
    else()
      set(_DSR1_PP_MLIR_SUFFIX "-f16")
    endif()

    set(_DSR1_PP_PREFILL_MLIR "${_DSR1_PP_STAGE_DIR}/forward_prefill${_DSR1_PP_MLIR_SUFFIX}.mlir")
    set(_DSR1_PP_DECODE_MLIR "${_DSR1_PP_STAGE_DIR}/forward_decode${_DSR1_PP_MLIR_SUFFIX}.mlir")
    set(_DSR1_PP_PREFILL_O "${_DSR1_PP_STAGE_DIR}/forward_prefill.o")
    set(_DSR1_PP_DECODE_O "${_DSR1_PP_STAGE_DIR}/forward_decode.o")
    set(_DSR1_PP_SUB_PREFILL_O "${_DSR1_PP_STAGE_DIR}/subgraph_prefill.o")
    set(_DSR1_PP_SUB_DECODE_O "${_DSR1_PP_STAGE_DIR}/subgraph_decode.o")

    set(_DSR1_PP_VERIFY_DEPS
      "${_DSR1_PP_VERIFY_SCRIPT}"
      "${_DSR1_PP_CONFIG}"
      "${_DSR1_PP_PREFILL_MLIR}"
      "${_DSR1_PP_DECODE_MLIR}"
      ${_DSR1_PP_WEIGHT_PATHS}
    )
    set(_DSR1_PP_OBJS
      "${_DSR1_PP_PREFILL_O}"
      "${_DSR1_PP_DECODE_O}"
      "${_DSR1_PP_SUB_PREFILL_O}"
      "${_DSR1_PP_SUB_DECODE_O}"
    )
    set(_DSR1_PP_REQ_FILES
      "${_DSR1_PP_PREFILL_MLIR}"
      "${_DSR1_PP_DECODE_MLIR}"
      ${_DSR1_PP_OBJS}
      ${_DSR1_PP_WEIGHT_PATHS}
    )
    if(_DSR1_PP_IS_W4G32)
      set(_DSR1_PP_K3_MLIR "${_DSR1_PP_STAGE_DIR}/k3_kernels-w4g32.mlir")
      set(_DSR1_PP_K3_O "${_DSR1_PP_STAGE_DIR}/k3_kernels.o")
      list(APPEND _DSR1_PP_VERIFY_DEPS "${_DSR1_PP_K3_MLIR}")
      list(APPEND _DSR1_PP_OBJS "${_DSR1_PP_K3_O}")
      list(APPEND _DSR1_PP_REQ_FILES "${_DSR1_PP_K3_MLIR}" "${_DSR1_PP_K3_O}")
      if(_DSR1_PP_USE_PREFILL_IME)
        set(_DSR1_PP_K3_IME_MLIR "${_DSR1_PP_STAGE_DIR}/k3_kernels_ime-w4g32.mlir")
        set(_DSR1_PP_K3_IME_O "${_DSR1_PP_STAGE_DIR}/k3_kernels_ime.o")
        list(APPEND _DSR1_PP_VERIFY_DEPS "${_DSR1_PP_K3_IME_MLIR}")
        list(APPEND _DSR1_PP_OBJS "${_DSR1_PP_K3_IME_O}")
        list(APPEND _DSR1_PP_REQ_FILES "${_DSR1_PP_K3_IME_MLIR}" "${_DSR1_PP_K3_IME_O}")
      endif()
    endif()

    foreach(_req_file IN LISTS _DSR1_PP_REQ_FILES)
      if(NOT EXISTS "${_req_file}")
        message(FATAL_ERROR
          "[deepseek_r1] stage ${_DSR1_PP_RANK}: missing required input file: ${_req_file}")
      endif()
    endforeach()

    # Per-stage runtime shared libraries (skip libomp when thread_pool=true).
    set(_DSR1_PP_DEP_LIB_PATHS ${_DSR1_PP_BASE_DEP_PATHS})
    set(_DSR1_PP_DEP_LIB_NAMES ${_DSR1_PP_BASE_DEP_NAMES})
    if(NOT _DSR1_PP_USE_THREAD_POOL)
      if(NOT _DSR1_PP_OMP_REAL)
        message(FATAL_ERROR
          "[deepseek_r1] stage ${_DSR1_PP_RANK}: OpenMP runtime library required when thread_pool=false "
          "(set BUDDY_OPENMP_RUNTIME_LIBRARY or RISCV_OMP_SHARED).")
      endif()
      list(APPEND _DSR1_PP_DEP_LIB_PATHS "${_DSR1_PP_OMP_REAL}")
      list(APPEND _DSR1_PP_DEP_LIB_NAMES "${_DSR1_PP_OMP_NAME}")
    endif()

    set(_DSR1_PP_DEP_MANIFEST_ARGS)
    foreach(_dep_name IN LISTS _DSR1_PP_DEP_LIB_NAMES)
      list(APPEND _DSR1_PP_DEP_MANIFEST_ARGS --dep-shared-lib "${_dep_name}")
    endforeach()
    list(LENGTH _DSR1_PP_DEP_LIB_PATHS _DSR1_PP_DEP_COUNT)
    math(EXPR _DSR1_PP_DEP_LAST "${_DSR1_PP_DEP_COUNT} - 1")

    set(_DSR1_PP_VOCAB_SRC "")
    if(_DSR1_PP_IS_FIRST)
      if(EXISTS "${_DSR1_PP_STAGE_DIR}/${_DSR1_PP_VOCAB_FILE}")
        set(_DSR1_PP_VOCAB_SRC "${_DSR1_PP_STAGE_DIR}/${_DSR1_PP_VOCAB_FILE}")
      else()
        set(_DSR1_PP_VOCAB_SRC "${DEEPSEEKR1_PIPELINE_VOCAB_FILE}")
      endif()
      if(NOT EXISTS "${_DSR1_PP_VOCAB_SRC}")
        message(FATAL_ERROR
          "[deepseek_r1] stage ${_DSR1_PP_RANK}: missing required vocab file: ${_DSR1_PP_VOCAB_SRC}")
      endif()
    endif()

    set(_DSR1_PP_WORK_DIR
      "${CMAKE_CURRENT_BINARY_DIR}/pipeline_stages/stage${_DSR1_PP_RANK}")
    set(_DSR1_PP_CONFIG_INC "${_DSR1_PP_WORK_DIR}/stage-config.inc")
    set(_DSR1_PP_MANIFEST "${_DSR1_PP_WORK_DIR}/stage.mlir")

    # 1. Verify stage MLIR ABI & weights, emit stage-config.inc.
    add_custom_command(
      OUTPUT "${_DSR1_PP_CONFIG_INC}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DSR1_PP_WORK_DIR}"
      COMMAND "${Python3_EXECUTABLE}" "${_DSR1_PP_VERIFY_SCRIPT}"
        --stage-dir "${_DSR1_PP_STAGE_DIR}"
        --header "${_DSR1_PP_CONFIG_INC}"
      DEPENDS ${_DSR1_PP_VERIFY_DEPS}
      COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK}: verifying ABI and generating stage-config.inc"
      VERBATIM
    )

    # Runtime C support sources (arena / thread_pool / SpacemiT TCM).
    set(_DSR1_PP_RT_C_SRCS)
    if(_DSR1_PP_USE_ARENA)
      list(APPEND _DSR1_PP_RT_C_SRCS "${CMAKE_SOURCE_DIR}/runtime/arena/BuddyArena.c")
    endif()
    if(_DSR1_PP_USE_PREFILL_IME)
      list(APPEND _DSR1_PP_RT_C_SRCS "${CMAKE_SOURCE_DIR}/runtime/spacemit/BuddySpacemitTcm.c")
    endif()
    if(_DSR1_PP_USE_THREAD_POOL)
      list(APPEND _DSR1_PP_RT_C_SRCS "${CMAKE_SOURCE_DIR}/runtime/threadpool/BuddyThreadPool.c")
    endif()

    # 2. Compile Stage.cpp (+ runtime support) and link stage.so with symbol closure & export check.
    set(_DSR1_PP_SO_TARGET "deepseek_r1_pipeline_stage_${_DSR1_PP_RANK}_so")
    if(IS_RVV_CROSSCOMPILE)
      set(_DSR1_PP_STAGE_SO_FILE "${_DSR1_PP_WORK_DIR}/stage.so")
      set(_DSR1_PP_RT_C_OBJS)
      foreach(_csrc IN LISTS _DSR1_PP_RT_C_SRCS)
        get_filename_component(_cname "${_csrc}" NAME_WE)
        set(_cobj "${_DSR1_PP_WORK_DIR}/${_cname}.o")
        set(_copts ${_DSR1_PP_RV_TARGET_OPTS} -O2 -fPIC)
        if(_cname STREQUAL "BuddyArena" AND _DSR1_PP_ARENA_PREFAULT_MB GREATER 0)
          list(APPEND _copts "-DBUDDY_ARENA_DEFAULT_PREFAULT_MB=${_DSR1_PP_ARENA_PREFAULT_MB}")
        endif()
        add_custom_command(
          OUTPUT "${_cobj}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DSR1_PP_WORK_DIR}"
          COMMAND "${_DSR1_PP_CROSS_CC}" ${_copts} -c "${_csrc}" -o "${_cobj}"
          DEPENDS "${_csrc}" "${_DSR1_PP_CONFIG}"
          COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK} riscv64: compiling ${_cname}.c"
          VERBATIM
        )
        list(APPEND _DSR1_PP_RT_C_OBJS "${_cobj}")
      endforeach()

      set(_DSR1_PP_MLIR_INC_ARGS)
      foreach(_inc IN LISTS MLIR_INCLUDE_DIRS)
        if(NOT _inc MATCHES "^/usr(/local)?/include/?$")
          list(APPEND _DSR1_PP_MLIR_INC_ARGS "-I${_inc}")
        endif()
      endforeach()

      set(_DSR1_PP_STAGE_OBJ "${_DSR1_PP_WORK_DIR}/Stage.o")
      add_custom_command(
        OUTPUT "${_DSR1_PP_STAGE_OBJ}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DSR1_PP_WORK_DIR}"
        COMMAND "${_DSR1_PP_CROSS_CXX}" ${_DSR1_PP_RV_TARGET_OPTS}
          -std=c++17 -O2 -DNDEBUG -fPIC "-march=${_DSR1_PP_RV_MARCH}"
          "-I${_DSR1_PP_WORK_DIR}"
          ${_DSR1_PP_MLIR_INC_ARGS}
          "-I${CMAKE_SOURCE_DIR}/runtime/include"
          -c "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/Stage.cpp"
          -o "${_DSR1_PP_STAGE_OBJ}"
        DEPENDS
          "${CMAKE_CURRENT_SOURCE_DIR}/pipeline/Stage.cpp"
          "${_DSR1_PP_CONFIG_INC}"
        COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK} riscv64: compiling Stage.cpp"
        VERBATIM
      )

      set(_DSR1_PP_RV_LINK_LIBS "${_DSR1_PP_CRUNNER_REAL}" -lm -lpthread -ldl)
      if(NOT _DSR1_PP_USE_THREAD_POOL)
        list(APPEND _DSR1_PP_RV_LINK_LIBS "${_DSR1_PP_OMP_REAL}")
      endif()

      add_custom_command(
        OUTPUT "${_DSR1_PP_STAGE_SO_FILE}"
        COMMAND "${_DSR1_PP_CROSS_CXX}" ${_DSR1_PP_RV_TARGET_OPTS}
          -shared -fPIC
          -Wl,-soname,stage.so
          -Wl,--allow-multiple-definition
          -Wl,-z,defs
          "-Wl,-rpath,\$ORIGIN"
          -o "${_DSR1_PP_STAGE_SO_FILE}"
          "${_DSR1_PP_STAGE_OBJ}"
          ${_DSR1_PP_OBJS}
          ${_DSR1_PP_RT_C_OBJS}
          ${_DSR1_PP_RV_LINK_LIBS}
        COMMAND "${Python3_EXECUTABLE}" -c "${_DSR1_PP_SYMCHECK_PY}"
          "${_DSR1_PP_STAGE_SO_FILE}" "${_DSR1_PP_NM}"
        DEPENDS
          "${_DSR1_PP_STAGE_OBJ}"
          ${_DSR1_PP_OBJS}
          ${_DSR1_PP_RT_C_OBJS}
        COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK} riscv64: linking and verifying stage.so"
        VERBATIM
      )
      add_custom_target(${_DSR1_PP_SO_TARGET} DEPENDS "${_DSR1_PP_STAGE_SO_FILE}")
    else()
      set(_DSR1_PP_STAGE_SO_FILE "$<TARGET_FILE:${_DSR1_PP_SO_TARGET}>")
      set_source_files_properties(${_DSR1_PP_OBJS} PROPERTIES
        EXTERNAL_OBJECT TRUE
      )
      add_library(${_DSR1_PP_SO_TARGET} SHARED EXCLUDE_FROM_ALL
        pipeline/Stage.cpp
        "${_DSR1_PP_CONFIG_INC}"
        ${_DSR1_PP_RT_C_SRCS}
        ${_DSR1_PP_OBJS}
      )
      set_target_properties(${_DSR1_PP_SO_TARGET} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_DSR1_PP_WORK_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${_DSR1_PP_WORK_DIR}"
        OUTPUT_NAME "stage"
        PREFIX ""
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "\$ORIGIN"
      )
      target_include_directories(${_DSR1_PP_SO_TARGET} PRIVATE
        "${_DSR1_PP_WORK_DIR}"
        ${MLIR_INCLUDE_DIRS}
        "${CMAKE_SOURCE_DIR}/runtime/include"
      )
      if(_DSR1_PP_USE_ARENA AND _DSR1_PP_ARENA_PREFAULT_MB GREATER 0)
        target_compile_definitions(${_DSR1_PP_SO_TARGET} PRIVATE
          "BUDDY_ARENA_DEFAULT_PREFAULT_MB=${_DSR1_PP_ARENA_PREFAULT_MB}")
      endif()
      set(_DSR1_PP_HOST_LINK_LIBS "${_DSR1_PP_CRUNNER_REAL}")
      if(_DSR1_PP_USE_THREAD_POOL)
        list(APPEND _DSR1_PP_HOST_LINK_LIBS -lpthread ${CMAKE_DL_LIBS})
      else()
        list(APPEND _DSR1_PP_HOST_LINK_LIBS "${_DSR1_PP_OMP_REAL}")
      endif()
      target_link_libraries(${_DSR1_PP_SO_TARGET} PRIVATE ${_DSR1_PP_HOST_LINK_LIBS})
      if(NOT APPLE)
        target_link_options(${_DSR1_PP_SO_TARGET} PRIVATE "-Wl,-z,defs")
        if(_DSR1_PP_IS_W4G32)
          target_link_options(${_DSR1_PP_SO_TARGET} PRIVATE
            "-Wl,--allow-multiple-definition")
        endif()
      endif()
      target_compile_features(${_DSR1_PP_SO_TARGET} PRIVATE cxx_std_17)
      add_custom_command(TARGET ${_DSR1_PP_SO_TARGET} POST_BUILD
        COMMAND "${Python3_EXECUTABLE}" -c "${_DSR1_PP_SYMCHECK_PY}"
          "$<TARGET_FILE:${_DSR1_PP_SO_TARGET}>" "${_DSR1_PP_NM}"
        COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK}: verifying stage.so factory and MLIR entry symbols"
        VERBATIM
      )
    endif()

    # 3. Generate per-stage RHAL manifest (stage.mlir).
    set(_DSR1_PP_VERSION_ARGS)
    if(DEFINED BUDDY_VERSION AND NOT "${BUDDY_VERSION}" STREQUAL "")
      set(_DSR1_PP_VERSION_ARGS --version "${BUDDY_VERSION}")
    endif()
    add_custom_command(
      OUTPUT "${_DSR1_PP_MANIFEST}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory "${_DSR1_PP_WORK_DIR}"
      COMMAND "${Python3_EXECUTABLE}" "${_DSR1_PP_GEN_MANIFEST_SCRIPT}"
        --config "${_DSR1_PP_CONFIG}"
        ${_DSR1_PP_DEP_MANIFEST_ARGS}
        --runner-library "${_DSR1_PP_RUNNER_NAME}"
        ${_DSR1_PP_VERSION_ARGS}
        -o "${_DSR1_PP_MANIFEST}"
      DEPENDS
        "${_DSR1_PP_GEN_MANIFEST_SCRIPT}"
        "${_DSR1_PP_CONFIG}"
      COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK}: generating RHAL manifest"
      VERBATIM
    )

    # 4. Stage referenced files, run rax-pack, and publish stage<i>.rax.
    if(DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD)
      set(_DSR1_PP_STAGE_OUT_DIR "${_DSR1_PP_WORK_DIR}")
      set(_DSR1_PP_RAX_TMP "${_DSR1_PP_WORK_DIR}/stage${_DSR1_PP_RANK}.rax.tmp")
      set(_DSR1_PP_RAX_OUT "${_DSR1_PP_OUT_DIR}/stage${_DSR1_PP_RANK}.rax")
      set(_DSR1_PP_RAX_PACK_EXTRA_ARGS --embed-payload)
      set(_DSR1_PP_STAGE_SO_CMD)
    else()
      set(_DSR1_PP_STAGE_OUT_DIR "${_DSR1_PP_OUT_DIR}/stage${_DSR1_PP_RANK}")
      set(_DSR1_PP_RAX_TMP "${_DSR1_PP_STAGE_OUT_DIR}/stage${_DSR1_PP_RANK}.rax.tmp")
      set(_DSR1_PP_RAX_OUT "${_DSR1_PP_STAGE_OUT_DIR}/stage${_DSR1_PP_RANK}.rax")
      set(_DSR1_PP_RAX_PACK_EXTRA_ARGS)
      set(_DSR1_PP_STAGE_SO_CMD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_DSR1_PP_STAGE_SO_FILE}"
          "${_DSR1_PP_STAGE_OUT_DIR}/stage.so")
    endif()

    set(_DSR1_PP_STAGE_RUNNER_CMD)
    if(_DSR1_PP_RUNNER_FILE)
      set(_DSR1_PP_STAGE_RUNNER_CMD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_DSR1_PP_RUNNER_FILE}"
          "${_DSR1_PP_STAGE_OUT_DIR}/${_DSR1_PP_RUNNER_NAME}")
    endif()

    set(_DSR1_PP_STAGE_DEP_CMDS)
    foreach(_dep_idx RANGE 0 ${_DSR1_PP_DEP_LAST})
      list(GET _DSR1_PP_DEP_LIB_PATHS ${_dep_idx} _dep_path)
      list(GET _DSR1_PP_DEP_LIB_NAMES ${_dep_idx} _dep_name)
      list(APPEND _DSR1_PP_STAGE_DEP_CMDS
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_dep_path}" "${_DSR1_PP_STAGE_OUT_DIR}/${_dep_name}")
    endforeach()

    set(_DSR1_PP_STAGE_WEIGHT_CMDS)
    foreach(_widx RANGE 0 ${_DSR1_PP_WEIGHT_LAST})
      list(GET _DSR1_PP_WEIGHT_PATHS ${_widx} _wpath)
      list(GET _DSR1_PP_WEIGHT_FILES ${_widx} _wfile)
      list(APPEND _DSR1_PP_STAGE_WEIGHT_CMDS
        COMMAND "${Python3_EXECUTABLE}" -c "${_DSR1_PP_HARDLINK_PY}"
          "${_wpath}" "${_DSR1_PP_STAGE_OUT_DIR}/${_wfile}")
    endforeach()

    set(_DSR1_PP_STAGE_VOCAB_CMD)
    set(_DSR1_PP_VOCAB_DEP)
    if(_DSR1_PP_IS_FIRST)
      set(_DSR1_PP_STAGE_VOCAB_CMD
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "${_DSR1_PP_VOCAB_SRC}" "${_DSR1_PP_STAGE_OUT_DIR}/${_DSR1_PP_VOCAB_FILE}")
      set(_DSR1_PP_VOCAB_DEP "${_DSR1_PP_VOCAB_SRC}")
    endif()

    add_custom_command(
      OUTPUT "${_DSR1_PP_RAX_OUT}"
      COMMAND "${CMAKE_COMMAND}" -E make_directory
        "${_DSR1_PP_STAGE_OUT_DIR}" "${_DSR1_PP_OUT_DIR}"
      ${_DSR1_PP_STAGE_SO_CMD}
      ${_DSR1_PP_STAGE_RUNNER_CMD}
      ${_DSR1_PP_STAGE_DEP_CMDS}
      ${_DSR1_PP_STAGE_VOCAB_CMD}
      ${_DSR1_PP_STAGE_WEIGHT_CMDS}
      COMMAND "${_DSR1_PP_RAX_PACK_BIN}" "${_DSR1_PP_MANIFEST}"
        -o "${_DSR1_PP_RAX_TMP}" ${_DSR1_PP_RAX_PACK_EXTRA_ARGS}
      COMMAND "${CMAKE_COMMAND}" -E rename
        "${_DSR1_PP_RAX_TMP}" "${_DSR1_PP_RAX_OUT}"
      DEPENDS
        ${_DSR1_PP_RAX_PACK_DEP}
        "${_DSR1_PP_MANIFEST}"
        ${_DSR1_PP_SO_TARGET}
        "${_DSR1_PP_STAGE_SO_FILE}"
        ${_DSR1_PP_RUNNER_DEPS}
        ${_DSR1_PP_WEIGHT_PATHS}
        ${_DSR1_PP_DEP_LIB_PATHS}
        ${_DSR1_PP_VOCAB_DEP}
      COMMENT "[deepseek_r1_pipeline] stage${_DSR1_PP_RANK}: packing ${_DSR1_PP_RAX_OUT}"
      VERBATIM
    )

    set(_DSR1_PP_STAGE_TARGET "deepseek_r1_pipeline_stage_${_DSR1_PP_RANK}_rax")
    add_custom_target(${_DSR1_PP_STAGE_TARGET} DEPENDS "${_DSR1_PP_RAX_OUT}")
    list(APPEND _DSR1_PP_STAGE_TARGETS ${_DSR1_PP_STAGE_TARGET})

    math(EXPR _DSR1_PP_RANK "${_DSR1_PP_RANK} + 1")
  endforeach()

  add_custom_target(deepseek_r1_pipeline_rax
    DEPENDS ${_DSR1_PP_STAGE_TARGETS}
  )
  message(STATUS
    "[deepseek_r1] Registered deepseek_r1_pipeline_rax for ${_DSR1_PP_RANK} stage(s) "
    "(cross=${IS_RVV_CROSSCOMPILE}, embed=${DEEPSEEKR1_PIPELINE_EMBED_PAYLOAD}, out=${_DSR1_PP_OUT_DIR}).")
endif()
