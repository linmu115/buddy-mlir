# ===- tests/Pipeline/CMakeLists.txt - Pipeline-parallel CTest suite -------===//
#
# Uses CTest (consistent with tools/buddy-server/tests and models/*/CMakeLists)
# because pipeline tests orchestrate multi-rank mpiexec jobs, MPI_Abort bad-frame
# injection checks with per-test timeouts, MPI-absent SKIP reporting, and
# optional model-artifact end-to-end tests.
#
# Layer 1 (default ON, no model artifacts required):
#   - StageManifest parser unit test (no MPI required)
#   - BlockingMPI 16-byte header protocol & bad-frame injection tests (MPI)
#   - checkAllStages stage-chain & metadata validation tests (MPI)
#
# Layer 2 (default OFF, enable with -DBUDDY_ENABLE_PIPELINE_E2E_TESTS=ON):
#   - 2-stage buddy-cli --stage-model bit-exact logits check via check_result.py
#   - buddy-cli --stage-model error paths
#
# ===----------------------------------------------------------------------===//

if(NOT Python3_EXECUTABLE)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

find_package(MPI COMPONENTS CXX QUIET)
if(MPI_CXX_FOUND AND NOT MPIEXEC_EXECUTABLE)
  find_program(MPIEXEC_EXECUTABLE NAMES mpiexec mpirun)
endif()

set(_BUDDY_PIPELINE_HAVE_MPI FALSE)
if(MPI_CXX_FOUND AND MPIEXEC_EXECUTABLE)
  set(_BUDDY_PIPELINE_HAVE_MPI TRUE)
endif()

set(_BUDDY_PIPELINE_MPI_NUMPROC_FLAG "${MPIEXEC_NUMPROC_FLAG}")
if(NOT _BUDDY_PIPELINE_MPI_NUMPROC_FLAG)
  set(_BUDDY_PIPELINE_MPI_NUMPROC_FLAG "-n")
endif()

function(_add_buddy_pipeline_skipped_mpi_test test_name extra_labels)
  add_test(NAME ${test_name}
    COMMAND "${Python3_EXECUTABLE}" -c
      "import sys; print('SKIPPED: MPI not available'); sys.exit(77)")
  set_tests_properties(${test_name} PROPERTIES
    SKIP_RETURN_CODE 77
    LABELS "pipeline;${extra_labels};pipeline-mpi")
endfunction()

set(_BUDDY_PIPELINE_UNIT_DEPENDS)
set(_BUDDY_PIPELINE_ALL_DEPENDS)

# ------------------------------------------------------------------------------
# Layer 1A: StageManifest parser unit test (no MPI required)
# ------------------------------------------------------------------------------
add_executable(buddy-pipeline-stage-manifest-test
  StageManifestTest.cpp
)
target_compile_features(buddy-pipeline-stage-manifest-test PRIVATE cxx_std_17)
target_link_libraries(buddy-pipeline-stage-manifest-test PRIVATE
  buddy_runtime_core
)
add_test(NAME buddy-pipeline-stage-manifest
  COMMAND buddy-pipeline-stage-manifest-test
)
set_tests_properties(buddy-pipeline-stage-manifest PROPERTIES
  TIMEOUT 30
  PASS_REGULAR_EXPRESSION "StageManifest tests passed"
  LABELS "pipeline;pipeline-unit"
)
list(APPEND _BUDDY_PIPELINE_UNIT_DEPENDS buddy-pipeline-stage-manifest-test)

# ------------------------------------------------------------------------------
# Layer 1B & 1C: MPI protocol & stage-chain unit tests (skipped if no MPI)
# ------------------------------------------------------------------------------
set(_BUDDY_PIPELINE_PROTOCOL_TESTS
  buddy-pipeline-protocol-normal
  buddy-pipeline-protocol-forward3
  buddy-pipeline-protocol-bad-duplicate-step
  buddy-pipeline-protocol-bad-wrong-pos
  buddy-pipeline-protocol-bad-oob-pos
  buddy-pipeline-protocol-bad-short-payload
  buddy-pipeline-protocol-bad-short-prefill
)

set(_BUDDY_PIPELINE_CHAIN_TESTS
  buddy-pipeline-stage-chain-valid-1rank
  buddy-pipeline-stage-chain-valid-2rank
  buddy-pipeline-stage-chain-valid-3rank
  buddy-pipeline-stage-chain-overlap-2rank
  buddy-pipeline-stage-chain-overlap-3rank
  buddy-pipeline-stage-chain-gap-2rank
  buddy-pipeline-stage-chain-model-name-mismatch
  buddy-pipeline-stage-chain-field-mismatch
)

if(_BUDDY_PIPELINE_HAVE_MPI)
  # Protocol test binary (Step 3)
  add_executable(buddy-pipeline-protocol-test
    BlockingMPIProtocolTest.cpp
  )
  target_compile_features(buddy-pipeline-protocol-test PRIVATE cxx_std_17)
  target_link_libraries(buddy-pipeline-protocol-test PRIVATE
    buddy_runtime_core
    MPI::MPI_CXX
  )
  list(APPEND _BUDDY_PIPELINE_UNIT_DEPENDS buddy-pipeline-protocol-test)

  set(_BUDDY_PIPELINE_PROTOCOL_RUNNER
    "${CMAKE_CURRENT_SOURCE_DIR}/run_mpi_protocol_test.py")
  set(_BUDDY_PIPELINE_PROTOCOL_WORK_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/protocol-work")

  add_test(NAME buddy-pipeline-protocol-normal
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 2
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode normal
      --timeout 60
      --expect-output "PASS normal-2rank"
  )
  add_test(NAME buddy-pipeline-protocol-forward3
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 3
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode forward3
      --timeout 60
      --expect-output "PASS forward-3rank"
  )
  add_test(NAME buddy-pipeline-protocol-bad-duplicate-step
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 2
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode bad-duplicate-step
      --timeout 60
      --expect-fail
      --expect-output "header step mismatch"
      --counter-file "${_BUDDY_PIPELINE_PROTOCOL_WORK_DIR}/bad-duplicate-step.counter"
      --expect-compute-calls 2
      --expect-prefill-calls 1
      --expect-decode-calls 1
  )
  add_test(NAME buddy-pipeline-protocol-bad-wrong-pos
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 2
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode bad-wrong-pos
      --timeout 60
      --expect-fail
      --expect-output "header pos mismatch"
      --counter-file "${_BUDDY_PIPELINE_PROTOCOL_WORK_DIR}/bad-wrong-pos.counter"
      --expect-compute-calls 1
      --expect-prefill-calls 1
      --expect-decode-calls 0
  )
  add_test(NAME buddy-pipeline-protocol-bad-oob-pos
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 2
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode bad-oob-pos
      --timeout 60
      --expect-fail
      --expect-output "header phase/pos/rows invalid"
      --counter-file "${_BUDDY_PIPELINE_PROTOCOL_WORK_DIR}/bad-oob-pos.counter"
      --expect-compute-calls 1
      --expect-prefill-calls 1
      --expect-decode-calls 0
  )
  add_test(NAME buddy-pipeline-protocol-bad-short-payload
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 2
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode bad-short-payload
      --timeout 60
      --expect-fail
      --expect-output "payload element count mismatch"
      --counter-file "${_BUDDY_PIPELINE_PROTOCOL_WORK_DIR}/bad-short-payload.counter"
      --expect-compute-calls 1
      --expect-prefill-calls 1
      --expect-decode-calls 0
  )
  add_test(NAME buddy-pipeline-protocol-bad-short-prefill
    COMMAND "${Python3_EXECUTABLE}" "${_BUDDY_PIPELINE_PROTOCOL_RUNNER}"
      --mpiexec "${MPIEXEC_EXECUTABLE}"
      "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
      --ranks 2
      --binary "$<TARGET_FILE:buddy-pipeline-protocol-test>"
      --mode bad-short-prefill
      --timeout 60
      --expect-fail
      --expect-output "payload element count mismatch"
      --counter-file "${_BUDDY_PIPELINE_PROTOCOL_WORK_DIR}/bad-short-prefill.counter"
      --expect-compute-calls 0
      --expect-prefill-calls 0
      --expect-decode-calls 0
  )
  foreach(_proto_test IN LISTS _BUDDY_PIPELINE_PROTOCOL_TESTS)
    set_tests_properties(${_proto_test} PROPERTIES
      TIMEOUT 75
      LABELS "pipeline;pipeline-unit;pipeline-mpi"
    )
  endforeach()

  # Stage chain validation test binary (Step 4)
  add_executable(buddy-pipeline-stage-chain-test
    StageChainCheckTest.cpp
  )
  target_compile_features(buddy-pipeline-stage-chain-test PRIVATE cxx_std_17)
  target_include_directories(buddy-pipeline-stage-chain-test PRIVATE
    "${CMAKE_SOURCE_DIR}/models/deepseek_r1/pipeline"
    "${CMAKE_SOURCE_DIR}/frontend/Interfaces"
    "${CMAKE_BINARY_DIR}/frontend/Interfaces"
  )
  target_link_libraries(buddy-pipeline-stage-chain-test PRIVATE
    buddy_runtime_core
    MPI::MPI_CXX
  )
  list(APPEND _BUDDY_PIPELINE_UNIT_DEPENDS buddy-pipeline-stage-chain-test)

  function(_add_stage_chain_test test_name ranks mode)
    add_test(NAME ${test_name}
      COMMAND "${MPIEXEC_EXECUTABLE}" "${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
        "${ranks}" "$<TARGET_FILE:buddy-pipeline-stage-chain-test>" "${mode}"
    )
    set_tests_properties(${test_name} PROPERTIES
      TIMEOUT 60
      PASS_REGULAR_EXPRESSION "PASS stage-chain ${mode}"
      LABELS "pipeline;pipeline-unit;pipeline-mpi"
    )
  endfunction()

  _add_stage_chain_test(buddy-pipeline-stage-chain-valid-1rank 1 valid-1rank)
  _add_stage_chain_test(buddy-pipeline-stage-chain-valid-2rank 2 valid-2rank)
  _add_stage_chain_test(buddy-pipeline-stage-chain-valid-3rank 3 valid-3rank)
  _add_stage_chain_test(buddy-pipeline-stage-chain-overlap-2rank 2 overlap-2rank)
  _add_stage_chain_test(buddy-pipeline-stage-chain-overlap-3rank 3 overlap-3rank)
  _add_stage_chain_test(buddy-pipeline-stage-chain-gap-2rank 2 gap-2rank)
  _add_stage_chain_test(buddy-pipeline-stage-chain-model-name-mismatch 2 model-name-mismatch)
  _add_stage_chain_test(buddy-pipeline-stage-chain-field-mismatch 2 field-mismatch)
else()
  message(STATUS
    "[pipeline-tests] MPI not found; MPI unit tests will be skipped in CTest.")
  foreach(_mpi_test IN LISTS _BUDDY_PIPELINE_PROTOCOL_TESTS _BUDDY_PIPELINE_CHAIN_TESTS)
    _add_buddy_pipeline_skipped_mpi_test("${_mpi_test}" "pipeline-unit")
  endforeach()
endif()

list(APPEND _BUDDY_PIPELINE_ALL_DEPENDS ${_BUDDY_PIPELINE_UNIT_DEPENDS})

# ------------------------------------------------------------------------------
# Layer 2: Model-artifact E2E tests (default OFF)
# ------------------------------------------------------------------------------
option(BUDDY_ENABLE_PIPELINE_E2E_TESTS
  "Enable pipeline-parallel E2E tests requiring pre-built stage artifacts and baseline logits"
  OFF)
set(BUDDY_PIPELINE_STAGE_ARTIFACTS_DIR "" CACHE PATH
  "Directory containing pre-built pipeline stage directories (e.g. 0-14, 14-28)")
set(BUDDY_PIPELINE_VALIDATION_DIR "" CACHE PATH
  "Directory containing baseline validation JSON and FP16 logits")
set(BUDDY_PIPELINE_E2E_STAGE_SUBDIRS "0-14;14-28" CACHE STRING
  "Ordered semicolon-separated stage subdirectories under BUDDY_PIPELINE_STAGE_ARTIFACTS_DIR")
set(BUDDY_PIPELINE_E2E_VOCAB_FILE
  "${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/vocab.txt" CACHE FILEPATH
  "Fallback vocab.txt for E2E pipeline stage packaging")
set(BUDDY_PIPELINE_E2E_RAX_PACK "" CACHE FILEPATH
  "Optional pre-built rax-pack executable for pipeline E2E fixtures (defaults to in-tree rax-pack)")
option(BUDDY_PIPELINE_E2E_EMBED_PAYLOAD
  "Embed payloads into E2E test stage .rax fixtures (OFF = fast manifest-only)"
  OFF)
set(BUDDY_PIPELINE_E2E_PROMPT
  "Explain step by step how to add the integers from 1 to 100." CACHE STRING
  "Prompt matching the reference logits in BUDDY_PIPELINE_VALIDATION_DIR")
set(BUDDY_PIPELINE_E2E_MAX_TOKENS "32" CACHE STRING
  "Number of generated tokens matching the reference logits")
set(BUDDY_PIPELINE_E2E_VOCAB_SIZE "151936" CACHE STRING
  "Vocabulary size passed to check_result.py for logits comparison")

if(BUDDY_ENABLE_PIPELINE_E2E_TESTS)
  if(NOT _BUDDY_PIPELINE_HAVE_MPI OR NOT BUDDY_CLI_ENABLE_MPI)
    message(STATUS
      "[pipeline-tests] MPI or BUDDY_CLI_ENABLE_MPI unavailable; "
      "pipeline E2E tests will be skipped in CTest.")
    _add_buddy_pipeline_skipped_mpi_test(buddy-pipeline-e2e-2stage-logits "pipeline-e2e")
    _add_buddy_pipeline_skipped_mpi_test(buddy-pipeline-e2e-cli-errors "pipeline-e2e")
  else()
    if(NOT BUDDY_PIPELINE_STAGE_ARTIFACTS_DIR OR NOT IS_DIRECTORY "${BUDDY_PIPELINE_STAGE_ARTIFACTS_DIR}")
      message(FATAL_ERROR
        "[pipeline-tests] BUDDY_ENABLE_PIPELINE_E2E_TESTS=ON requires "
        "BUDDY_PIPELINE_STAGE_ARTIFACTS_DIR pointing to pre-built stage directories.")
    endif()
    if(NOT BUDDY_PIPELINE_VALIDATION_DIR OR NOT IS_DIRECTORY "${BUDDY_PIPELINE_VALIDATION_DIR}")
      message(FATAL_ERROR
        "[pipeline-tests] BUDDY_ENABLE_PIPELINE_E2E_TESTS=ON requires "
        "BUDDY_PIPELINE_VALIDATION_DIR pointing to baseline JSON/logits files.")
    endif()

    set(_PP_E2E_VERIFY_SCRIPT
      "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/verify_pipeline_stage.py")
    set(_PP_E2E_GEN_MANIFEST_SCRIPT
      "${CMAKE_SOURCE_DIR}/tools/buddy-codegen/gen_manifest.py")
    set(_PP_E2E_CHECK_RESULT_SCRIPT
      "${CMAKE_SOURCE_DIR}/examples/BuddyDeepSeekR1/pipeline/check_result.py")
    set(_PP_E2E_RUNNER_SCRIPT
      "${CMAKE_CURRENT_SOURCE_DIR}/run_pipeline_e2e_test.py")

    if(BUDDY_PIPELINE_E2E_RAX_PACK)
      if(NOT EXISTS "${BUDDY_PIPELINE_E2E_RAX_PACK}")
        message(FATAL_ERROR
          "[pipeline-tests] BUDDY_PIPELINE_E2E_RAX_PACK not found: ${BUDDY_PIPELINE_E2E_RAX_PACK}")
      endif()
      set(_PP_E2E_RAX_PACK_BIN "${BUDDY_PIPELINE_E2E_RAX_PACK}")
      set(_PP_E2E_RAX_PACK_DEP "${BUDDY_PIPELINE_E2E_RAX_PACK}")
    elseif(DEFINED DEEPSEEKR1_PIPELINE_RAX_PACK AND DEEPSEEKR1_PIPELINE_RAX_PACK AND EXISTS "${DEEPSEEKR1_PIPELINE_RAX_PACK}")
      set(_PP_E2E_RAX_PACK_BIN "${DEEPSEEKR1_PIPELINE_RAX_PACK}")
      set(_PP_E2E_RAX_PACK_DEP "${DEEPSEEKR1_PIPELINE_RAX_PACK}")
    else()
      set(_PP_E2E_RAX_PACK_BIN "$<TARGET_FILE:rax-pack>")
      set(_PP_E2E_RAX_PACK_DEP rax-pack)
    endif()

    # Reuse the DeepSeek R1 pipeline runner plugin target if present; otherwise
    # build it for the E2E fixture so E2E tests work even when
    # BUDDY_BUILD_DEEPSEEK_R1_MODEL=OFF.
    if(TARGET buddy_models_deepseek_r1_pipeline_runner)
      set(_PP_E2E_RUNNER_TARGET buddy_models_deepseek_r1_pipeline_runner)
    else()
      add_library(buddy-pipeline-e2e-runner SHARED EXCLUDE_FROM_ALL
        "${CMAKE_SOURCE_DIR}/models/deepseek_r1/pipeline/PipelineRunner.cpp"
        "${CMAKE_SOURCE_DIR}/models/deepseek_r1/pipeline/PipelineRunnerPlugin.cpp"
      )
      set_target_properties(buddy-pipeline-e2e-runner PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/e2e_fixtures"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/e2e_fixtures"
        OUTPUT_NAME "deepseek_r1_pipeline_runner"
        PREFIX ""
      )
      target_include_directories(buddy-pipeline-e2e-runner PRIVATE
        "${CMAKE_SOURCE_DIR}/models/deepseek_r1/pipeline"
        "${CMAKE_SOURCE_DIR}/frontend/Interfaces"
        "${CMAKE_BINARY_DIR}/frontend/Interfaces"
      )
      target_link_libraries(buddy-pipeline-e2e-runner PRIVATE
        buddy_runtime_core
        MPI::MPI_CXX
        ${CMAKE_DL_LIBS}
      )
      target_compile_features(buddy-pipeline-e2e-runner PRIVATE cxx_std_17)
      set(_PP_E2E_RUNNER_TARGET buddy-pipeline-e2e-runner)
    endif()

    # Locate MLIR & OpenMP runtime shared libraries needed by stage.so.
    set(_PP_E2E_CRUNNER_CAND
      "${LLVM_LIBRARY_DIR}/libmlir_c_runner_utils${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_PP_E2E_CRUNNER_CAND}")
      message(FATAL_ERROR
        "[pipeline-tests] Missing required MLIR runtime library: ${_PP_E2E_CRUNNER_CAND}")
    endif()
    file(REAL_PATH "${_PP_E2E_CRUNNER_CAND}" _PP_E2E_CRUNNER_REAL)
    get_filename_component(_PP_E2E_CRUNNER_NAME "${_PP_E2E_CRUNNER_REAL}" NAME)

    set(_PP_E2E_OMP_LIB "")
    if(BUDDY_OPENMP_RUNTIME_LIBRARY AND EXISTS "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
      set(_PP_E2E_OMP_LIB "${BUDDY_OPENMP_RUNTIME_LIBRARY}")
    else()
      get_filename_component(_PP_E2E_LLVM_ROOT "${LLVM_LIBRARY_DIR}" DIRECTORY)
      foreach(_omp_cand
        "${_PP_E2E_LLVM_ROOT}/runtimes/runtimes-bins/openmp/runtime/src/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
        "${LLVM_LIBRARY_DIR}/libomp${CMAKE_SHARED_LIBRARY_SUFFIX}"
      )
        if(EXISTS "${_omp_cand}")
          set(_PP_E2E_OMP_LIB "${_omp_cand}")
          break()
        endif()
      endforeach()
    endif()
    if(NOT _PP_E2E_OMP_LIB)
      find_library(_PP_E2E_OMP_LIB NAMES omp)
    endif()
    if(NOT _PP_E2E_OMP_LIB OR NOT EXISTS "${_PP_E2E_OMP_LIB}")
      message(FATAL_ERROR
        "[pipeline-tests] Missing OpenMP runtime library (set BUDDY_OPENMP_RUNTIME_LIBRARY).")
    endif()
    file(REAL_PATH "${_PP_E2E_OMP_LIB}" _PP_E2E_OMP_REAL)
    get_filename_component(_PP_E2E_OMP_NAME "${_PP_E2E_OMP_REAL}" NAME)

    set(_PP_E2E_DEP_LIB_PATHS)
    set(_PP_E2E_DEP_LIB_NAMES)
    foreach(_mlir_dep_base libmlir_apfloat_wrappers libmlir_float16_utils)
      set(_mlir_dep_cand
        "${LLVM_LIBRARY_DIR}/${_mlir_dep_base}${CMAKE_SHARED_LIBRARY_SUFFIX}")
      if(EXISTS "${_mlir_dep_cand}")
        file(REAL_PATH "${_mlir_dep_cand}" _mlir_dep_real)
        get_filename_component(_mlir_dep_name "${_mlir_dep_real}" NAME)
        list(APPEND _PP_E2E_DEP_LIB_PATHS "${_mlir_dep_real}")
        list(APPEND _PP_E2E_DEP_LIB_NAMES "${_mlir_dep_name}")
      endif()
    endforeach()
    list(APPEND _PP_E2E_DEP_LIB_PATHS "${_PP_E2E_CRUNNER_REAL}" "${_PP_E2E_OMP_REAL}")
    list(APPEND _PP_E2E_DEP_LIB_NAMES "${_PP_E2E_CRUNNER_NAME}" "${_PP_E2E_OMP_NAME}")

    set(_PP_E2E_DEP_MANIFEST_ARGS)
    foreach(_dep_name IN LISTS _PP_E2E_DEP_LIB_NAMES)
      list(APPEND _PP_E2E_DEP_MANIFEST_ARGS --dep-shared-lib "${_dep_name}")
    endforeach()
    list(LENGTH _PP_E2E_DEP_LIB_PATHS _PP_E2E_DEP_COUNT)
    math(EXPR _PP_E2E_DEP_LAST "${_PP_E2E_DEP_COUNT} - 1")

    set(_PP_E2E_HARDLINK_PY
      "import os, shutil, sys; s, d = sys.argv[1], sys.argv[2]; exec('if os.path.lexists(d):\\n  try:\\n    if not os.path.islink(d) and os.path.samefile(s, d): sys.exit(0)\\n  except OSError: pass\\n  os.unlink(d)\\ntry: os.link(s, d)\\nexcept OSError: shutil.copy2(s, d)')")

    set(_PP_E2E_STAGE_RAX_ARGS)
    set(_PP_E2E_STAGE_OUTPUTS)
    set(_PP_E2E_RANK 0)
    foreach(_raw_sub IN LISTS BUDDY_PIPELINE_E2E_STAGE_SUBDIRS)
      if(IS_ABSOLUTE "${_raw_sub}")
        set(_stage_dir "${_raw_sub}")
      else()
        set(_stage_dir "${BUDDY_PIPELINE_STAGE_ARTIFACTS_DIR}/${_raw_sub}")
      endif()
      get_filename_component(_stage_dir "${_stage_dir}" ABSOLUTE)
      if(NOT IS_DIRECTORY "${_stage_dir}")
        message(FATAL_ERROR
          "[pipeline-tests] E2E stage ${_PP_E2E_RANK} dir not found: ${_stage_dir}")
      endif()

      set(_cfg_path "${_stage_dir}/config.json")
      if(NOT EXISTS "${_cfg_path}")
        message(FATAL_ERROR
          "[pipeline-tests] E2E stage ${_PP_E2E_RANK} missing config.json: ${_cfg_path}")
      endif()
      file(READ "${_cfg_path}" _cfg_json)
      string(JSON _start_layer GET "${_cfg_json}" pipeline_stage start_layer)
      string(JSON _weight_file GET "${_cfg_json}" weights 0 file)
      string(JSON _vocab_file GET "${_cfg_json}" tokens vocab_file)

      set(_is_first FALSE)
      if(_start_layer EQUAL 0)
        set(_is_first TRUE)
      endif()

      set(_prefill_mlir "${_stage_dir}/forward_prefill-f16.mlir")
      set(_decode_mlir "${_stage_dir}/forward_decode-f16.mlir")
      set(_prefill_o "${_stage_dir}/forward_prefill.o")
      set(_decode_o "${_stage_dir}/forward_decode.o")
      set(_sub_prefill_o "${_stage_dir}/subgraph_prefill.o")
      set(_sub_decode_o "${_stage_dir}/subgraph_decode.o")
      set(_weight_path "${_stage_dir}/${_weight_file}")
      foreach(_req
        "${_prefill_mlir}" "${_decode_mlir}"
        "${_prefill_o}" "${_decode_o}"
        "${_sub_prefill_o}" "${_sub_decode_o}"
        "${_weight_path}"
      )
        if(NOT EXISTS "${_req}")
          message(FATAL_ERROR
            "[pipeline-tests] E2E stage ${_PP_E2E_RANK} missing required file: ${_req}")
        endif()
      endforeach()

      set(_vocab_src "")
      if(_is_first)
        if(EXISTS "${_stage_dir}/${_vocab_file}")
          set(_vocab_src "${_stage_dir}/${_vocab_file}")
        else()
          set(_vocab_src "${BUDDY_PIPELINE_E2E_VOCAB_FILE}")
        endif()
        if(NOT EXISTS "${_vocab_src}")
          message(FATAL_ERROR
            "[pipeline-tests] E2E stage ${_PP_E2E_RANK} missing vocab file: ${_vocab_src}")
        endif()
      endif()

      set(_work_dir "${CMAKE_CURRENT_BINARY_DIR}/e2e_fixtures/stage${_PP_E2E_RANK}")
      set(_cfg_inc "${_work_dir}/stage-config.inc")
      set(_manifest "${_work_dir}/stage.mlir")
      set(_rax_tmp "${_work_dir}/stage${_PP_E2E_RANK}.rax.tmp")
      set(_rax_out "${_work_dir}/stage${_PP_E2E_RANK}.rax")

      add_custom_command(
        OUTPUT "${_cfg_inc}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_work_dir}"
        COMMAND "${Python3_EXECUTABLE}" "${_PP_E2E_VERIFY_SCRIPT}"
          --stage-dir "${_stage_dir}"
          --header "${_cfg_inc}"
        DEPENDS
          "${_PP_E2E_VERIFY_SCRIPT}"
          "${_cfg_path}"
          "${_prefill_mlir}"
          "${_decode_mlir}"
          "${_weight_path}"
        COMMENT "[pipeline-tests] E2E stage${_PP_E2E_RANK}: verifying ABI and generating stage-config.inc"
        VERBATIM
      )

      set(_so_target "buddy-pipeline-e2e-stage-${_PP_E2E_RANK}-so")
      set(_stage_objs "${_prefill_o}" "${_decode_o}" "${_sub_prefill_o}" "${_sub_decode_o}")
      set_source_files_properties(${_stage_objs} PROPERTIES EXTERNAL_OBJECT TRUE)
      add_library(${_so_target} SHARED EXCLUDE_FROM_ALL
        "${CMAKE_SOURCE_DIR}/models/deepseek_r1/pipeline/Stage.cpp"
        "${_cfg_inc}"
        ${_stage_objs}
      )
      set_target_properties(${_so_target} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${_work_dir}"
        RUNTIME_OUTPUT_DIRECTORY "${_work_dir}"
        OUTPUT_NAME "stage"
        PREFIX ""
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "\$ORIGIN"
      )
      target_include_directories(${_so_target} PRIVATE
        "${_work_dir}"
        ${MLIR_INCLUDE_DIRS}
        "${CMAKE_SOURCE_DIR}/runtime/include"
      )
      target_link_libraries(${_so_target} PRIVATE
        "${_PP_E2E_CRUNNER_REAL}"
        "${_PP_E2E_OMP_REAL}"
      )
      if(NOT APPLE)
        target_link_options(${_so_target} PRIVATE "-Wl,-z,defs")
      endif()
      target_compile_features(${_so_target} PRIVATE cxx_std_17)

      add_custom_command(
        OUTPUT "${_manifest}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_work_dir}"
        COMMAND "${Python3_EXECUTABLE}" "${_PP_E2E_GEN_MANIFEST_SCRIPT}"
          --config "${_cfg_path}"
          ${_PP_E2E_DEP_MANIFEST_ARGS}
          --runner-library "$<TARGET_FILE_NAME:${_PP_E2E_RUNNER_TARGET}>"
          -o "${_manifest}"
        DEPENDS
          "${_PP_E2E_GEN_MANIFEST_SCRIPT}"
          "${_cfg_path}"
        COMMENT "[pipeline-tests] E2E stage${_PP_E2E_RANK}: generating manifest"
        VERBATIM
      )

      set(_dep_cmds)
      foreach(_dep_idx RANGE 0 ${_PP_E2E_DEP_LAST})
        list(GET _PP_E2E_DEP_LIB_PATHS ${_dep_idx} _dep_path)
        list(GET _PP_E2E_DEP_LIB_NAMES ${_dep_idx} _dep_name)
        list(APPEND _dep_cmds
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_dep_path}" "${_work_dir}/${_dep_name}")
      endforeach()

      set(_vocab_cmd)
      set(_vocab_dep)
      if(_is_first)
        set(_vocab_cmd
          COMMAND "${CMAKE_COMMAND}" -E copy_if_different
            "${_vocab_src}" "${_work_dir}/${_vocab_file}")
        set(_vocab_dep "${_vocab_src}")
      endif()

      set(_embed_args)
      if(BUDDY_PIPELINE_E2E_EMBED_PAYLOAD)
        set(_embed_args --embed-payload)
      endif()

      add_custom_command(
        OUTPUT "${_rax_out}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${_work_dir}"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
          "$<TARGET_FILE:${_PP_E2E_RUNNER_TARGET}>"
          "${_work_dir}/$<TARGET_FILE_NAME:${_PP_E2E_RUNNER_TARGET}>"
        ${_dep_cmds}
        ${_vocab_cmd}
        COMMAND "${Python3_EXECUTABLE}" -c "${_PP_E2E_HARDLINK_PY}"
          "${_weight_path}" "${_work_dir}/${_weight_file}"
        COMMAND "${_PP_E2E_RAX_PACK_BIN}" "${_manifest}"
          -o "${_rax_tmp}" ${_embed_args}
        COMMAND "${CMAKE_COMMAND}" -E rename "${_rax_tmp}" "${_rax_out}"
        DEPENDS
          ${_PP_E2E_RAX_PACK_DEP}
          "${_manifest}"
          ${_so_target}
          "$<TARGET_FILE:${_so_target}>"
          ${_PP_E2E_RUNNER_TARGET}
          "$<TARGET_FILE:${_PP_E2E_RUNNER_TARGET}>"
          "${_weight_path}"
          ${_PP_E2E_DEP_LIB_PATHS}
          ${_vocab_dep}
        COMMENT "[pipeline-tests] E2E stage${_PP_E2E_RANK}: packing ${_rax_out}"
        VERBATIM
      )

      list(APPEND _PP_E2E_STAGE_OUTPUTS "${_rax_out}")
      list(APPEND _PP_E2E_STAGE_RAX_ARGS --stage-rax "${_rax_out}")
      math(EXPR _PP_E2E_RANK "${_PP_E2E_RANK} + 1")
    endforeach()

    add_custom_target(buddy-pipeline-e2e-fixtures
      DEPENDS ${_PP_E2E_STAGE_OUTPUTS}
    )

    add_test(NAME buddy-pipeline-e2e-2stage-logits
      COMMAND "${Python3_EXECUTABLE}" "${_PP_E2E_RUNNER_SCRIPT}"
        logits
        --mpiexec "${MPIEXEC_EXECUTABLE}"
        "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
        --buddy-cli "$<TARGET_FILE:buddy-cli>"
        ${_PP_E2E_STAGE_RAX_ARGS}
        --validation-dir "${BUDDY_PIPELINE_VALIDATION_DIR}"
        --check-result "${_PP_E2E_CHECK_RESULT_SCRIPT}"
        --prompt "${BUDDY_PIPELINE_E2E_PROMPT}"
        --max-tokens "${BUDDY_PIPELINE_E2E_MAX_TOKENS}"
        --vocab-size "${BUDDY_PIPELINE_E2E_VOCAB_SIZE}"
        --work-dir "${CMAKE_CURRENT_BINARY_DIR}/e2e-work-logits"
        --timeout 180
    )
    set_tests_properties(buddy-pipeline-e2e-2stage-logits PROPERTIES
      TIMEOUT 200
      PASS_REGULAR_EXPRESSION "PASS pipeline e2e 2-stage bit-exact logits"
      LABELS "pipeline;pipeline-e2e;pipeline-mpi"
    )

    add_test(NAME buddy-pipeline-e2e-cli-errors
      COMMAND "${Python3_EXECUTABLE}" "${_PP_E2E_RUNNER_SCRIPT}"
        cli-errors
        --mpiexec "${MPIEXEC_EXECUTABLE}"
        "--numproc-flag=${_BUDDY_PIPELINE_MPI_NUMPROC_FLAG}"
        --buddy-cli "$<TARGET_FILE:buddy-cli>"
        ${_PP_E2E_STAGE_RAX_ARGS}
        --prompt "${BUDDY_PIPELINE_E2E_PROMPT}"
        --work-dir "${CMAKE_CURRENT_BINARY_DIR}/e2e-work-cli-errors"
        --timeout 120
    )
    set_tests_properties(buddy-pipeline-e2e-cli-errors PROPERTIES
      TIMEOUT 240
      PASS_REGULAR_EXPRESSION "PASS pipeline e2e cli-errors"
      LABELS "pipeline;pipeline-e2e;pipeline-mpi"
    )

    list(APPEND _BUDDY_PIPELINE_ALL_DEPENDS
      buddy-cli
      buddy-pipeline-e2e-fixtures
    )

    add_custom_target(check-buddy-pipeline-e2e
      COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L "^pipeline-e2e$"
      DEPENDS buddy-cli buddy-pipeline-e2e-fixtures
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
    )
  endif()
endif()

add_custom_target(check-buddy-pipeline-unit
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L "^pipeline-unit$"
  DEPENDS ${_BUDDY_PIPELINE_UNIT_DEPENDS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)

add_custom_target(check-buddy-pipeline
  COMMAND "${CMAKE_CTEST_COMMAND}" --output-on-failure -L "^pipeline$"
  DEPENDS ${_BUDDY_PIPELINE_ALL_DEPENDS}
  WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
)
