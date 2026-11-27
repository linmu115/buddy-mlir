find_package(OpenMP QUIET COMPONENTS CXX)
set(driver "${BUDDY_SOURCE_DIR}/tests/Models/Inputs/deepseek_v41")

function(configure_v41_target target)
  target_include_directories(${target} PRIVATE "${BUDDY_SOURCE_DIR}/runtime/include")
  target_compile_features(${target} PRIVATE cxx_std_17)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-ffp-contract=off>)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
  endif()
endfunction()

add_executable(tiny-deepseek-v41 EXCLUDE_FROM_ALL tiny_main.cpp)
configure_v41_target(tiny-deepseek-v41)
add_executable(DeepSeekV4Bench EXCLUDE_FROM_ALL "${driver}/DeepSeekV4Bench.cpp")
configure_v41_target(DeepSeekV4Bench)

set(BUDDY_DEEPSEEK_V41_SPEC_DIR "" CACHE PATH "Upstream DeepSeek-V4.1 inference Python source directory for differential tests")
if(BUDDY_ENABLE_TESTS AND BUDDY_DEEPSEEK_V41_SPEC_DIR)
  set(fixture "${CMAKE_CURRENT_BINARY_DIR}/fixture")
  add_executable(buddy-v41-differential "${driver}/DeepSeekV41Tiny.cpp")
  configure_v41_target(buddy-v41-differential)
  add_test(NAME buddy-v41-reference COMMAND "${Python3_EXECUTABLE}" "${driver}/deepseek_v41_reference.py" "${BUDDY_DEEPSEEK_V41_SPEC_DIR}")
  add_test(NAME buddy-v41-fixture COMMAND "${Python3_EXECUTABLE}" "${driver}/export_v41_fixture.py" "${BUDDY_DEEPSEEK_V41_SPEC_DIR}" "${fixture}")
  set_tests_properties(buddy-v41-fixture PROPERTIES FIXTURES_SETUP v41_weights)
  add_test(NAME buddy-v41-differential COMMAND buddy-v41-differential "${fixture}")
  set_tests_properties(buddy-v41-differential PROPERTIES FIXTURES_REQUIRED v41_weights)
  set_tests_properties(buddy-v41-fixture buddy-v41-reference PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  add_test(NAME buddy-v41-mapped-table COMMAND "${Python3_EXECUTABLE}"
    "${driver}/engram_large_mapping.py" "$<TARGET_FILE:buddy-v41-differential>" "${CMAKE_CURRENT_BINARY_DIR}")
  if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
    add_executable(DeepSeekV4QuantTest "${driver}/DeepSeekV4QuantTest.cpp")
    configure_v41_target(DeepSeekV4QuantTest)
    add_test(NAME buddy-v41-quant-kernels COMMAND DeepSeekV4QuantTest "${CMAKE_CURRENT_BINARY_DIR}/quant")
    set_tests_properties(buddy-v41-quant-kernels PROPERTIES FIXTURES_REQUIRED v41_quant)
    add_dependencies(buddy-v41-differential DeepSeekV4QuantTest python-package-buddy python-package-buddy-mlir)
    add_test(NAME buddy-v41-quant-import COMMAND "${Python3_EXECUTABLE}"
      "${driver}/deepseek_v4_quant_test.py" "${BUDDY_DEEPSEEK_V41_SPEC_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/quant")
    set_tests_properties(buddy-v41-quant-import PROPERTIES
      FIXTURES_SETUP v41_quant
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
      ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${BUDDY_MLIR_PYTHON_PACKAGES_DIR}")
  endif()
endif()
