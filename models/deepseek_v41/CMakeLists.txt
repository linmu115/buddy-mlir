add_executable(tiny-deepseek-v41 EXCLUDE_FROM_ALL tiny_main.cpp)
target_include_directories(tiny-deepseek-v41 PRIVATE "${BUDDY_SOURCE_DIR}/runtime/include")
target_compile_features(tiny-deepseek-v41 PRIVATE cxx_std_17)
target_compile_options(tiny-deepseek-v41 PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-ffp-contract=off>)

set(BUDDY_DEEPSEEK_V41_SPEC_DIR "" CACHE PATH "Upstream DeepSeek-V4.1 inference Python source directory for differential tests")
if(BUDDY_ENABLE_TESTS AND BUDDY_DEEPSEEK_V41_SPEC_DIR)
  set(fixture "${CMAKE_CURRENT_BINARY_DIR}/fixture")
  set(driver "${BUDDY_SOURCE_DIR}/tests/Models/Inputs/deepseek_v41")
  add_executable(buddy-v41-differential "${driver}/DeepSeekV41Tiny.cpp")
  target_include_directories(buddy-v41-differential PRIVATE "${BUDDY_SOURCE_DIR}/runtime/include")
  target_compile_features(buddy-v41-differential PRIVATE cxx_std_17)
  target_compile_options(buddy-v41-differential PRIVATE $<$<COMPILE_LANG_AND_ID:CXX,Clang,GNU>:-ffp-contract=off>)
  add_test(NAME buddy-v41-reference COMMAND "${Python3_EXECUTABLE}" "${driver}/deepseek_v41_reference.py" "${BUDDY_DEEPSEEK_V41_SPEC_DIR}")
  add_test(NAME buddy-v41-fixture COMMAND "${Python3_EXECUTABLE}" "${driver}/export_v41_fixture.py" "${BUDDY_DEEPSEEK_V41_SPEC_DIR}" "${fixture}")
  set_tests_properties(buddy-v41-fixture PROPERTIES FIXTURES_SETUP v41_weights)
  add_test(NAME buddy-v41-differential COMMAND buddy-v41-differential "${fixture}")
  set_tests_properties(buddy-v41-differential PROPERTIES FIXTURES_REQUIRED v41_weights)
  set_tests_properties(buddy-v41-fixture buddy-v41-reference PROPERTIES ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1")
  if(BUDDY_MLIR_ENABLE_PYTHON_PACKAGES)
    add_dependencies(buddy-v41-differential python-package-buddy python-package-buddy-mlir)
    add_test(NAME buddy-v41-quant-import COMMAND "${Python3_EXECUTABLE}"
      "${driver}/deepseek_v4_quant_test.py" "${BUDDY_DEEPSEEK_V41_SPEC_DIR}"
      "${CMAKE_CURRENT_BINARY_DIR}/quant")
    set_tests_properties(buddy-v41-quant-import PROPERTIES
      ENVIRONMENT "PYTHONDONTWRITEBYTECODE=1"
      ENVIRONMENT_MODIFICATION "PYTHONPATH=path_list_prepend:${BUDDY_MLIR_PYTHON_PACKAGES_DIR}")
  endif()
endif()
