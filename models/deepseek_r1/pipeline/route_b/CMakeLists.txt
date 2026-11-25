cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(BJ47ComponentKernels LANGUAGES C CXX)
endif()
find_package(Python3 REQUIRED COMPONENTS Interpreter)
get_filename_component(repo "${CMAKE_CURRENT_LIST_DIR}/../../../.." ABSOLUTE)
set(codegen "${repo}/tools/buddy-codegen")
foreach(var BJ47_ROUTE_B_BUNDLE BJ47_BUDDY_OPT BJ47_LLVM_TOOLS BJ47_CRUNNER)
  if(NOT EXISTS "${${var}}")
    message(FATAL_ERROR "Set ${var} to an existing artifact")
  endif()
endforeach()
add_library(bj47_kernel_runtime SHARED Runtime.c
  "${repo}/runtime/arena/BuddyArena.c" "${repo}/runtime/threadpool/BuddyThreadPool.c"
  "${repo}/runtime/spacemit/BuddySpacemitTcm.c")
target_compile_definitions(bj47_kernel_runtime PRIVATE BUDDY_COMPONENT_RUNTIME)
target_compile_options(bj47_kernel_runtime PRIVATE -O2)
target_link_libraries(bj47_kernel_runtime PRIVATE pthread dl m)
set_target_properties(bj47_kernel_runtime PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${BJ47_ROUTE_B_BUNDLE}")
# Load this diagnostic runtime via LD_LIBRARY_PATH, with identical kernel DSOs.
add_library(bj47_checked_heap_runtime SHARED EXCLUDE_FROM_ALL Runtime.c CheckedHeap.c
  "${repo}/runtime/threadpool/BuddyThreadPool.c" "${repo}/runtime/spacemit/BuddySpacemitTcm.c")
target_compile_definitions(bj47_checked_heap_runtime PRIVATE BUDDY_COMPONENT_RUNTIME BJ47_CHECKED_HEAP)
target_compile_options(bj47_checked_heap_runtime PRIVATE -O2)
target_link_libraries(bj47_checked_heap_runtime PRIVATE pthread dl m)
set_target_properties(bj47_checked_heap_runtime PROPERTIES OUTPUT_NAME bj47_kernel_runtime
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/checked-heap")
foreach(role embed decoder_layer lm_head)
  set(out "${BJ47_ROUTE_B_BUNDLE}/${role}.so")
  file(GLOB mlir CONFIGURE_DEPENDS "${BJ47_ROUTE_B_BUNDLE}/mlir/${role}/*.mlir")
  set(receipt "${BJ47_ROUTE_B_BUNDLE}/${role}.kernel.json")
  add_custom_command(OUTPUT "${out}" "${receipt}"
    COMMAND "${Python3_EXECUTABLE}" "${codegen}/build_route_b.py"
      --bundle "${BJ47_ROUTE_B_BUNDLE}" --role "${role}" --work "${CMAKE_CURRENT_BINARY_DIR}/${role}"
      --buddy-opt "${BJ47_BUDDY_OPT}" --llvm "${BJ47_LLVM_TOOLS}" --cxx "${CMAKE_CXX_COMPILER}" --crunner "${BJ47_CRUNNER}"
    DEPENDS bj47_kernel_runtime ${mlir} "${BJ47_ROUTE_B_BUNDLE}/components.json"
      "${codegen}/build_route_b.py" "${codegen}/route_b_lower.py" "${codegen}/compile_pipeline.py"
      Shim.h "${repo}/runtime/include/buddy/runtime/pipeline/RouteB.h"
    VERBATIM)
  list(APPEND component_outputs "${out}" "${receipt}")
endforeach()
add_custom_target(bj47-route-b-kernels DEPENDS ${component_outputs})

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(CTest)
endif()
if(BUILD_TESTING)
  add_test(NAME test_route_b_counts_and_roles COMMAND "${Python3_EXECUTABLE}"
    "${repo}/tests/Pipeline/test_route_b_counts_and_roles.py")
  add_test(NAME test_route_b_pack_layout COMMAND "${Python3_EXECUTABLE}"
    "${repo}/tests/Pipeline/test_route_b_pack_layout.py" --self-test)
  set_tests_properties(test_route_b_counts_and_roles test_route_b_pack_layout
    PROPERTIES LABELS route-b TIMEOUT 240)
endif()
