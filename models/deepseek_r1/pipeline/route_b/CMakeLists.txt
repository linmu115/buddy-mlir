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
set(components "${BJ47_ROUTE_B_BUNDLE}/components.json")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${components}")
file(READ "${components}" metadata)
string(JSON prefault ERROR_VARIABLE missing GET "${metadata}" config arena_prefault_mb)
if(missing)
  set(prefault 512)
endif()
separate_arguments(shim_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS}")
string(TOUPPER "${CMAKE_BUILD_TYPE}" build_type)
separate_arguments(build_flags NATIVE_COMMAND "${CMAKE_CXX_FLAGS_${build_type}}")
list(APPEND shim_flags ${build_flags})
if(CMAKE_CXX_COMPILER_TARGET)
  list(APPEND shim_flags "--target=${CMAKE_CXX_COMPILER_TARGET}")
endif()
if(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN)
  list(APPEND shim_flags "--gcc-toolchain=${CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN}")
endif()
if(CMAKE_SYSROOT)
  list(APPEND shim_flags "--sysroot=${CMAKE_SYSROOT}")
endif()
list(TRANSFORM shim_flags PREPEND "--cxx-flag=")
add_library(bj47_kernel_runtime SHARED Runtime.c
  "${repo}/runtime/arena/BuddyArena.c" "${repo}/runtime/threadpool/BuddyThreadPool.c"
  "${repo}/runtime/spacemit/BuddySpacemitTcm.c")
target_compile_definitions(bj47_kernel_runtime PRIVATE BUDDY_COMPONENT_RUNTIME BUDDY_ARENA_DEFAULT_PREFAULT_MB=${prefault})
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
      --buddy-opt "${BJ47_BUDDY_OPT}" --llvm "${BJ47_LLVM_TOOLS}" --cxx "${CMAKE_CXX_COMPILER}" --crunner "${BJ47_CRUNNER}" ${shim_flags}
    DEPENDS bj47_kernel_runtime ${mlir} "${BJ47_ROUTE_B_BUNDLE}/components.json"
      "${codegen}/build_route_b.py" "${codegen}/route_b_lower.py" "${codegen}/compile_pipeline.py"
      K3ExpertDot.s K3ExpertTile.cpp Shim.h "${repo}/runtime/include/buddy/runtime/pipeline/RouteB.h"
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
  add_test(NAME test_route_b_tp_tiles COMMAND "${Python3_EXECUTABLE}"
    "${repo}/tests/Pipeline/test_route_b_tp_tiles.py")
  set_tests_properties(test_route_b_tp_tiles test_route_b_counts_and_roles test_route_b_pack_layout
    PROPERTIES LABELS route-b TIMEOUT 240)
endif()

option(BJ47_ROUTE_B_HOST "Also build the MPI Route B host" OFF)
if(BJ47_ROUTE_B_HOST)
  add_subdirectory(host)
endif()
