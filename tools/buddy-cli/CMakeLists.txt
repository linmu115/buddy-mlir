# tools/buddy-cli/CMakeLists.txt

add_executable(buddy-cli buddy-cli.cpp)

target_compile_features(buddy-cli PRIVATE cxx_std_17)
target_compile_definitions(buddy-cli PRIVATE
  BUDDY_VERSION="${BUDDY_PACKAGE_VERSION}")

# buddy_models_<name> is produced by model subdirectories under models/.
target_link_libraries(buddy-cli PRIVATE
  buddy_runtime_core
  buddy_runtime_llm
  buddy_runtime_rax_dependencies
  CLI11
  ${CMAKE_DL_LIBS}
)

if(BUDDY_BUILD_DEEPSEEK_R1_MODEL)
  target_link_libraries(buddy-cli PRIVATE buddy_models_deepseek_r1)
  target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_DEEPSEEK_R1_MODEL)
endif()

if(BUDDY_BUILD_LLAMA31_TT_MODEL OR BUDDY_BUILD_LLAMA32_TT_MODEL)
  target_link_libraries(buddy-cli PRIVATE buddy_models_llama31_tt)
  target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_LLAMA31_TT_MODEL)
endif()

if(BUDDY_BUILD_BGE_M3_MODEL)
  target_link_libraries(buddy-cli PRIVATE buddy_models_bge_m3)
endif()

# ── libnuma (optional but strongly recommended on NUMA systems) ────────────
# A cross build would find the host's libnuma (another architecture) in the
# host's prefixes: there it is off unless asked for, and then taken from the
# target's sysroot only (CMAKE_SYSROOT or CMAKE_FIND_ROOT_PATH).
if(CMAKE_CROSSCOMPILING)
  set(_buddy_cli_numa_default OFF)
else()
  set(_buddy_cli_numa_default ON)
endif()
option(BUDDY_CLI_ENABLE_NUMA
  "Use libnuma in buddy-cli when it is found (--numa-* flags)"
  ${_buddy_cli_numa_default})
if(BUDDY_CLI_ENABLE_NUMA)
  if(CMAKE_CROSSCOMPILING)
    if(NOT CMAKE_SYSROOT AND NOT CMAKE_FIND_ROOT_PATH)
      message(FATAL_ERROR
        "BUDDY_CLI_ENABLE_NUMA in a cross build takes libnuma from the "
        "target's sysroot: set CMAKE_SYSROOT (or CMAKE_FIND_ROOT_PATH).")
    endif()
    find_library(NUMA_LIB numa ONLY_CMAKE_FIND_ROOT_PATH)
    find_path(NUMA_INCLUDE_DIR numa.h ONLY_CMAKE_FIND_ROOT_PATH)
    if(NOT NUMA_LIB OR NOT NUMA_INCLUDE_DIR)
      message(FATAL_ERROR
        "BUDDY_CLI_ENABLE_NUMA: libnuma (numa.h, libnuma) is not in the "
        "target's sysroot.")
    endif()
    target_include_directories(buddy-cli PRIVATE ${NUMA_INCLUDE_DIR})
  else()
    find_library(NUMA_LIB numa)
  endif()
  if(NUMA_LIB)
    target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_NUMA)
    target_link_libraries(buddy-cli PRIVATE ${NUMA_LIB})
  endif()
endif()

# ── MPI (optional, enables --stage-model pipeline-parallel launches) ───────
if(CMAKE_CROSSCOMPILING)
  set(_buddy_cli_mpi_default OFF)
else()
  set(_buddy_cli_mpi_default ON)
endif()
option(BUDDY_CLI_ENABLE_MPI
  "Enable MPI pipeline-parallel support in buddy-cli (--stage-model)"
  ${_buddy_cli_mpi_default})
set(BUDDY_CLI_MPI_DIR "" CACHE PATH
  "Optional target MPI installation prefix (include/mpi.h, lib/libmpi.so) for cross-compiling buddy-cli")
if(BUDDY_CLI_ENABLE_MPI)
  if(BUDDY_CLI_MPI_DIR)
    set(_buddy_cli_mpi_inc "${BUDDY_CLI_MPI_DIR}/include")
    set(_buddy_cli_mpi_lib "${BUDDY_CLI_MPI_DIR}/lib/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}")
    if(NOT EXISTS "${_buddy_cli_mpi_inc}/mpi.h" OR NOT EXISTS "${_buddy_cli_mpi_lib}")
      message(FATAL_ERROR
        "[buddy-cli] BUDDY_CLI_MPI_DIR='${BUDDY_CLI_MPI_DIR}' must contain "
        "include/mpi.h and lib/libmpi${CMAKE_SHARED_LIBRARY_SUFFIX}.")
    endif()
    target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_MPI)
    target_include_directories(buddy-cli PRIVATE "${_buddy_cli_mpi_inc}")
    target_link_libraries(buddy-cli PRIVATE "${_buddy_cli_mpi_lib}")
    message(STATUS "[buddy-cli] Using target MPI from ${BUDDY_CLI_MPI_DIR}; enabling --stage-model support.")
  else()
    find_package(MPI COMPONENTS CXX QUIET)
    if(MPI_CXX_FOUND)
      target_compile_definitions(buddy-cli PRIVATE BUDDY_CLI_HAVE_MPI)
      target_link_libraries(buddy-cli PRIVATE MPI::MPI_CXX)
      message(STATUS "[buddy-cli] MPI found; enabling --stage-model support.")
    else()
      message(STATUS "[buddy-cli] MPI not found; building without --stage-model MPI support.")
    endif()
  endif()
else()
  message(STATUS "[buddy-cli] BUDDY_CLI_ENABLE_MPI=OFF; building without --stage-model MPI support.")
endif()

if(BUDDY_MLIR_USE_MIMALLOC)
  target_link_libraries(buddy-cli PRIVATE mimalloc)
endif()

install(TARGETS buddy-cli
        EXPORT BuddyMLIRTargets
        COMPONENT buddy_tools)
