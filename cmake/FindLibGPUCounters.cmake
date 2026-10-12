# FindLibGPUCounters
# ------------------
#
# Build Arm's libGPUCounters from a pinned upstream source revision.
#
# libGPUCounters reads Arm Mali hardware counters through the kernel driver, so it builds and runs
# on Android and Linux only, and it needs an Arm commercial driver at runtime.
#
# Input Variables
# ^^^^^^^^^^^^^^^
#
# ``GFXRECON_LIBGPUCOUNTERS_SOURCE_DIR`` : Use this local checkout instead of fetching the pinned
# revision. Useful on a build machine without network access.
#
# Targets
# ^^^^^^^
#
# This module defines ``LibGPUCounters::HWCPipe`` as an ALIAS for the ``hwcpipe`` target. That
# target carries the ``device`` backend as a public dependency, so linking it is enough.
#
# Result Variables
# ^^^^^^^^^^^^^^^^
#
# This module defines the following variables:
#
#  LibGPUCounters_FOUND : True if libGPUCounters was found.
if(TARGET LibGPUCounters::HWCPipe)
    set(LibGPUCounters_FOUND TRUE)
    return()
endif()

set(GFXRECON_LIBGPUCOUNTERS_REPOSITORY "https://github.com/ARM-software/libGPUCounters.git")
set(GFXRECON_LIBGPUCOUNTERS_VERSION    "2.5.0")
set(GFXRECON_LIBGPUCOUNTERS_TAG        "378c7b59e2dae87cbbffb0ac9cf96ed13c939823")

include(FetchContent)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

# The plugin is a shared module, so every object linked into it must be position independent.
set(HWCPIPE_PIC ON)
set(HWCPIPE_BUILD_EXAMPLES OFF)
set(HWCPIPE_FRONTEND_ENABLE_TESTS OFF)

set(HWCPIPE_WERROR OFF)

if(GFXRECON_LIBGPUCOUNTERS_SOURCE_DIR)
    message(STATUS "Building libGPUCounters from ${GFXRECON_LIBGPUCOUNTERS_SOURCE_DIR}")
    add_subdirectory(${GFXRECON_LIBGPUCOUNTERS_SOURCE_DIR} libgpucounters EXCLUDE_FROM_ALL)
else()
    message(STATUS "Building libGPUCounters ${GFXRECON_LIBGPUCOUNTERS_VERSION} from source "
                   "(${GFXRECON_LIBGPUCOUNTERS_REPOSITORY})")
    FetchContent_Declare(libgpucounters
                         GIT_REPOSITORY ${GFXRECON_LIBGPUCOUNTERS_REPOSITORY}
                         GIT_TAG        ${GFXRECON_LIBGPUCOUNTERS_TAG})
    FetchContent_MakeAvailable(libgpucounters)
endif()

if(NOT TARGET hwcpipe)
    message(FATAL_ERROR
            "libGPUCounters was fetched but the expected 'hwcpipe' target was not created. The "
            "upstream CMake configuration may have changed; see cmake/FindLibGPUCounters.cmake.")
endif()

add_library(LibGPUCounters::HWCPipe ALIAS hwcpipe)

set(LibGPUCounters_FOUND TRUE)
