###############################################################################
# Copyright (c) 2026 LunarG, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

# A replay event plugin that writes each replay event together with the Arm Mali hardware counters
# accumulated since the previous event, read with libGPUCounters.

get_filename_component(GFXRECON_REPO_ROOT ${CMAKE_CURRENT_LIST_DIR}/../../.. ABSOLUTE)
list(APPEND CMAKE_MODULE_PATH ${GFXRECON_REPO_ROOT}/cmake)
include(GNUInstallDirs)

find_package(LibGPUCounters REQUIRED)
find_package(Threads REQUIRED)

set(TARGET_NAME gfxrecon_replay_event_counter_recorder)

add_library(${TARGET_NAME}
    MODULE
        ${CMAKE_CURRENT_LIST_DIR}/replay_event_counter_recorder.cpp)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        ${GFXR_REPLAY_PLUGIN_ABI_TARGET_NAME}
        gfxrecon_util
        LibGPUCounters::HWCPipe
        Threads::Threads)

if(ANDROID)
    target_link_options(${TARGET_NAME} PRIVATE "-Wl,-z,max-page-size=16384")
else()
    common_build_directives(${TARGET_NAME})
endif()

install(TARGETS ${TARGET_NAME} LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
