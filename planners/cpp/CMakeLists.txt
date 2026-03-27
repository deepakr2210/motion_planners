cmake_minimum_required(VERSION 3.16)
project(cpp_planner LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────

# libzmq (system)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

# cppzmq — header-only C++ bindings for libzmq
include(FetchContent)

FetchContent_Declare(cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG        v4.10.0
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cppzmq)

# nlohmann/json — header-only JSON library
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Executable ────────────────────────────────────────────────────────────────

add_executable(cpp_planner
    src/main.cpp
)

target_include_directories(cpp_planner PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIRS}
)

target_link_libraries(cpp_planner PRIVATE
    cppzmq
    nlohmann_json::nlohmann_json
    ${ZMQ_LIBRARIES}
)

target_compile_options(cpp_planner PRIVATE
    -O2 -Wall -Wextra -Wpedantic
)

# ── Install ───────────────────────────────────────────────────────────────────

install(TARGETS cpp_planner RUNTIME DESTINATION bin)
