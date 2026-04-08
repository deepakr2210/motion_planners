cmake_minimum_required(VERSION 3.16)
project(cpp_diff_ik LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Dependencies ──────────────────────────────────────────────────────────────

find_package(Eigen3 REQUIRED NO_MODULE)

find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ REQUIRED libzmq)

include(FetchContent)

FetchContent_Declare(cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG        v4.10.0
)
set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cppzmq)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG        0.8.0
)
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(yaml-cpp)

# ── MuJoCo — resolved via the Python venv ────────────────────────────────────

find_package(Python3 COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c
        "import mujoco, os; print(os.path.dirname(mujoco.__file__))"
    OUTPUT_VARIABLE MUJOCO_PKG_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _mj_ret
)

if(NOT _mj_ret EQUAL 0 OR NOT MUJOCO_PKG_DIR)
    message(FATAL_ERROR
        "Could not locate the mujoco Python package.\n"
        "Activate the venv first:  source .venv/bin/activate")
endif()

set(MUJOCO_INCLUDE_DIR "${MUJOCO_PKG_DIR}/include")
find_library(MUJOCO_LIB NAMES mujoco PATHS "${MUJOCO_PKG_DIR}" NO_DEFAULT_PATH)

if(NOT MUJOCO_LIB)
    message(FATAL_ERROR "libmujoco not found in ${MUJOCO_PKG_DIR}")
endif()

message(STATUS "MuJoCo include : ${MUJOCO_INCLUDE_DIR}")
message(STATUS "MuJoCo library : ${MUJOCO_LIB}")

# ── Executable ────────────────────────────────────────────────────────────────

add_executable(cpp_diff_ik src/main.cpp)

target_include_directories(cpp_diff_ik PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ZMQ_INCLUDE_DIRS}
    ${MUJOCO_INCLUDE_DIR}
)

target_link_libraries(cpp_diff_ik PRIVATE
    cppzmq
    nlohmann_json::nlohmann_json
    yaml-cpp::yaml-cpp
    Eigen3::Eigen
    ${ZMQ_LIBRARIES}
    ${MUJOCO_LIB}
)

target_compile_options(cpp_diff_ik PRIVATE -O2 -Wall -Wextra -Wpedantic)

set_target_properties(cpp_diff_ik PROPERTIES
    BUILD_RPATH   "${MUJOCO_PKG_DIR}"
    INSTALL_RPATH "${MUJOCO_PKG_DIR}"
)
