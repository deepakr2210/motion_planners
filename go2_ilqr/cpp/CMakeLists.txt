cmake_minimum_required(VERSION 3.16)
project(go2_ilqr LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build type default
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

# MuJoCo
# Set MUJOCO_ROOT to your MuJoCo install prefix, e.g.:
#   cmake -DMUJOCO_ROOT=/usr/local/mujoco ..
# Or install via apt / brew and rely on find_package.
find_package(mujoco QUIET)

if(NOT mujoco_FOUND)
  # Fall back: look for a local MuJoCo install
  if(NOT DEFINED MUJOCO_ROOT)
    set(MUJOCO_ROOT "$ENV{HOME}/.mujoco/mujoco-3.2.0" CACHE PATH
        "Path to MuJoCo installation")
  endif()

  message(STATUS "Using MUJOCO_ROOT = ${MUJOCO_ROOT}")

  find_library(MUJOCO_LIB
    NAMES mujoco
    HINTS "${MUJOCO_ROOT}/lib" "${MUJOCO_ROOT}/bin"
    REQUIRED)

  add_library(mujoco::mujoco SHARED IMPORTED)
  set_target_properties(mujoco::mujoco PROPERTIES
    IMPORTED_LOCATION "${MUJOCO_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${MUJOCO_ROOT}/include")
endif()

# Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------

add_executable(go2_ilqr_demo
  main.cpp
  src/ilqr.cpp
  src/go2_task.cpp
)

target_include_directories(go2_ilqr_demo PRIVATE
  include
)

target_link_libraries(go2_ilqr_demo PRIVATE
  mujoco::mujoco
  Eigen3::Eigen
)

target_compile_options(go2_ilqr_demo PRIVATE
  -O3
  -march=native
  -Wall
  -Wextra
  -Wno-unused-parameter
)

# ---------------------------------------------------------------------------
# Optional: unit tests (requires GTest)
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" OFF)

if(BUILD_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()

  add_executable(go2_ilqr_test
    tests/test_ilqr.cpp
    src/ilqr.cpp
    src/go2_task.cpp
  )
  target_include_directories(go2_ilqr_test PRIVATE include)
  target_link_libraries(go2_ilqr_test PRIVATE
    mujoco::mujoco Eigen3::Eigen GTest::gtest_main)
  add_test(NAME go2_ilqr_test COMMAND go2_ilqr_test)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS go2_ilqr_demo DESTINATION bin)
install(FILES ${CMAKE_SOURCE_DIR}/../../assets/mujoco_menagerie/unitree_go2/go2.xml DESTINATION models)

message(STATUS "Build configuration:")
message(STATUS "  Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX std     : ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen       : ${EIGEN3_INCLUDE_DIR}")
