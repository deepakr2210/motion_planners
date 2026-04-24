cmake_minimum_required(VERSION 3.28)
project(search_planning_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── matplotlib-cpp ─────────────────────────────────────────────────────────────
# Header-only C++ wrapper around matplotlib. FetchContent_Populate downloads
# it without running its CMakeLists (avoids building its example binaries).
include(FetchContent)
FetchContent_Declare(
    matplotlibcpp
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        master
    EXCLUDE_FROM_ALL   # skip building matplotlibcpp's own examples
)
FetchContent_MakeAvailable(matplotlibcpp)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# ── Helper: one self-contained .cpp per algorithm ──────────────────────────────
function(add_algo name)
    add_executable(${name} Search_2D/${name}.cpp)
    target_include_directories(${name} PRIVATE
        ${matplotlibcpp_SOURCE_DIR}
        ${Python3_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/Search_2D   # so #include "common.cpp" resolves
    )
    target_link_libraries(${name} PRIVATE ${Python3_LIBRARIES})
endfunction()

# ── Implemented ────────────────────────────────────────────────────────────────
add_algo(astar)

# ── Stubs — uncomment as you implement each one ────────────────────────────────
# add_algo(dijkstra)
# add_algo(best_first)
# add_algo(bfs)
# add_algo(dfs)
# add_algo(bi_astar)
# add_algo(arastar)
# add_algo(lpastar)
# add_algo(lrtastar)
# add_algo(rtaastar)
# add_algo(dstar)
# add_algo(dstar_lite)
# add_algo(anytime_dstar)
