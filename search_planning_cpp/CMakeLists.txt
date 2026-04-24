cmake_minimum_required(VERSION 3.15)
project(search_planning_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── matplotlib-cpp ──────────────────────────────────────────────────────────
# Header-only C++ wrapper around Python's matplotlib. Requires Python3 + NumPy.
# No extra native libs — uses whatever matplotlib is already installed.
include(FetchContent)
FetchContent_Declare(
    matplotlibcpp
    GIT_REPOSITORY https://github.com/lava/matplotlib-cpp.git
    GIT_TAG        master
)
FetchContent_MakeAvailable(matplotlibcpp)

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# ── Core library (env + visualization) ─────────────────────────────────────
add_library(planning_core STATIC
    src/env.cpp
    src/plotting.cpp
)
target_include_directories(planning_core PUBLIC
    include
    ${matplotlibcpp_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
)
# WITHOUT_NUMPY: use PyList instead of raw numpy pointers.
# Avoids dangling-pointer UB when passing temporary vectors, and numpy 2.x ABI issues.
target_compile_definitions(planning_core PUBLIC WITHOUT_NUMPY)
target_link_libraries(planning_core PUBLIC ${Python3_LIBRARIES})

# ── Algorithms ──────────────────────────────────────────────────────────────
# Add a new entry here each time you implement an algorithm.

add_executable(run_astar
    examples/run_astar.cpp
    src/algorithms/astar.cpp
)
target_link_libraries(run_astar PRIVATE planning_core)

# Stubs — uncomment and add src/algorithms/<name>.cpp when implemented:
# add_executable(run_dijkstra   examples/run_dijkstra.cpp   src/algorithms/dijkstra.cpp)
# add_executable(run_bfs        examples/run_bfs.cpp        src/algorithms/bfs.cpp)
# add_executable(run_dfs        examples/run_dfs.cpp        src/algorithms/dfs.cpp)
# add_executable(run_best_first examples/run_best_first.cpp src/algorithms/best_first.cpp)
# add_executable(run_bi_astar   examples/run_bi_astar.cpp   src/algorithms/bidirectional_astar.cpp)
# add_executable(run_arastar    examples/run_arastar.cpp    src/algorithms/arastar.cpp)
# add_executable(run_lpastar    examples/run_lpastar.cpp    src/algorithms/lpastar.cpp)
# add_executable(run_lrtastar   examples/run_lrtastar.cpp   src/algorithms/lrtastar.cpp)
# add_executable(run_rtaastar   examples/run_rtaastar.cpp   src/algorithms/rtaastar.cpp)
# add_executable(run_dstar      examples/run_dstar.cpp      src/algorithms/dstar.cpp)
# add_executable(run_dstar_lite examples/run_dstar_lite.cpp src/algorithms/dstar_lite.cpp)
# add_executable(run_ad_star    examples/run_ad_star.cpp    src/algorithms/anytime_dstar.cpp)
