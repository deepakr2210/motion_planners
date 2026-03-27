#!/usr/bin/env bash
# Install system-level C++ dependencies for the cpp_planner.
# Run once before building: bash scripts/setup_cpp_deps.sh

set -euo pipefail

echo "==> Installing C++ build dependencies..."
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libzmq3-dev   # libzmq — ZeroMQ system library

echo ""
echo "==> Dependencies installed."
echo "==> Build the C++ planner with:"
echo "      mkdir -p build && cd build"
echo "      cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "      make -j\$(nproc)"
echo "      cd .."
