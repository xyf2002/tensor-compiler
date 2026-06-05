cmake_minimum_required(VERSION 3.20)
project(TensorCompiler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MLIR/LLVM integration
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "MLIR version: ${MLIR_VERSION}")
message(STATUS "MLIR include dirs: ${MLIR_INCLUDE_DIRS}")

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Enable tablegen support
include(AddMLIR)

# Subdirectories
add_subdirectory(lib)
add_subdirectory(tools/tensor-opt)
add_subdirectory(tests)
