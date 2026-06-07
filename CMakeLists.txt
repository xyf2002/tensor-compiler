cmake_minimum_required(VERSION 3.20)
project(TensorCompiler VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# MLIR/LLVM integration
find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "MLIR version: ${MLIR_VERSION}")
message(STATUS "LLVM version: ${LLVM_VERSION}")

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)

list(APPEND CMAKE_MODULE_PATH "${MLIR_DIR}")
include(AddMLIR)

include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Subdirectories
add_subdirectory(lib)
add_subdirectory(tools/tensor-opt)
add_subdirectory(tools/tensor-runner)
add_subdirectory(tests)
