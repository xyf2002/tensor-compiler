add_library(TensorOps STATIC
  TensorOpsDialect.cpp
  TensorOpsOps.cpp
)

target_link_libraries(TensorOps PUBLIC MLIR)
target_include_directories(TensorOps PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# TableGen for ODS
set(LLVM_TARGET_DEFINITIONS TensorOpsOps.td)
mlir_tablegen(TensorOpsOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOpsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TensorOpsOpsIncGen)
add_dependencies(TensorOps TensorOpsOpsIncGen)
