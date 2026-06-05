add_library(TensorOps STATIC
  TensorOpsDialect.cpp
  TensorOpsOps.cpp
  TensorOpsPasses.cpp
  Conversion/TensorOpsToLinalg.cpp
  Fusion/FuseMatmulRelu.cpp
)

target_link_libraries(TensorOps PUBLIC
  MLIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRFuncDialect
)

target_include_directories(TensorOps PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

# TableGen: dialect
set(LLVM_TARGET_DEFINITIONS TensorOpsOps.td)
mlir_tablegen(TensorOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TensorOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(TensorOpsOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOpsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TensorOpsIncGen)
add_dependencies(TensorOps TensorOpsIncGen)
