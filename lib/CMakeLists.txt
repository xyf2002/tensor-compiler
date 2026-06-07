# TableGen include paths
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY
  INCLUDE_DIRECTORIES
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/include/TensorOps
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)

add_library(TensorOps STATIC
  TensorOpsDialect.cpp
  TensorOpsOps.cpp
  TensorOpsPasses.cpp
  Conversion/TensorOpsToLinalg.cpp
  Fusion/FuseMatmulRelu.cpp
  LowerTensorOpsToLLVM.cpp
  LowerTensorOpsToNVVM.cpp
)

target_link_libraries(TensorOps PUBLIC
  MLIR
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSCFToGPU
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRFuncToLLVM
  MLIRLLVMDialect
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRGPUToNVVMTransforms
  MLIRNVVMDialect
  MLIRReconcileUnrealizedCasts
)

target_include_directories(TensorOps PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_CURRENT_BINARY_DIR}
)

# TableGen: dialect and ops from ODS
set(LLVM_TARGET_DEFINITIONS ${CMAKE_SOURCE_DIR}/include/TensorOps/TensorOpsOps.td)
mlir_tablegen(TensorOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(TensorOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(TensorOpsOps.h.inc -gen-op-decls)
mlir_tablegen(TensorOpsOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(TensorOpsIncGen)
add_dependencies(TensorOps TensorOpsIncGen)
