add_executable(tensor-runner
  tensor-runner.cpp
)
target_link_libraries(tensor-runner PRIVATE
  TensorOps
  MLIR
  MLIRExecutionEngine
  MLIRLLVMToLLVMIRTranslation
)
