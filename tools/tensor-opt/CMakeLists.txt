add_executable(tensor-opt
  tensor-opt.cpp
)
target_link_libraries(tensor-opt PRIVATE TensorOps MLIR)
