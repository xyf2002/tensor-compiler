//===-- TensorOpsDialect.td - TensorOps dialect definition -*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_OPS_DIALECT
#define TENSOR_OPS_DIALECT

include "mlir/IR/DialectBase.td"

def TensorOpsDialect : Dialect {
  let name = "tensor";
  let cppNamespace = "::tensor";
  let summary = "A dialect for basic tensor operations";
  let description = [{
    The TensorOps dialect provides basic tensor operations:
    matmul, relu, and add. It is designed as a high-level dialect
    that is lowered to Linalg for further optimization.
  }];
}

#endif // TENSOR_OPS_DIALECT
