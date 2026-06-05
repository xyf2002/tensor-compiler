//===- TensorOpsOps.h - TensorOps operation declarations -------*- C++ -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_OPS_OPS_H
#define TENSOR_OPS_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "TensorOps/TensorOpsDialect.h"

// Op classes generated from ODS
#define GET_OP_CLASSES
#include "TensorOpsOps.h.inc"

#endif // TENSOR_OPS_OPS_H
