//===- TensorOpsOps.cpp - TensorOps operation definitions ----------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"

#define GET_OP_CLASSES
#include "TensorOpsOps.cpp.inc"

//===----------------------------------------------------------------------===//
// MatmulOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult tensor::MatmulOp::verify() {
  auto lhsType = getLhs().getType().dyn_cast<mlir::RankedTensorType>();
  auto rhsType = getRhs().getType().dyn_cast<mlir::RankedTensorType>();

  if (!lhsType || !rhsType)
    return emitOpError("operands must be ranked tensors");

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("matmul requires 2D tensors");

  // Check inner dimensions match: lhs[K] == rhs[K]
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return emitOpError("shape mismatch: lhs columns (")
           << lhsType.getDimSize(1) << ") != rhs rows (" << rhsType.getDimSize(0)
           << ")";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ReluOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult tensor::ReluOp::verify() {
  auto inputType = getInput().getType().dyn_cast<mlir::RankedTensorType>();
  if (!inputType)
    return emitOpError("input must be a ranked tensor");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult tensor::AddOp::verify() {
  auto lhsType = getLhs().getType().dyn_cast<mlir::RankedTensorType>();
  auto rhsType = getRhs().getType().dyn_cast<mlir::RankedTensorType>();

  if (!lhsType || !rhsType)
    return emitOpError("operands must be ranked tensors");

  if (lhsType.getShape() != rhsType.getShape())
    return emitOpError("shape mismatch: ")
           << lhsType.getShape() << " != " << rhsType.getShape();

  return mlir::success();
}
