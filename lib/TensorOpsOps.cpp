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

mlir::LogicalResult ten::MatmulOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

  if (!mlir::isa<mlir::RankedTensorType>(lhsType) || !mlir::isa<mlir::RankedTensorType>(rhsType))
    return emitOpError("operands must be ranked tensors");

  auto lhsRT = mlir::cast<mlir::RankedTensorType>(lhsType);
  auto rhsRT = mlir::cast<mlir::RankedTensorType>(rhsType);

  if (lhsRT.getRank() != 2 || rhsRT.getRank() != 2)
    return emitOpError("matmul requires 2D tensors");

  if (lhsRT.getDimSize(1) != rhsRT.getDimSize(0))
    return emitOpError("shape mismatch: lhs columns (")
           << lhsRT.getDimSize(1) << ") != rhs rows (" << rhsRT.getDimSize(0)
           << ")";

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ReluOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult ten::ReluOp::verify() {
  if (!mlir::isa<mlir::RankedTensorType>(getInput().getType()))
    return emitOpError("input must be a ranked tensor");
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// AddOp verifier
//===----------------------------------------------------------------------===//

mlir::LogicalResult ten::AddOp::verify() {
  auto lhsType = getLhs().getType();
  auto rhsType = getRhs().getType();

  if (!mlir::isa<mlir::RankedTensorType>(lhsType) || !mlir::isa<mlir::RankedTensorType>(rhsType))
    return emitOpError("operands must be ranked tensors");

  auto lhsRT = mlir::cast<mlir::RankedTensorType>(lhsType);
  auto rhsRT = mlir::cast<mlir::RankedTensorType>(rhsType);

  if (lhsRT.getShape() != rhsRT.getShape())
    return emitOpError("shape mismatch: ")
           << lhsRT.getShape() << " != " << rhsRT.getShape();

  return mlir::success();
}
