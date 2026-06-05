//===- TensorOpsDialect.cpp - TensorOps dialect implementation ------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsDialect.h"
#include "TensorOps/TensorOpsOps.h"

// Dialect initialization generated from ODS
#include "TensorOpsDialect.cpp.inc"

void ten::TensorOpsDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "TensorOpsOps.cpp.inc"
  >();
}
