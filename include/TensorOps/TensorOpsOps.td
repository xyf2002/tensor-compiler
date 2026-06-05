//===-- TensorOpsOps.td - TensorOps operation definitions -*- tablegen -*-===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#ifndef TENSOR_OPS_OPS
#define TENSOR_OPS_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "TensorOpsDialect.td"

// Base class for TensorOps ops
class TensorOpsOp<string mnemonic, list<Trait> traits = []>
    : Op<TensorOpsDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// tensor.matmul
//===----------------------------------------------------------------------===//

def MatmulOp : TensorOpsOp<"matmul"> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication of two 2D tensors.
    Computes: C[M,N] = A[M,K] * B[K,N]
  }];

  let arguments = (ins
    RankedTensorOf<[F32]>:$lhs,
    RankedTensorOf<[F32]>:$rhs
  );

  let results = (outs
    RankedTensorOf<[F32]>:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tensor.relu
//===----------------------------------------------------------------------===//

def ReluOp : TensorOpsOp<"relu"> {
  let summary = "Rectified Linear Unit";
  let description = [{
    Applies ReLU activation element-wise.
    Computes: out = max(0, in)
  }];

  let arguments = (ins
    RankedTensorOf<[F32]>:$input
  );

  let results = (outs
    RankedTensorOf<[F32]>:$result
  );

  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// tensor.add
//===----------------------------------------------------------------------===//

def AddOp : TensorOpsOp<"add"> {
  let summary = "Element-wise tensor addition";
  let description = [{
    Adds two tensors element-wise.
    Both tensors must have the same shape.
  }];

  let arguments = (ins
    RankedTensorOf<[F32]>:$lhs,
    RankedTensorOf<[F32]>:$rhs
  );

  let results = (outs
    RankedTensorOf<[F32]>:$result
  );

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
}

#endif // TENSOR_OPS_OPS
