//===- TensorOpsToLinalg.cpp - TensorOps to Linalg conversion -------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsOps.h"
#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// MatmulOp -> linalg.matmul
//===----------------------------------------------------------------------===//

struct MatmulOpLowering : public OpRewritePattern<tensor::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType();
    auto rhsType = op.getRhs().getType();
    auto resultType = op.getResult().getType();

    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, resultType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// AddOp -> linalg.elemwise_broadcast (generic add)
//===----------------------------------------------------------------------===//

struct AddOpLowering : public OpRewritePattern<tensor::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto resultType = op.getResult().getType();
    rewriter.replaceOpWithNewOp<linalg::AddOp>(
        op, resultType, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReluOp -> linalg.generic { max(0, x) }
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public OpRewritePattern<tensor::ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto input = op.getInput();
    auto resultType = op.getResult().getType();

    // Build linalg.generic for element-wise max(0, x)
    auto loc = op.getLoc();
    auto inputType = input.getType().cast<RankedTensorType>();
    SmallVector<AffineMap> indexMaps = {
        AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, input, ValueRange{}, indexMaps, iteratorTypes,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value zero = b.create<arith::ConstantOp>(loc, b.getF32FloatAttr(0.0f));
          Value cmp = b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT,
                                              args[0], zero);
          Value result = b.create<arith::SelectOp>(loc, cmp, args[0], zero);
          b.create<linalg::YieldOp>(loc, result);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// TensorOpsToLinalgPass
//===----------------------------------------------------------------------===//

struct TensorOpsToLinalgPass
    : public PassWrapper<TensorOpsToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorOpsToLinalgPass)

  StringRef getArgument() const override { return "tensor-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower TensorOps dialect to Linalg on tensors";
  }

  void runOnOperation() override {
    auto *ctx = &getContext();
    ConversionTarget target(*ctx);

    target.addLegalDialect<arith::ArithDialect,
                          linalg::LinalgDialect,
                          tensor::TensorDialect>();
    target.addIllegalDialect<tensor::TensorOpsDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulOpLowering, AddOpLowering, ReluOpLowering>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> tensor::createLowerToLinalgPass() {
  return std::make_unique<TensorOpsToLinalgPass>();
}
