//===- TensorOpsToLinalg.cpp - TensorOps to Linalg conversion -------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsOps.h"
#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;

/// Create an empty tensor (init) of the given type.
static Value createEmptyTensor(PatternRewriter &rewriter, Location loc,
                                RankedTensorType type) {
  SmallVector<Value> dynSizes;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i))
      dynSizes.push_back(rewriter.create<tensor::DimOp>(loc, Value(), i));
  }
  return rewriter.create<tensor::EmptyOp>(loc, type.getShape(),
                                          type.getElementType(), dynSizes);
}

namespace {

struct MatmulOpLowering : public OpRewritePattern<ten::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ten::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, resultType, ValueRange{op.getLhs(), op.getRhs()}, init);
    return success();
  }
};

struct AddOpLowering : public OpRewritePattern<ten::AddOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ten::AddOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    rewriter.replaceOpWithNewOp<linalg::AddOp>(
        op, resultType, ValueRange{op.getLhs(), op.getRhs()}, init);
    return success();
  }
};

struct ReluOpLowering : public OpRewritePattern<ten::ReluOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ten::ReluOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());

    Value init = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
                                                   resultType.getElementType());
    auto inputType = mlir::cast<RankedTensorType>(input.getType());

    SmallVector<AffineMap> indexMaps = {
        AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext()),
        AffineMap::getMultiDimIdentityMap(inputType.getRank(), rewriter.getContext())};
    SmallVector<utils::IteratorType> iteratorTypes(
        inputType.getRank(), utils::IteratorType::parallel);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{input}, ValueRange{init}, indexMaps, iteratorTypes,
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

struct TensorOpsToLinalgPass
    : public PassWrapper<TensorOpsToLinalgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorOpsToLinalgPass)

  StringRef getArgument() const override { return "tensor-to-linalg"; }
  StringRef getDescription() const override {
    return "Lower TensorOps dialect to Linalg on tensors";
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);

    ctx->loadDialect<arith::ArithDialect, linalg::LinalgDialect,
                     tensor::TensorDialect, func::FuncDialect>();

    target.addLegalDialect<arith::ArithDialect,
                          linalg::LinalgDialect,
                          tensor::TensorDialect,
                          func::FuncDialect>();
    target.addIllegalDialect<ten::TensorOpsDialect>();

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulOpLowering, AddOpLowering, ReluOpLowering>(ctx);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> ten::createLowerToLinalgPass() {
  return std::make_unique<TensorOpsToLinalgPass>();
}
