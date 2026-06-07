//===- tensor-runner.cpp - JIT runner for TensorOps dialect ---------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// Parses MLIR containing ten.matmul, applies the full lowering pipeline,
// writes the lowered LLVM-dialect module to a temp file, translates it to
// LLVM IR via mlir-translate, and JIT-executes it via lli.
//
// The input MLIR is augmented with a @main() wrapper that generates random
// input data (embedded as arith.constant dense), calls the target function,
// and performs a side-effecting store to prevent dead-code elimination.
//
// Usage:
//   tensor-runner test.mlir
//   tensor-runner --benchmark --warmup=2 --iterations=10 test.mlir
//
//===----------------------------------------------------------------------===//

#include "TensorOps/TensorOpsDialect.h"
#include "TensorOps/TensorOpsPasses.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"

#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <unistd.h>
#include <vector>

using namespace mlir;

// CLI options
static llvm::cl::opt<std::string> inputFilename(
    llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::init("-"));

static llvm::cl::opt<std::string> entryFunc(
    "e", llvm::cl::desc("Function to invoke"),
    llvm::cl::value_desc("<function name>"), llvm::cl::init("test"));

static llvm::cl::opt<int> optIterations(
    "iterations", llvm::cl::desc("Number of benchmark iterations"),
    llvm::cl::init(10));

static llvm::cl::opt<int> optWarmup(
    "warmup", llvm::cl::desc("Number of warmup iterations"),
    llvm::cl::init(0));

static llvm::cl::opt<bool> optBenchmark(
    "benchmark", llvm::cl::desc("Run in benchmark mode with timing"),
    llvm::cl::init(false));

static OwningOpRef<ModuleOp> parseMLIR(const std::string &filename,
                                       MLIRContext &context) {
  std::string error;
  auto file = openInputFile(filename, &error);
  if (!file) {
    llvm::errs() << "Error: " << error << "\n";
    return nullptr;
  }
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, ParserConfig(&context));
}

static LogicalResult buildLoweringPipeline(PassManager &pm) {
  pm.addPass(ten::createFuseMatmulReluPass());
  pm.addPass(ten::createLowerToLinalgPass());

  bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  bufOpts.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::InferLayoutMap);
  pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

  pm.addPass(bufferization::createBufferResultsToOutParamsPass());
  pm.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  pm.addPass(createConvertLinalgToLoopsPass());
  pm.addPass(createConvertSCFToCFPass());
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(createArithToLLVMConversionPass());
  pm.addPass(createConvertControlFlowToLLVMPass());
  pm.addPass(createConvertFuncToLLVMPass());
  pm.addPass(createReconcileUnrealizedCastsPass());
  return success();
}

// Build a @main() wrapper MLIR string that creates input data via
// arith.constant dense and calls the target function.
static std::string buildWrapper(int64_t M, int64_t K, int64_t N,
                                 const std::string &funcName,
                                 const float *aData, const float *bData) {
  auto fmt = [](float v) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.8e", (double)v);
    return std::string(buf);
  };

  std::string aDim = std::to_string(M) + "x" + std::to_string(K);
  std::string bDim = std::to_string(K) + "x" + std::to_string(N);
  std::string cDim = std::to_string(M) + "x" + std::to_string(N);

  std::string s;
  s += "func.func @main() -> i32 {\n";
  s += "  %c0 = arith.constant 0 : index\n";

  // Tensor A
  s += "  %a = arith.constant dense<[";
  for (int64_t i = 0; i < M; ++i) {
    if (i > 0) s += ", ";
    s += "[";
    for (int64_t j = 0; j < K; ++j) {
      if (j > 0) s += ", ";
      s += fmt(aData[i * K + j]);
    }
    s += "]";
  }
  s += "]> : tensor<" + aDim + "xf32>\n";

  // Tensor B
  s += "  %b = arith.constant dense<[";
  for (int64_t i = 0; i < K; ++i) {
    if (i > 0) s += ", ";
    s += "[";
    for (int64_t j = 0; j < N; ++j) {
      if (j > 0) s += ", ";
      s += fmt(bData[i * N + j]);
    }
    s += "]";
  }
  s += "]> : tensor<" + bDim + "xf32>\n";

  // Call original function
  s += "  %c = call @" + funcName + "(%a, %b) : (tensor<" + aDim
       + "xf32>, tensor<" + bDim + "xf32>) -> tensor<" + cDim + "xf32>\n";

  // Side-effecting store to prevent DCE
  s += "  %m = bufferization.to_memref %c : tensor<" + cDim + "xf32> to memref<" + cDim + "xf32>\n";
  s += "  %v = memref.load %m[%c0, %c0] : memref<" + cDim + "xf32>\n";
  s += "  memref.store %v, %m[%c0, %c0] : memref<" + cDim + "xf32>\n";
  s += "  %ret = arith.constant 0 : i32\n";
  s += "  return %ret : i32\n";
  s += "}\n";

  return s;
}

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "TensorOps dialect CPU runner\n");

  DialectRegistry registry;
  registerAllDialects(registry);
  registry.insert<ten::TensorOpsDialect>();
  MLIRContext context(registry);

  // Read the original input file content (for re-parsing after adding wrapper)
  std::string error;
  auto file = openInputFile(inputFilename, &error);
  if (!file) {
    llvm::errs() << "Error: " << error << "\n";
    return 1;
  }
  std::string originalContent(file->getBuffer().data(),
                              file->getBuffer().size());

  // Parse original module to extract function info
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(file), SMLoc());
  auto module = parseSourceFile<ModuleOp>(sourceMgr, ParserConfig(&context));
  if (!module)
    return 1;

  auto origFuncOp = dyn_cast_or_null<func::FuncOp>(
      SymbolTable::lookupSymbolIn(module.get(), entryFunc));
  if (!origFuncOp) {
    llvm::errs() << "Error: function '" << entryFunc
                 << "' not found in module\n";
    return 1;
  }
  auto origType = origFuncOp.getFunctionType();
  if (origType.getNumInputs() < 2) {
    llvm::errs() << "Error: expected at least 2 tensor arguments, got "
                 << origType.getNumInputs() << "\n";
    return 1;
  }
  auto tensorType = dyn_cast<RankedTensorType>(origType.getInput(0));
  auto tensorType2 = dyn_cast<RankedTensorType>(origType.getInput(1));
  if (!tensorType || !tensorType.getElementType().isF32() ||
      tensorType.getRank() != 2) {
    llvm::errs() << "Error: expected first arg to be tensor<AxBxf32>\n";
    return 1;
  }
  int64_t M = tensorType.getDimSize(0);
  int64_t K = tensorType.getDimSize(1);
  int64_t N = tensorType2.getDimSize(1);
  if (M == ShapedType::kDynamic || K == ShapedType::kDynamic ||
      N == ShapedType::kDynamic) {
    llvm::errs() << "Error: dynamic dimensions not supported\n";
    return 1;
  }

  // Generate random data
  std::vector<float> aData(M * K), bData(K * N);
  {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &v : aData) v = dist(rng);
    for (auto &v : bData) v = dist(rng);
  }

  // Build wrapper with embedded data, combine with original, and re-parse
  std::string wrapperStr = buildWrapper(M, K, N, entryFunc,
                                        aData.data(), bData.data());
  std::string combinedStr = wrapperStr + "\n" + originalContent;
  auto combinedModule = parseSourceString<ModuleOp>(combinedStr, &context);
  if (!combinedModule) {
    llvm::errs() << "Error: failed to parse combined module\n";
    return 1;
  }

  // Run the lowering pipeline
  PassManager pm(&context, "builtin.module");
  if (failed(buildLoweringPipeline(pm))) {
    llvm::errs() << "Error: lowering pipeline failed\n";
    return 1;
  }
  if (failed(pm.run(combinedModule.get()))) {
    llvm::errs() << "Error: pass pipeline execution failed\n";
    return 1;
  }

  llvm::dbgs() << "=== Lowered module ===\n";
  combinedModule.get()->dump();
  llvm::dbgs() << "\n";

  // Write lowered module to temp .mlir file
  std::string tempMlir = "/tmp/tensor_runner_" +
                          std::to_string(::getpid()) +
                         ".mlir";
  std::string llFile = "/tmp/tensor_runner_" +
                       std::to_string(::getpid()) +
                       ".ll";

  {
    std::error_code ec;
    llvm::raw_fd_ostream out(tempMlir, ec);
    if (ec) {
      llvm::errs() << "Error: cannot write " << tempMlir << ": "
                   << ec.message() << "\n";
      return 1;
    }
    combinedModule.get()->print(out);
  }

  // Translate to LLVM IR via mlir-translate
  std::string translateCmd =
      "/usr/lib/llvm-20/bin/mlir-translate --mlir-to-llvmir " + tempMlir +
      " > " + llFile + " 2>/dev/null";
  int translateRet = system(translateCmd.c_str());
  if (translateRet != 0) {
    llvm::errs() << "Error: mlir-translate failed (exit " << translateRet
                 << ")\n";
    std::remove(tempMlir.c_str());
    return 1;
  }

  // Build the lli command (no optimization to keep constant-folding from
  // eliminating the computation when the JIT sees compile-time-known data)
  std::string runCmd =
      "/usr/lib/llvm-20/bin/lli -O0 " + llFile + " 2>/dev/null";

  // Warmup
  for (int i = 0; i < optWarmup; ++i) {
    int ret = system(runCmd.c_str());
    if (ret != 0) {
      llvm::errs() << "Error: lli failed during warmup (exit " << ret << ")\n";
      std::remove(tempMlir.c_str());
      std::remove(llFile.c_str());
      return 1;
    }
  }

  // Benchmark or single-run
  if (optBenchmark) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < optIterations; ++i) {
      int ret = system(runCmd.c_str());
      if (ret != 0) {
        llvm::errs() << "Error: lli failed at iteration " << i
                     << " (exit " << ret << ")\n";
        std::remove(tempMlir.c_str());
        std::remove(llFile.c_str());
        return 1;
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    double avgUs = static_cast<double>(elapsed) / optIterations;
    llvm::outs() << "Benchmark: " << optIterations << " iters, "
                 << avgUs << " us/iter, total " << elapsed << " us\n";
  } else {
    auto start = std::chrono::steady_clock::now();
    int ret = system(runCmd.c_str());
    auto end = std::chrono::steady_clock::now();
    if (ret != 0) {
      llvm::errs() << "Error: lli failed (exit " << ret << ")\n";
      std::remove(tempMlir.c_str());
      std::remove(llFile.c_str());
      return 1;
    }
    auto elapsed =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    llvm::outs() << "Execution time: " << elapsed << " us\n";
  }

  // Cleanup temp files
  std::remove(tempMlir.c_str());
  std::remove(llFile.c_str());

  return 0;
}
