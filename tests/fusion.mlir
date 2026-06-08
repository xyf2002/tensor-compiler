// RUN: tensor-opt %s -fuse-matmul-relu -tensor-to-linalg -split-input-file | FileCheck %s

// Positive test: matmul followed by relu should be fused into a single
// linalg.generic contraction that computes matmul + max(0, x).
// CHECK-LABEL: func @test_fusion
func.func @test_fusion(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // After fusion + lowering, a single linalg.generic with reduction
  // (the contraction) that yields max(0, acc) instead of plain acc.
  // CHECK: linalg.generic
  // CHECK-SAME: reduction
  // CHECK: arith.mulf
  // CHECK: arith.addf
  // CHECK: arith.cmpf ogt
  // CHECK: arith.select
  // CHECK: linalg.yield
  // CHECK-NOT: linalg.matmul
  %0 = ten.matmul %a, %b : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  %1 = ten.relu %0 : tensor<4x16xf32> -> tensor<4x16xf32>
  func.return %1 : tensor<4x16xf32>
}

// -----

// Negative test: multi-use matmul should NOT fuse
// CHECK-LABEL: func @test_no_fusion_multi_use
func.func @test_no_fusion_multi_use(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> (tensor<4x16xf32>, tensor<4x16xf32>) {
  // Matmul result feeds both relu and add — fusion would break the add
  // CHECK: linalg.matmul
  // CHECK: linalg.generic
  // CHECK: linalg.add
  %0 = ten.matmul %a, %b : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  %1 = ten.relu %0 : tensor<4x16xf32> -> tensor<4x16xf32>
  %2 = ten.add %0, %0 : tensor<4x16xf32>, tensor<4x16xf32> -> tensor<4x16xf32>
  func.return %1, %2 : tensor<4x16xf32>, tensor<4x16xf32>
}
