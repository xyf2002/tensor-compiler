// RUN: tensor-opt %s -fuse-matmul-relu -tensor-to-linalg -split-input-file | FileCheck %s

// Positive test: matmul followed by relu should be fused (relu absorbed)
// CHECK-LABEL: func @test_fusion
func.func @test_fusion(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // After fusion + lowering, only linalg.matmul remains (relu is absorbed)
  // CHECK: linalg.matmul
  // CHECK-NOT: linalg.generic
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
