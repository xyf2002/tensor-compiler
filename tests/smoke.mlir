// RUN: tensor-opt %s -tensor-to-linalg -split-input-file | FileCheck %s

// CHECK-LABEL: func @test_matmul
func.func @test_matmul(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // CHECK: linalg.matmul
  %0 = ten.matmul %a, %b : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}

// CHECK-LABEL: func @test_relu
func.func @test_relu(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.generic
  %0 = ten.relu %x : tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func @test_add
func.func @test_add(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: linalg.add
  %0 = ten.add %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
