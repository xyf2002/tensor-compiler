// RUN: tensor-opt %s -tensor-to-linalg -lower-tensor-ops-to-llvm -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_matmul_llvm
func.func @test_matmul_llvm(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  // CHECK: llvm.call @malloc
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: llvm.store
  %0 = ten.matmul %a, %b : tensor<4x8xf32>, tensor<8x16xf32> -> tensor<4x16xf32>
  func.return %0 : tensor<4x16xf32>
}

// -----

// CHECK-LABEL: func.func @test_relu_llvm
func.func @test_relu_llvm(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: arith.cmpf
  // CHECK: arith.select
  // CHECK: llvm.store
  %0 = ten.relu %x : tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @test_add_llvm
func.func @test_add_llvm(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: llvm.getelementptr
  // CHECK: llvm.load
  // CHECK: llvm.store
  %0 = ten.add %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
