// RUN: tensor-opt %s -tensor-to-linalg -lower-tensor-ops-to-gpu-nvvm -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @test_gpu_matmul
func.func @test_gpu_matmul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: gpu.module
  // CHECK: nvvm.read.ptx.sreg.ctaid.x
  // CHECK: nvvm.read.ptx.sreg.ctaid.y
  %0 = ten.matmul %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// -----

// CHECK-LABEL: func.func @test_gpu_relu
func.func @test_gpu_relu(%x: tensor<4x4xf32>) -> tensor<4x4xf32> {
  // CHECK: gpu.module
  // CHECK: nvvm.read.ptx.sreg.ctaid.x
  %0 = ten.relu %x : tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}
