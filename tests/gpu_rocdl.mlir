// RUN: %tensor-opt --tensor-to-linalg --lower-tensor-ops-to-gpu-rocdl %s | FileCheck %s --check-prefix=ROCDL

func.func @test_rocdl(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = ten.matmul %arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  func.return %0 : tensor<4x4xf32>
}

// ROCDL:      gpu.module @test_rocdl_kernel
// ROCDL:        rocdl.workgroup.id.x
