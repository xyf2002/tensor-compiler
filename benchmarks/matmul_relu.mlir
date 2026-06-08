func.func @matmul_relu(%a: tensor<64x64xf32>, %b: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %0 = ten.matmul %a, %b : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  %1 = ten.relu %0 : tensor<64x64xf32> -> tensor<64x64xf32>
  func.return %1 : tensor<64x64xf32>
}
