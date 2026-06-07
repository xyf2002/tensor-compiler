//===- gpu-runner.cpp - GPU kernel loader and launcher --------------------===//
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
//
//===----------------------------------------------------------------------===//
//
// Usage: gpu-runner <cubin_file> <kernel_name> <matrix_size>
//
// Loads a cubin, launches the kernel with 3D grid/block dimensions computed
// from matrix_size, measures GPU time with CUDA events, and prints timing.
//
//===----------------------------------------------------------------------===//

#include <cuda.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <fstream>
#include <vector>

#define CHECK_CUDA(err, msg)                                               \
  do {                                                                     \
    if (err != CUDA_SUCCESS) {                                             \
      const char *errStr;                                                  \
      cuGetErrorString(err, &errStr);                                      \
      fprintf(stderr, "Error: %s: %s\n", msg, errStr);                     \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

static std::vector<char> readFile(const char *path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fprintf(stderr, "Error: Failed to open %s\n", path);
    exit(1);
  }
  size_t size = file.tellg();
  file.seekg(0);
  std::vector<char> buf(size);
  file.read(buf.data(), size);
  return buf;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "Usage: %s <cubin_file> <kernel_name> [matrix_size]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "Loads a cubin and launches the named kernel.\n");
    fprintf(stderr, "If matrix_size is given, grid/block dims are computed\n");
    fprintf(stderr, "for a square matmul of that size (block 16x16).\n");
    return 1;
  }

  const char *cubinPath = argv[1];
  const char *kernelName = argv[2];
  int matrixSize = (argc > 3) ? atoi(argv[3]) : 1024;

  // Initialize CUDA driver API
  CHECK_CUDA(cuInit(0), "cuInit failed (no GPU?)");

  // Get first device
  CUdevice device;
  CHECK_CUDA(cuDeviceGet(&device, 0), "cuDeviceGet failed");
  
  char name[128];
  cuDeviceGetName(name, sizeof(name), device);
  fprintf(stderr, "GPU: %s\n", name);

  // Create context
  CUcontext context;
  CHECK_CUDA(cuCtxCreate(&context, 0, device), "cuCtxCreate failed");

  // Read cubin file
  std::vector<char> cubinData = readFile(cubinPath);

  // Load module
  CUmodule module;
  CUresult modErr = cuModuleLoadData(&module, cubinData.data());
  if (modErr != CUDA_SUCCESS) {
    const char *errStr;
    cuGetErrorString(modErr, &errStr);
    fprintf(stderr, "Error: Failed to load CUDA module from %s: %s\n", cubinPath, errStr);
    cuCtxDestroy(context);
    return 1;
  }

  // Get kernel function
  CUfunction kernel;
  CHECK_CUDA(cuModuleGetFunction(&kernel, module, kernelName),
             "cuModuleGetFunction failed");
  fprintf(stderr, "Kernel: %s loaded from %s\n", kernelName, cubinPath);

  // Compute grid/block dimensions
  // For a square matmul: block is 16x16, grid covers output matrix
  const int blockSize = 16;
  int gridDim = (matrixSize + blockSize - 1) / blockSize;
  
  // Allocate device memory for inputs and output
  // For matmul C = A * B: each is matrixSize x matrixSize floats
  size_t matBytes = (size_t)matrixSize * matrixSize * sizeof(float);
  
  // Initialize host data
  std::vector<float> h_A(matrixSize * matrixSize);
  std::vector<float> h_B(matrixSize * matrixSize);
  std::vector<float> h_C(matrixSize * matrixSize, 0);

  // Fill with simple values for verification
  for (int i = 0; i < matrixSize * matrixSize; i++) {
    h_A[i] = 1.0f;
    h_B[i] = 1.0f;
  }

  // Allocate device memory
  CUdeviceptr d_A, d_B, d_C;
  CHECK_CUDA(cuMemAlloc(&d_A, matBytes), "cuMemAlloc d_A failed");
  CHECK_CUDA(cuMemAlloc(&d_B, matBytes), "cuMemAlloc d_B failed");
  CHECK_CUDA(cuMemAlloc(&d_C, matBytes), "cuMemAlloc d_C failed");

  // Copy inputs to device
  CHECK_CUDA(cuMemcpyHtoD(d_A, h_A.data(), matBytes), "cuMemcpyHtoD d_A failed");
  CHECK_CUDA(cuMemcpyHtoD(d_B, h_B.data(), matBytes), "cuMemcpyHtoD d_B failed");

  // Set kernel arguments: pointers to A, B, C, and matrix size
  // Note: kernel signature is __global__ void matmul(float* A, float* B, float* C, int N)
  void *args[4];
  args[0] = &d_A;
  args[1] = &d_B;
  args[2] = &d_C;
  args[3] = &matrixSize;

  // Create CUDA events for timing
  CUevent start, stop;
  CHECK_CUDA(cuEventCreate(&start, CU_EVENT_DEFAULT), "cuEventCreate start");
  CHECK_CUDA(cuEventCreate(&stop, CU_EVENT_DEFAULT), "cuEventCreate stop");

  // Warm-up run
  CHECK_CUDA(cuLaunchKernel(kernel, gridDim, gridDim, 1,
                            blockSize, blockSize, 1,
                            0, nullptr, args, nullptr),
             "Warm-up kernel launch failed");
  CHECK_CUDA(cuCtxSynchronize(), "Warm-up sync failed");

  // Timed runs
  const int numRuns = 10;
  float totalMs = 0.0f;
  
  for (int i = 0; i < numRuns; i++) {
    CHECK_CUDA(cuEventRecord(start, nullptr), "cuEventRecord start");
    
    CHECK_CUDA(cuLaunchKernel(kernel, gridDim, gridDim, 1,
                              blockSize, blockSize, 1,
                              0, nullptr, args, nullptr),
               "Kernel launch failed");
    
    CHECK_CUDA(cuEventRecord(stop, nullptr), "cuEventRecord stop");
    CHECK_CUDA(cuEventSynchronize(stop), "cuEventSynchronize stop");
    
    float ms;
    CHECK_CUDA(cuEventElapsedTime(&ms, start, stop), "cuEventElapsedTime");
    totalMs += ms;
  }

  float avgMs = totalMs / numRuns;
  printf("Kernel execution time (avg of %d runs): %.3f ms\n", numRuns, avgMs);

  // Copy output back for verification
  CHECK_CUDA(cuMemcpyDtoH(h_C.data(), d_C, matBytes), "cuMemcpyDtoH failed");

  // Verify: for 1.0 * 1.0 matmul, each element = N
  float expected = (float)matrixSize;
  bool pass = true;
  int errors = 0;
  for (int i = 0; i < matrixSize * matrixSize && errors < 5; i++) {
    if (fabs(h_C[i] - expected) > 0.01f * expected) {
      fprintf(stderr, "Mismatch at [%d]: got %f, expected %f\n", i, h_C[i], expected);
      pass = false;
      errors++;
    }
  }

  printf("Result: %s (matches expected)\n", pass ? "PASS" : "FAIL");

  // Cleanup
  cuEventDestroy(start);
  cuEventDestroy(stop);
  cuMemFree(d_A);
  cuMemFree(d_B);
  cuMemFree(d_C);
  cuModuleUnload(module);
  cuCtxDestroy(context);

  return pass ? 0 : 1;
}
