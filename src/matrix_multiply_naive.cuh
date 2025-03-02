#ifndef MATRIX_MULTIPLY_NAIVE_CUH
#define MATRIX_MULTIPLY_NAIVE_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"
#include <random>

// Naive CUDA kernel for matrix multiplication using only global memory
__global__ void matrixMulGlobalNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Function to allocate memory, launch the naive kernel, and measure performance
inline std::pair<double, double> runMatrixMulNaive(int M, int N, int K, int blockSize) {

    populateMatrix(h_A, M, K);
    populateMatrix(h_B, K, N);

    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(blockSize, blockSize, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, 1);

    auto result = measurePerformance([&]() { matrixMulGlobalNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K); }, M, N, K);
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    freeDeviceMemory(d_A, d_B, d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    
    return result;
}

#endif // MATRIX_MULTIPLY_NAIVE_CUH

