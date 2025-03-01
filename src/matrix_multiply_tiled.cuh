#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"

// Tiled CUDA kernel for matrix multiplication using shared memory
__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K, int tileSize) {
    // Allocate shared memory for the tiles of A and B
    __shared__ float tileA[tileSize][tileSize];
    __shared__ float tileB[tileSize][tileSize];

    int row = blockIdx.y * tileSize + threadIdx.y;  // Row index of the C matrix
    int col = blockIdx.x * tileSize + threadIdx.x;  // Column index of the C matrix
    float sum = 0.0f;

    // Iterate over all tiles
    for (int tileIdx = 0; tileIdx < (K + tileSize - 1) / tileSize; ++tileIdx) {
        
        // Load the corresponding tiles from A and B to shared memory
        int tiledRow = row;
        int tiledColA = tileIdx * tileSize + threadIdx.x;
        int tiledColB = col;
        int tiledRowB = tileIdx * tileSize + threadIdx.y;

        // Handle boundary conditions: if the thread is out of bounds, load 0
        tileA[threadIdx.y][threadIdx.x] = (tiledRow < M && tiledColA < K) ? A[tiledRow * K + tiledColA] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRowB < K && tiledColB < N) ? B[tiledRowB * N + tiledColB] : 0.0f;

        __syncthreads();  // Ensure that all threads have loaded their respective tiles

        // Compute the sum for this block of C
        for (int k = 0; k < tileSize; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();  // Synchronize threads before loading the next tile
    }

    // Store the result in C, only if the thread is within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to allocate memory, launch the tiled kernel, and measure performance
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);

    auto result = measurePerformance([&]() {
        dim3 blockDim(tileSize, tileSize);
        dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);
        matrixMulTiled<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K, tileSize);
    }, M, N, K);

    freeDeviceMemory(d_A, d_B, d_C);
    return result;
}

#endif // MATRIX_MULTIPLY_TILED_CUH