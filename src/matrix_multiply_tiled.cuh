#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"

// Tiled CUDA kernel for matrix multiplication using shared memory
template <int TILE_SIZE>
__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE]; // Shared memory for A
    __shared__ float tileB[TILE_SIZE][TILE_SIZE]; // Shared memory for B

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Row index of the C matrix
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Column index of the C matrix
    float sum = 0.0f;

    // Iterate over all tiles
    for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {

        // Load the corresponding tiles from A and B to shared memory
        int tiledRow = row;
        int tiledColA = tileIdx * TILE_SIZE + threadIdx.x;
        int tiledColB = col;
        int tiledRowB = tileIdx * TILE_SIZE + threadIdx.y;

        // Handle boundary conditions: if the thread is out of bounds, load 0
        tileA[threadIdx.y][threadIdx.x] = (tiledRow < M && tiledColA < K) ? A[tiledRow * K + tiledColA] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRowB < K && tiledColB < N) ? B[tiledRowB * N + tiledColB] : 0.0f;

        __syncthreads();  // Ensure that all threads have loaded their respective tiles

        // Compute the sum for this block of C
        for (int k = 0; k < TILE_SIZE; ++k) {
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

    // Launch kernel based on template with fixed TILE_SIZE
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 32)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<dim3((N + 63) / 64, (M + 63) / 64), dim3(64, 64)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<dim3((N + 127) / 128, (M + 127) / 128), dim3(128, 128)>>>(d_A, d_B, d_C, M, N, K);
                break;
            default:
                std::cerr << "Unsupported tile size!" << std::endl;
                exit(EXIT_FAILURE);
        }
    }, M, N, K);

    freeDeviceMemory(d_A, d_B, d_C);
    return result;
}

#endif // MATRIX_MULTIPLY_TILED_CUH
