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

    // assign shared memory for tile a and tile b
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    // calculate the row and column indexes
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // sum register
    float sum = 0;

    // Iterate over all tiles required to compute C(row, col)
    for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        // load the a tile into memory
        int tiledRow = row;
        int tiledColA = tileIdx * TILE_SIZE + threadIdx.x;
        if (tiledRow < M && tiledColA < K)
            tileA[threadIdx.y][threadIdx.x] = A[tiledRow * K + tiledColA];
        else
            tileA[threadIdx.y][threadIdx.x] = 0; // if the row and col are smaller than M and K respectively then it is not 0

        // load the b tile into memory
        int tiledRowB = tileIdx * TILE_SIZE + threadIdx.y;
        int tiledColB = col;
        if (tiledRowB < K && tiledColB < N)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + tiledColB];
        else
            tileB[threadIdx.y][threadIdx.x] = 0; // if the row and col are smaller than N and K respectively then it is not 0

        // synchronize
        __syncthreads();

        // matrix multiplication
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        
        // synchronize before loading new tiles
        __syncthreads();
    }

    // computed value is stored in global memory
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
                matrixMulTiled<8><<<dim3((N + 7) / 8, (M + 7) / 8), dim3(8, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<dim3((N + 15) / 16, (M + 15) / 16), dim3(16, 16)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 32)>>>(d_A, d_B, d_C, M, N, K);
                break;
            default:
                std::cerr << "Unsupported tile size!" << std::endl;
                exit(EXIT_FAILURE);
        }
    }, M, N, K);

    freeDeviceMemory(d_A, d_B, d_C);
    return result;
}

#endif
