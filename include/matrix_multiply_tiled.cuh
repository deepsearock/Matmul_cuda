#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "utils.cuh"

// Optimized Tiled Matrix Multiplication Kernel
// TILE_SIZE is a compile-time parameter.
// blockDim.y is flexible.
// This version uses padding for the B tile to reduce bank conflicts,
// unrolls the inner loop, and uses __restrict__ qualifiers.
#include <cuda_pipeline.h>  // May be required for __cp_async intrinsics.
#include <cstdint>

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE>
__global__ void matrixMulTiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K)
{
    // Ensure TILE_SIZE is divisible by BLOCK_DIM_Y.
    const int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y;  // e.g., 64/16 = 4

    // Block indices.
    int bx = blockIdx.x, by = blockIdx.y;
    // Thread indices.
    int tx = threadIdx.x, ty = threadIdx.y;

    // Compute starting coordinates for the output tile in C.
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;
    // With blockDim.x = TILE_SIZE, each thread covers one column of the tile.
    int col = colTile + tx;

    // Each thread accumulates MICRO_TILE_ROWS results in registers.
    float accum[MICRO_TILE_ROWS];
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        accum[i] = 0.0f;
    }

    // Shared memory for tiles.
    // As: TILE_SIZE x TILE_SIZE for matrix A.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    // Bs: TILE_SIZE x (TILE_SIZE+1) for matrix B (padding to reduce bank conflicts).
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Number of tiles along the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles.
    for (int t = 0; t < numTiles; t++) {
        // Load A tile into shared memory
        #pragma unroll
        for (int i = 0; i < MICRO_TILE_ROWS; i += 2) { // Load two rows per iteration
            int rowA0 = rowTile + ty + i * BLOCK_DIM_Y;
            int rowA1 = rowA0 + BLOCK_DIM_Y; // Next row
            int colA = t * TILE_SIZE + tx;
    
            // Ensure row indices are within bounds before loading
            As[ty + i * BLOCK_DIM_Y][tx] = (rowA0 < M && colA < K) ? A[rowA0 * K + colA] : 0.0f;
            if (rowA1 < M) {
                As[ty + (i+1) * BLOCK_DIM_Y][tx] = (colA < K) ? A[rowA1 * K + colA] : 0.0f;
            }
        }
    
        // Load B tile into shared memory (vectorized load)
        #pragma unroll
        for (int i = ty; i < TILE_SIZE; i += BLOCK_DIM_Y) {
            int rowB = t * TILE_SIZE + i;
            int colB = colTile + tx;
            Bs[i][tx] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
        }
    
        __syncthreads(); // Ensure shared memory is fully loaded
    
        // Compute partial products using loop unrolling and register blocking
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float bVal = Bs[k][tx]; // Load once into register
    
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i += 4) { // Unroll to process 4 rows at a time
                int rowIndex0 = ty + i * BLOCK_DIM_Y;
                int rowIndex1 = rowIndex0 + BLOCK_DIM_Y;
                int rowIndex2 = rowIndex1 + BLOCK_DIM_Y;
                int rowIndex3 = rowIndex2 + BLOCK_DIM_Y;
    
                accum[i]   += As[rowIndex0][k] * bVal;
                if (rowIndex1 < TILE_SIZE) accum[i+1] += As[rowIndex1][k] * bVal;
                if (rowIndex2 < TILE_SIZE) accum[i+2] += As[rowIndex2][k] * bVal;
                if (rowIndex3 < TILE_SIZE) accum[i+3] += As[rowIndex3][k] * bVal;
            }
        }
    
        __syncthreads(); // Synchronize before next tile load
    }
    


        __syncthreads();  // Wait before loading the next tile.
    }

    // Write the computed micro‑tile back to global memory.
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int rowC = rowTile + ty + i * BLOCK_DIM_Y;
        if (rowC < M && col < N)
            C[rowC * N + col] = accum[i];
    }
}


// wrapper function that measures performance and does memory management
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);

    dim3 blockDim(tileSize, 256 / tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 16:
                matrixMulTiled<16, 16, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8, 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64, 4, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            default:
                std::cerr << "Unsupported tile size" << std::endl;
                exit(EXIT_FAILURE);
        }
    }, M, N, K);

    freeDeviceMemory(d_A, d_B, d_C);
    return result;
}

inline std::pair<double, double> runMatrixMulTiledWithErrorCheck(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    // Initialize host memory with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate and copy memory to device
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);


    // Launch kernel using runtime-determined grid and block sizes
    dim3 blockDim(tileSize, 256 / tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 16:
                matrixMulTiled<16, 16, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8, 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64, 4, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            default:
                std::cerr << "Unsupported tile size" << std::endl;
                exit(EXIT_FAILURE);
        }
    }, M, N, K);
    // Copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference result
    matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Compute error metrics
    double mse = 0.0, max_error = 0.0;
    int error_count = 0;
    double error_threshold = 1e-3; // Acceptable error threshold

    for (int i = 0; i < M * N; ++i) {
        double diff = fabs(h_C[i] - h_C_ref[i]);
        mse += diff * diff;
        max_error = std::max(max_error, diff);
        if (diff > error_threshold) error_count++;
    }
    mse /= (M * N);
    double error_percentage = (error_count * 100.0) / (M * N);

    // Print error results
    std::cout << "Error Percentage: " << error_percentage << "%" << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Max Absolute Error: " << max_error << std::endl;

    // Clean up
    freeDeviceMemory(d_A, d_B, d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return result;
}


#endif
