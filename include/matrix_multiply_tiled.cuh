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
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K)
{
    // For example, if TILE_SIZE=64 and BLOCK_DIM_X=64, BLOCK_DIM_Y=16 => MICRO_TILE_ROWS=64/16=4.
    const int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y;

    // Block indices.
    int bx = blockIdx.x, by = blockIdx.y;
    // Thread indices.
    int tx = threadIdx.x, ty = threadIdx.y;

    // Compute starting coordinates for this block’s tile in C.
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;
    // With blockDim.x = TILE_SIZE, each thread handles one column in the tile.
    int col = colTile + tx;

    // Register accumulation array. Each thread computes MICRO_TILE_ROWS rows of output.
    float accum[MICRO_TILE_ROWS];
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        accum[i] = 0.0f;
    }

    // Double-buffered shared memory:
    // For A: no padding needed. For B: +1 padding to reduce bank conflicts.
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    // Number of tiles along K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // We'll alternate between pingpong=0 and pingpong=1 buffers.
    int pingpong = 0;

    // --- Preload the first tile (if any) into buffer "pingpong=0" ---
    if (numTiles > 0) {
        // Load A tile into As[pingpong].
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int rowA = rowTile + ty + i * BLOCK_DIM_Y;
            int colA = 0 * TILE_SIZE + tx;
            if (rowA < M && colA < K)
                As[pingpong][ty + i * BLOCK_DIM_Y][tx] = A[rowA * K + colA];
            else
                As[pingpong][ty + i * BLOCK_DIM_Y][tx] = 0.0f;
        }
        // Load B tile into Bs[pingpong].
        for (int i = ty; i < TILE_SIZE; i += BLOCK_DIM_Y) {
            int rowB = 0 * TILE_SIZE + i;
            int colB = colTile + tx;
            if (rowB < K && colB < N)
                Bs[pingpong][i][tx] = B[rowB * N + colB];
            else
                Bs[pingpong][i][tx] = 0.0f;
        }
    }
    __syncthreads();

    // --- Main loop over tiles ---
    for (int t = 0; t < numTiles; t++) {
        // Compute using the tile in As[pingpong], Bs[pingpong].
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float bVal = Bs[pingpong][k][tx];
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int rowIndex = ty + i * BLOCK_DIM_Y; 
                accum[i] += As[pingpong][rowIndex][k] * bVal;
            }
        }
        __syncthreads();

        // If there's a next tile, load it into the alternate buffer while
        // we've just finished computing on the current buffer.
        int nextTile = t + 1;
        int nextPingpong = 1 - pingpong;
        if (nextTile < numTiles) {
            // Load A tile
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int rowA = rowTile + ty + i * BLOCK_DIM_Y;
                int colA = nextTile * TILE_SIZE + tx;
                if (rowA < M && colA < K)
                    As[nextPingpong][ty + i * BLOCK_DIM_Y][tx] = A[rowA * K + colA];
                else
                    As[nextPingpong][ty + i * BLOCK_DIM_Y][tx] = 0.0f;
            }
            // Load B tile
            for (int i = ty; i < TILE_SIZE; i += BLOCK_DIM_Y) {
                int rowB = nextTile * TILE_SIZE + i;
                int colB = colTile + tx;
                if (rowB < K && colB < N)
                    Bs[nextPingpong][i][tx] = B[rowB * N + colB];
                else
                    Bs[nextPingpong][i][tx] = 0.0f;
            }
        }
        __syncthreads();  

        // Switch buffers for the next iteration (if any).
        pingpong = nextPingpong;
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

    dim3 blockDim(64, 4);
    dim3 gridDim((N + 64 - 1) / 64, (M + 64 - 1) / 64);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<64, 4, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<64, 4, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
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

    dim3 blockDim(64, 4);
    dim3 gridDim((N + 64 - 1) / 64, (M + 64 - 1) / 64);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<64, 4, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<64, 4, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
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
