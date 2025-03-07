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
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE, int WARP_DIM_X, int WARP_DIM_Y>
__global__ void matrixMulTiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K)
{
    // Ensure the warp dimensions multiply to 32.
    static_assert(WARP_DIM_X * WARP_DIM_Y == 32, "Warp dimensions must multiply to 32");

    // Compute how many warps per block in each dimension.
    constexpr int WARPS_PER_BLOCK_X = BLOCK_DIM_X / WARP_DIM_X;
    constexpr int WARPS_PER_BLOCK_Y = BLOCK_DIM_Y / WARP_DIM_Y;

    // Define the micro-tile sizes computed from the block dimensions.
    constexpr int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y;
    constexpr int MICRO_TILE_COLS = TILE_SIZE / BLOCK_DIM_X;

    // Block and thread indices.
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Compute warp indices within the block.
    int warp_idx_x = tx / WARP_DIM_X;
    int warp_idx_y = ty / WARP_DIM_Y;
    int warp_id = warp_idx_y * WARPS_PER_BLOCK_X + warp_idx_x;

    // Compute lane indices within each warp.
    int lane_x = tx % WARP_DIM_X;
    int lane_y = ty % WARP_DIM_Y;
    int lane_id = lane_y * WARP_DIM_X + lane_x;  // In range [0, 31]

    // Compute the starting row and column of the output tile in C.
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;

    // Each thread computes a MICRO_TILE_ROWS x MICRO_TILE_COLS submatrix.
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Shared memory tiles for A and B.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Number of tiles along the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // === Load A tile into shared memory ===
        // Each thread loads MICRO_TILE_ROWS elements from A.
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int rowA = rowTile + ty + i * BLOCK_DIM_Y;
            int colA = t * TILE_SIZE + tx;
            if (rowA < M && colA < K)
                As[ty + i * BLOCK_DIM_Y][tx] = A[rowA * K + colA];
            else
                As[ty + i * BLOCK_DIM_Y][tx] = 0.0f;
        }

        // === Load B tile into shared memory ===
        // Here we use each thread to load MICRO_TILE_COLS elements from B.
        for (int j = 0; j < MICRO_TILE_COLS; j++) {
            int rowB = t * TILE_SIZE + ty;
            int colB = colTile + tx + j * BLOCK_DIM_X;
            if (rowB < K && colB < N)
                Bs[ty][tx + j * BLOCK_DIM_X] = B[rowB * N + colB];
            else
                Bs[ty][tx + j * BLOCK_DIM_X] = 0.0f;
        }

        __syncthreads();  // Ensure both tiles are fully loaded.

        // === Compute partial products ===
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                float a_val = As[ty + i * BLOCK_DIM_Y][k];
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    float b_val = Bs[k][tx + j * BLOCK_DIM_X];
                    accum[i][j] += a_val * b_val;
                }
            }
        }

        __syncthreads();  // Synchronize before loading the next tile.
    }

    // === Write the computed submatrix to global memory ===
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        for (int j = 0; j < MICRO_TILE_COLS; j++) {
            int rowC = rowTile + ty + i * BLOCK_DIM_Y;
            int colC = colTile + tx + j * BLOCK_DIM_X;
            if (rowC < M && colC < N)
                C[rowC * N + colC] = accum[i][j];
        }
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
                matrixMulTiled<16, 16, 16, 8, 4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8, 32, 8, 4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64, 4, 64, 8, 4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
                matrixMulTiled<16, 16, 16, 8, 4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8, 32, 8, 4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64, 4, 64, 8 ,4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
