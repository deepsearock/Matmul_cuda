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
#define WARP_SIZE 32
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE>
__global__ void matrixMulTiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K)
{
    constexpr int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y;
    constexpr int MICRO_TILE_COLS = TILE_SIZE / BLOCK_DIM_X;

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int warp_id = (ty / (WARP_SIZE / BLOCK_DIM_X));  // Warp row index
    int lane_id = tx;  // Lane within the warp

    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;
    int col = colTile + tx;

    // Use registers instead of shared memory where possible
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {

        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int rowA = rowTile + ty + i * BLOCK_DIM_Y;
            int colA = t * TILE_SIZE + tx;
            As[ty + i * BLOCK_DIM_Y][tx] = (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
        }

        // Load B tile into shared memory using warp-wide coalescing
        float regB[MICRO_TILE_COLS];

        for (int i = 0; i < MICRO_TILE_COLS; i++) {
            int rowB = t * TILE_SIZE + ty + i * BLOCK_DIM_Y;
            int colB = colTile + lane_id;
            regB[i] = (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
            Bs[ty + i * BLOCK_DIM_Y][lane_id] = regB[i];
        }

        __syncthreads();

        // Compute partial products with warp-level optimizations

        for (int k = 0; k < TILE_SIZE; k++) {
            float bVal = Bs[k][lane_id];
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                accum[i][0] += As[ty + i * BLOCK_DIM_Y][k] * bVal;
            }
        }

        __syncthreads();
    }

    // Store results back to global memory efficiently
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int rowC = rowTile + ty + i * BLOCK_DIM_Y;
        if (rowC < M && col < N) {
            C[rowC * N + col] = accum[i][0];
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
