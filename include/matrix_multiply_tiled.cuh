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
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE>
__global__ void matrixMulTiled(
    const float *__restrict__ A,
    const float *__restrict__ B,
    float *__restrict__ C,
    int M, int N, int K)
{
    // Assume TILE_SIZE % BLOCK_DIM_Y == 0.
    const int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y; // e.g. 32/8 = 4

    // Block indices
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Thread indices
    int tx = threadIdx.x;  // 0 .. BLOCK_DIM_X-1   (should equal 32)
    int ty = threadIdx.y;  // 0 .. BLOCK_DIM_Y-1   (should equal 8)

    // Compute the starting coordinates of the tile in C.
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;
    // Global column index for output.
    int col = colTile + tx;

    // Each thread will compute MICRO_TILE_ROWS output elements (its micro-tile) in registers.
    float accum[MICRO_TILE_ROWS];
#pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        accum[i] = 0.0f;
    }

    // Declare double-buffered shared memory.
    // For A: no padding needed.
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    // For B: pad second dimension by 1 to reduce bank conflicts.
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE + 1];

    // Number of tiles along the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int pingpong = 0;

    // --- Preload the first tile into buffer "pingpong" ---
    if (numTiles > 0) {
        #pragma unroll
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int rowA = rowTile + ty + i * BLOCK_DIM_Y;
            int colA = 0 * TILE_SIZE + tx;
            As[pingpong][ty + i * BLOCK_DIM_Y][tx] =
                (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int rowB = 0 * TILE_SIZE + ty + i * BLOCK_DIM_Y;
            int colB = colTile + tx;
            Bs[pingpong][ty + i * BLOCK_DIM_Y][tx] =
                (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
        }
    }
    __syncthreads();

    // --- Loop over tiles ---
    for (int t = 0; t < numTiles; t++) {
        int nextTile = t + 1;
        int nextPingpong = 1 - pingpong;
        // Preload the next tile into the alternate buffer if available.
        if (nextTile < numTiles) {
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int rowA = rowTile + ty + i * BLOCK_DIM_Y;
                int colA = nextTile * TILE_SIZE + tx;
                As[nextPingpong][ty + i * BLOCK_DIM_Y][tx] =
                    (rowA < M && colA < K) ? A[rowA * K + colA] : 0.0f;
            }
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int rowB = nextTile * TILE_SIZE + ty + i * BLOCK_DIM_Y;
                int colB = colTile + tx;
                Bs[nextPingpong][ty + i * BLOCK_DIM_Y][tx] =
                    (rowB < K && colB < N) ? B[rowB * N + colB] : 0.0f;
            }
        }
        __syncthreads();

        // Compute partial products using the tile in the current pingpong buffer.
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

        // Switch buffers if there is another tile.
        if (nextTile < numTiles)
            pingpong = nextPingpong;
    }

    // --- Write the computed micro-tile back to global memory ---
#pragma unroll
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

    int minGridSize, blockSize;

    // Determine block size dynamically based on tile size
    switch (tileSize) {
        case 8:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 32, 32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<16, 16, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16, 16, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<16, 16, 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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

    int minGridSize, blockSize;

    // Determine block size dynamically based on tile size
    switch (tileSize) {
        case 8:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 16, 32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(32, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<32, 8, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<32, 8, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8, 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
