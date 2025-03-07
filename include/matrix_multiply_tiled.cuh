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

// Example: 2D Warp-Tiled Matrix Multiply Kernel
// BLOCK_DIM_X, BLOCK_DIM_Y: block dimensions (in threads)
// TILE_SIZE: side length of the block tile (in elements)
// WARP_DIM_X, WARP_DIM_Y: dimensions of each warp’s 2D layout (with WARP_DIM_X * WARP_DIM_Y == 32)
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE,
          int WARP_DIM_X, int WARP_DIM_Y>
__global__ void matrixMulTiled(
    const float * __restrict__ A,
    const float * __restrict__ B,
    float * __restrict__ C,
    int M, int N, int K)
{
    // Enforce that each warp is 32 threads.
    static_assert(WARP_DIM_X * WARP_DIM_Y == 32, "Warp dimensions must multiply to 32");

    // Compute the starting (row, col) for this block’s output tile.
    int bx = blockIdx.x, by = blockIdx.y;
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;

    // Allocate shared memory for the A and B tiles.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Flatten the thread’s 2D index within the block.
    int threadId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    int totalThreads = BLOCK_DIM_X * BLOCK_DIM_Y;

    // Compute the warp ID and lane (thread) ID within the warp.
    int warpId = threadId / 32;
    int laneId = threadId % 32;

    // Determine the number of warps in the block’s x-direction.
    int warpsPerRow = BLOCK_DIM_X / WARP_DIM_X;
    // Identify this warp’s 2D location (which warp tile) within the block tile.
    int warpRowIdx = warpId / warpsPerRow;   // warp’s row index
    int warpColIdx = warpId % warpsPerRow;     // warp’s column index

    // Each warp is assigned a sub-tile of the block tile.
    // Here we assume the block tile is evenly subdivided among the warps.
    constexpr int warpTileRows = TILE_SIZE / (BLOCK_DIM_Y / WARP_DIM_Y);
    constexpr int warpTileCols = TILE_SIZE / (BLOCK_DIM_X / WARP_DIM_X);

    // Within each warp, each thread computes a micro-tile.
    constexpr int microTileRows = warpTileRows / WARP_DIM_Y;
    constexpr int microTileCols = warpTileCols / WARP_DIM_X;

    // Compute the starting global coordinate for this warp’s tile.
    int warpTileRowStart = rowTile + warpRowIdx * warpTileRows;
    int warpTileColStart = colTile + warpColIdx * warpTileCols;

    // Compute the thread’s local position within its warp (using the lane id).
    int warpLocalRow = laneId / WARP_DIM_X;
    int warpLocalCol = laneId % WARP_DIM_X;

    // Each thread will accumulate a micro-tile of size microTileRows x microTileCols in registers.
    float accum[microTileRows][microTileCols];
#pragma unroll
    for (int i = 0; i < microTileRows; i++) {
        for (int j = 0; j < microTileCols; j++) {
            accum[i][j] = 0.0f;
        }
    }

    // Compute how many tiles we need to iterate over in the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // --- Load A and B tiles into shared memory.
        // We use a strided loop so that every thread in the block participates.
        for (int index = threadId; index < TILE_SIZE * TILE_SIZE; index += totalThreads) {
            int row = index / TILE_SIZE;
            int col = index % TILE_SIZE;
            int globalRow = rowTile + row;
            int globalCol = t * TILE_SIZE + col;
            if (globalRow < M && globalCol < K)
                As[row][col] = A[globalRow * K + globalCol];
            else
                As[row][col] = 0.0f;
        }
        for (int index = threadId; index < TILE_SIZE * TILE_SIZE; index += totalThreads) {
            int row = index / TILE_SIZE;
            int col = index % TILE_SIZE;
            int globalRow = t * TILE_SIZE + row;
            int globalCol = colTile + col;
            if (globalRow < K && globalCol < N)
                Bs[row][col] = B[globalRow * N + globalCol];
            else
                Bs[row][col] = 0.0f;
        }
        __syncthreads();

        // --- Each warp computes on its micro-tile.
        // Compute the offsets into the shared memory tiles for this warp’s micro tile.
        int aRowOffset = warpRowIdx * warpTileRows + warpLocalRow * microTileRows;
        int bColOffset = warpColIdx * warpTileCols + warpLocalCol * microTileCols;

        // Loop over the shared tile’s K dimension.
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load a column of elements from the A tile for the micro tile rows.
            float aElements[microTileRows];
#pragma unroll
            for (int i = 0; i < microTileRows; i++) {
                aElements[i] = As[aRowOffset + i][k];
            }
            // Multiply these A values with a row of B tile values.
#pragma unroll
            for (int j = 0; j < microTileCols; j++) {
                float bVal = Bs[k][bColOffset + j];
#pragma unroll
                for (int i = 0; i < microTileRows; i++) {
                    accum[i][j] += aElements[i] * bVal;
                }
            }
        }
        __syncthreads();
    }

    // --- Write the computed micro tile back to global memory.
    for (int i = 0; i < microTileRows; i++) {
        for (int j = 0; j < microTileCols; j++) {
            int globalRow = warpTileRowStart + warpLocalRow * microTileRows + i;
            int globalCol = warpTileColStart + warpLocalCol * microTileCols + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = accum[i][j];
            }
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
                matrixMulTiled<16, 16, 16, 4, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8, 32, 4, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64, 4, 64, 4, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
                matrixMulTiled<64, 4, 64, 8, 4><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
