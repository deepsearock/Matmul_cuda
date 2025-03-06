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
template <int TILE_SIZE>
__global__ void matrixMulTiledOptimized(const float *__restrict__ A,
                                        const float *__restrict__ B,
                                        float *__restrict__ C,
                                        int M, int N, int K) {
    // Block indices determine the tile of C computed by this block.
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // The starting row and column for the tile in C.
    int rowBlock = by * TILE_SIZE;
    int colBlock = bx * TILE_SIZE;

    // Compute how many rows in the tile each thread is responsible for.
    // Each thread starting at threadIdx.y computes rows at stride blockDim.y.
    int numRows = (TILE_SIZE + blockDim.y - 1) / blockDim.y;

    // Local accumulation registers.
    // We allocate TILE_SIZE elements (the maximum possible) but only use the first numRows.
    float accum[TILE_SIZE];
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        accum[i] = 0.0f;
    }

    // Shared memory tiles.
    // For A, a standard tile.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    // For B, pad the second dimension by 1 to avoid bank conflicts.
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Number of tiles required in the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load the A tile: each thread loads multiple rows (with stride blockDim.y).
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            int aRow = rowBlock + i;
            int aCol = t * TILE_SIZE + tx;
            As[i][tx] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        }

        // Load the B tile: each thread loads multiple rows.
        for (int i = ty; i < TILE_SIZE; i += blockDim.y) {
            int bRow = t * TILE_SIZE + i;
            int bCol = colBlock + tx;
            Bs[i][tx] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        }

        __syncthreads();

        // Compute the partial products.
        // Unroll the loop over the shared dimension.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float bVal = Bs[k][tx];
            // Each thread computes its micro-tile: rows given by (ty + i * blockDim.y)
            for (int i = 0; i < numRows; i++) {
                int rowInTile = ty + i * blockDim.y;
                if (rowInTile < TILE_SIZE) {
                    accum[i] += As[rowInTile][k] * bVal;
                }
            }
        }

        __syncthreads();
    }

    // Write the accumulated results to global memory.
    for (int i = 0; i < numRows; i++) {
        int row = rowBlock + ty + i * blockDim.y;
        int col = colBlock + tx;
        if (row < M && col < N) {
            C[row * N + col] = accum[i];
        }
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
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(tileSize, tileSize);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<gridDim, dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
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
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(tileSize, tileSize);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<gridDim, dim3(32, 32)>>>(d_A, d_B, d_C, M, N, K);
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
