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
    // Each thread is responsible for one output column.
    int col = colTile + tx;

    // Each thread accumulates MICRO_TILE_ROWS results in registers.
    float accum[MICRO_TILE_ROWS];
#pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        accum[i] = 0.0f;
    }

    // Flatten shared memory arrays for A and B tiles.
    __shared__ float As[TILE_SIZE * TILE_SIZE];
    __shared__ float Bs[TILE_SIZE * TILE_SIZE];

    // Number of tiles along the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // For vectorized loads with float4, determine how many full groups of 4 exist
    int numVecLoadsPerRow = TILE_SIZE / 4;   // full groups of 4 elements
    int remainder = TILE_SIZE % 4;             // extra elements that don’t form a full float4

    // Loop over tiles.
    for (int t = 0; t < numTiles; t++) {
        // --- Load A tile into shared memory.
        for (int r = ty; r < TILE_SIZE; r += BLOCK_DIM_Y) {
            int globalRow = rowTile + r;
            const float* rowPtr = A + globalRow * K + t * TILE_SIZE;
            const float4* vecRowPtr = reinterpret_cast<const float4*>(rowPtr);
            // Vectorized load portion.
            for (int c = tx; c < numVecLoadsPerRow; c += BLOCK_DIM_X) {
                float4 data;
                int globalCol = t * TILE_SIZE + c * 4;
                if (globalRow < M && (globalCol + 3) < K) {
                    data = vecRowPtr[c];
                } else {
                    // For boundary cases, load element-by-element.
                    float tmp[4];
                    for (int j = 0; j < 4; j++) {
                        int colIdx = t * TILE_SIZE + c * 4 + j;
                        tmp[j] = (globalRow < M && colIdx < K) ? A[globalRow * K + colIdx] : 0.0f;
                    }
                    data = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                int shIndex = r * TILE_SIZE + c * 4;
                As[shIndex + 0] = data.x;
                As[shIndex + 1] = data.y;
                As[shIndex + 2] = data.z;
                As[shIndex + 3] = data.w;
            }
            // Remainder: load leftover elements in this row.
            int remStart = numVecLoadsPerRow * 4;
            for (int c = remStart + tx; c < TILE_SIZE; c += BLOCK_DIM_X) {
                int globalCol = t * TILE_SIZE + c;
                float value = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
                As[r * TILE_SIZE + c] = value;
            }
        }

        // --- Load B tile into shared memory.
        for (int r = ty; r < TILE_SIZE; r += BLOCK_DIM_Y) {
            int globalRow = t * TILE_SIZE + r;
            const float* rowPtr = B + globalRow * N + colTile;
            const float4* vecRowPtr = reinterpret_cast<const float4*>(rowPtr);
            // Vectorized load.
            for (int c = tx; c < numVecLoadsPerRow; c += BLOCK_DIM_X) {
                float4 data;
                int globalCol = colTile + c * 4;
                if (globalRow < K && (globalCol + 3) < N) {
                    data = vecRowPtr[c];
                } else {
                    float tmp[4];
                    for (int j = 0; j < 4; j++) {
                        int colIdx = colTile + c * 4 + j;
                        tmp[j] = (globalRow < K && colIdx < N) ? B[globalRow * N + colIdx] : 0.0f;
                    }
                    data = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
                }
                int shIndex = r * TILE_SIZE + c * 4;
                Bs[shIndex + 0] = data.x;
                Bs[shIndex + 1] = data.y;
                Bs[shIndex + 2] = data.z;
                Bs[shIndex + 3] = data.w;
            }
            // Remainder portion for B.
            int remStart = numVecLoadsPerRow * 4;
            for (int c = remStart + tx; c < TILE_SIZE; c += BLOCK_DIM_X) {
                int globalCol = colTile + c;
                float value = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
                Bs[r * TILE_SIZE + c] = value;
            }
        }

        __syncthreads();  // Ensure the entire tile is loaded.

        // --- Compute partial products.
        for (int k = 0; k < TILE_SIZE; k++) {
            float bVal = Bs[k * TILE_SIZE + tx];  // Bs[row k, column tx]
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int rowIndex = ty + i * BLOCK_DIM_Y;
                accum[i] += As[rowIndex * TILE_SIZE + k] * bVal;
            }
        }

        __syncthreads();  // Wait before loading the next tile.
    }

    // --- Write the computed micro‑tile back to global memory.
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
