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
__global__ void matrixMulTiled(const float * __restrict__ A,
                                     const float * __restrict__ B,
                                     float * __restrict__ C,
                                     int M, int N, int K)
{
    // Each thread computes a MICRO_TILE_ROWS x MICRO_TILE_COLS sub-tile.
    constexpr int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y;
    constexpr int MICRO_TILE_COLS = TILE_SIZE / BLOCK_DIM_X;

    // Block and thread indices.
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Global tile offset in C.
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;

    // Shared memory arrays with extra column padding to avoid bank conflicts.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Use a double-precision accumulator for improved numerical accuracy.
    double accum[MICRO_TILE_ROWS][MICRO_TILE_COLS];
#pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
#pragma unroll
        for (int j = 0; j < MICRO_TILE_COLS; j++) {
            accum[i][j] = 0.0;
        }
    }

    // Per-thread starting positions for the micro-tile.
    int baseRow = ty * MICRO_TILE_ROWS;
    int baseCol = tx * MICRO_TILE_COLS;

    // Compute the number of tiles along the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        //
        // --- Load tile of A into shared memory ---
        //
        // Determine if the tile for A is fully in bounds so we can use vectorized loads.
        bool fullTileA = ((rowTile + baseRow + MICRO_TILE_ROWS) <= M) &&
                         ((t * TILE_SIZE + baseCol + MICRO_TILE_COLS) <= K);
        if (fullTileA) {
            // --- Hybrid vectorized load ---
            // Use float4 for the bulk load.
            const int vecWidth = 4;
            int numVecs   = MICRO_TILE_COLS / vecWidth;           // number of full vector loads
            int remainder = MICRO_TILE_COLS - numVecs * vecWidth;    // remaining elements
#pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + baseRow + i;
                int aGlobalColStart = t * TILE_SIZE + baseCol;
                // Bulk vectorized loads:
                if (numVecs > 0) {
                    const float4* aVecPtr = reinterpret_cast<const float4*>(&A[globalRow * K + aGlobalColStart]);
#pragma unroll
                    for (int v = 0; v < numVecs; v++) {
                        float4 vecVal = aVecPtr[v];
                        int sharedCol = baseCol + v * vecWidth;
                        As[baseRow + i][sharedCol]     = vecVal.x;
                        As[baseRow + i][sharedCol + 1] = vecVal.y;
                        As[baseRow + i][sharedCol + 2] = vecVal.z;
                        As[baseRow + i][sharedCol + 3] = vecVal.w;
                    }
                }
                // Remainder scalar loads:
                int start = numVecs * vecWidth;
#pragma unroll
                for (int j = 0; j < remainder; j++) {
                    int globalCol = t * TILE_SIZE + baseCol + start + j;
                    int sharedCol = baseCol + start + j;
                    As[baseRow + i][sharedCol] = A[globalRow * K + globalCol];
                }
            }
        } else {
            // --- Partial tile: scalar loads with bounds checking ---
#pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + baseRow + i;
                int aGlobalColStart = t * TILE_SIZE + baseCol;
#pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = aGlobalColStart + j;
                    int sharedCol = baseCol + j;
                    As[baseRow + i][sharedCol] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
                }
            }
        }

        //
        // --- Load tile of B into shared memory ---
        //
        bool fullTileB = ((t * TILE_SIZE + baseRow + MICRO_TILE_ROWS) <= K) &&
                         ((colTile + baseCol + MICRO_TILE_COLS) <= N);
        if (fullTileB) {
            const int vecWidth = 4;
            int numVecs   = MICRO_TILE_COLS / vecWidth;
            int remainder = MICRO_TILE_COLS - numVecs * vecWidth;
#pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + baseRow + i;
                int bGlobalColStart = colTile + baseCol;
                if (numVecs > 0) {
                    const float4* bVecPtr = reinterpret_cast<const float4*>(&B[globalRow * N + bGlobalColStart]);
#pragma unroll
                    for (int v = 0; v < numVecs; v++) {
                        float4 vecVal = bVecPtr[v];
                        int sharedCol = baseCol + v * vecWidth;
                        Bs[baseRow + i][sharedCol]     = vecVal.x;
                        Bs[baseRow + i][sharedCol + 1] = vecVal.y;
                        Bs[baseRow + i][sharedCol + 2] = vecVal.z;
                        Bs[baseRow + i][sharedCol + 3] = vecVal.w;
                    }
                }
                int start = numVecs * vecWidth;
#pragma unroll
                for (int j = 0; j < remainder; j++) {
                    int globalCol = colTile + baseCol + start + j;
                    int sharedCol = baseCol + start + j;
                    Bs[baseRow + i][sharedCol] = B[globalRow * N + globalCol];
                }
            }
        } else {
            // Partial tile: load with bounds checking.
#pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + baseRow + i;
                int bGlobalColStart = colTile + baseCol;
#pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = bGlobalColStart + j;
                    int sharedCol = baseCol + j;
                    Bs[baseRow + i][sharedCol] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
                }
            }
        }

        __syncthreads();  // Ensure both tiles are fully loaded.

        //
        // --- Multiply the two tiles ---
        //
#pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
#pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                // Cast to double for high-accuracy FMA.
                double aVal = static_cast<double>(As[baseRow + i][k]);
#pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    accum[i][j] = fma(aVal, static_cast<double>(Bs[k][baseCol + j]), accum[i][j]);
                }
            }
        }

        __syncthreads();  // Prepare for the next tile.
    }

    //
    // --- Write the accumulated sub-tile back to global memory ---
    //
#pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
#pragma unroll
        for (int j = 0; j < MICRO_TILE_COLS; j++) {
            int globalCol = colTile + tx * MICRO_TILE_COLS + j;
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = static_cast<float>(accum[i][j]);
        }
    }
}






//wrapper function that measures performance and does memory management
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    //initialize host memory with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    //allocate and copy memory to device
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(tileSize, 256 / tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    //launch kernel using runtime determined grid and block sizes
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
    //copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    //free memory
    freeDeviceMemory(d_A, d_B, d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return result;
}

inline std::pair<double, double> runMatrixMulTiledWithErrorCheck(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    //initialize host memory with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    //allocate and copy memory to device
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(tileSize, 256 / tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    //launch kernel using runtime determined grid and block sizes
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
    //copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    //compute cpu reference result
    matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);

    //errors
    double mse = 0.0, max_error = 0.0;
    int error_count = 0;
    double error_threshold = 1e-3; //error threshold

    for (int i = 0; i < M * N; ++i) {
        double diff = fabs(h_C[i] - h_C_ref[i]);
        mse += diff * diff;
        max_error = std::max(max_error, diff);
        if (diff > error_threshold) error_count++;
    }
    mse /= (M * N);
    double error_percentage = (error_count * 100.0) / (M * N);

    //error results
    std::cout << "Error Percentage: " << error_percentage << "%" << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Max Absolute Error: " << max_error << std::endl;

    //free memory
    freeDeviceMemory(d_A, d_B, d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return result;
}


#endif
