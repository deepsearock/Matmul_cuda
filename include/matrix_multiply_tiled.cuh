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

    // Shared memory tiles with extra padding to avoid bank conflicts.
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];

    // Each thread accumulates its micro-tile in registers.
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Precompute this thread’s base indices for shared memory.
    int baseRow = ty * MICRO_TILE_ROWS;
    int baseCol = tx * MICRO_TILE_COLS;

    // Loop over the K-dimension tiles.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // --- Load tile of A into shared memory ---
        // Determine if we can use vectorized loads.
        bool fullTileA = ((rowTile + baseRow + MICRO_TILE_ROWS) <= M) &&
                         ((t * TILE_SIZE + baseCol + MICRO_TILE_COLS) <= K);
        if (fullTileA && (MICRO_TILE_COLS % 2 == 0)) {
            typedef float2 Vec;
            constexpr int vecWidth = 2;
            int numVecs = MICRO_TILE_COLS / vecWidth;
            // Precompute the starting global column for A.
            int aGlobalColStart = t * TILE_SIZE + baseCol;
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + baseRow + i;
                const Vec* aVecPtr = reinterpret_cast<const Vec*>(&A[globalRow * K + aGlobalColStart]);
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = aVecPtr[v];
                    int sharedCol = baseCol + v * vecWidth;
                    As[baseRow + i][sharedCol]     = vecVal.x;
                    As[baseRow + i][sharedCol + 1] = vecVal.y;
                }
            }
        } else {
            // Fallback: scalar load with bounds checking.
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + baseRow + i;
                int aGlobalColStart = t * TILE_SIZE + baseCol;
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = aGlobalColStart + j;
                    int sharedCol = baseCol + j;
                    if (globalRow < M && globalCol < K)
                        As[baseRow + i][sharedCol] = A[globalRow * K + globalCol];
                    else
                        As[baseRow + i][sharedCol] = 0.0f;
                }
            }
        }

        // --- Load tile of B into shared memory ---
        bool fullTileB = ((t * TILE_SIZE + baseRow + MICRO_TILE_ROWS) <= K) &&
                         ((colTile + baseCol + MICRO_TILE_COLS) <= N);
        if (fullTileB && (MICRO_TILE_COLS % 2 == 0)) {
            typedef float2 Vec;
            constexpr int vecWidth = 2;
            int numVecs = MICRO_TILE_COLS / vecWidth;
            // Precompute the starting global column for B.
            int bGlobalColStart = colTile + baseCol;
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + baseRow + i;
                const Vec* bVecPtr = reinterpret_cast<const Vec*>(&B[globalRow * N + bGlobalColStart]);
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = bVecPtr[v];
                    int sharedCol = baseCol + v * vecWidth;
                    Bs[baseRow + i][sharedCol]     = vecVal.x;
                    Bs[baseRow + i][sharedCol + 1] = vecVal.y;
                }
            }
        } else {
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + baseRow + i;
                int bGlobalColStart = colTile + baseCol;
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = bGlobalColStart + j;
                    int sharedCol = baseCol + j;
                    if (globalRow < K && globalCol < N)
                        Bs[baseRow + i][sharedCol] = B[globalRow * N + globalCol];
                    else
                        Bs[baseRow + i][sharedCol] = 0.0f;
                }
            }
        }

        __syncthreads();  // Ensure the full tile is loaded.

        // --- Compute the product for the current tile ---
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                float aVal = As[baseRow + i][k];
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    accum[i][j] = __fmaf_rn(aVal, Bs[k][baseCol + j], accum[i][j]);
                }
            }
        }

        __syncthreads();  // Prepare for loading the next tile.
    }

    // --- Write the accumulated sub-tile back to global memory ---
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
        for (int j = 0; j < MICRO_TILE_COLS; j++) {
            int globalCol = colTile + tx * MICRO_TILE_COLS + j;
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = accum[i][j];
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
