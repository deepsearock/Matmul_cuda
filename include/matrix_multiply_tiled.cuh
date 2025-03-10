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
    // ---- Warp tiling parameters ----
    // There are 256 threads per block, i.e. 8 warps.
    // We arrange them as 2 warp-rows x 4 warp-cols.
    constexpr int WARP_ROWS      = 2;
    constexpr int WARP_COLS      = 4;
    constexpr int WARPS_PER_BLOCK = 8;
    // Within each warp, arrange 32 threads as 4 rows x 8 cols.
    constexpr int WARP_SUB_ROWS  = 4;
    constexpr int WARP_SUB_COLS  = 8;
    static_assert(WARP_ROWS * WARP_COLS == WARPS_PER_BLOCK, "Must have 8 warps per block.");
    
    // Each warp computes a sub-tile of C of size:
    //   warpTileHeight = TILE_SIZE / WARP_ROWS, warpTileWidth = TILE_SIZE / WARP_COLS.
    // Then each thread computes a micro-tile of size:
    constexpr int MICRO_TILE_ROWS = TILE_SIZE / (WARP_ROWS * WARP_SUB_ROWS); // = TILE_SIZE/8
    constexpr int MICRO_TILE_COLS = TILE_SIZE / (WARP_COLS * WARP_SUB_COLS); // = TILE_SIZE/32

    // Compute the linear thread index within the block.
    int linearId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    int warpId = linearId / 32;  // There are 32 threads per warp.
    int laneId = linearId % 32;

    // Compute warp-level coordinates (assume 2 rows x 4 cols of warps).
    int warpRow = warpId / WARP_COLS; // 0 or 1.
    int warpCol = warpId % WARP_COLS; // 0 .. 3.

    // Compute lane (thread) coordinates within the warp.
    int laneRow = laneId / WARP_SUB_COLS; // 0 .. 3.
    int laneCol = laneId % WARP_SUB_COLS; // 0 .. 7.

    // Top-left of the block-tile in global C.
    int rowTile = blockIdx.y * TILE_SIZE;
    int colTile = blockIdx.x * TILE_SIZE;

    // Each thread’s micro-tile within its warp-tile.
    // The warp-tile covers:
    //    rows: warpRow * (TILE_SIZE / WARP_ROWS)   and columns: warpCol * (TILE_SIZE / WARP_COLS)
    int threadTileRow = rowTile + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS;
    int threadTileCol = colTile + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS;

    // Each thread accumulates its computed micro-tile in registers.
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Shared memory tiles for A and B.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along the K-dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        // --- Load tile of A into shared memory ---
        // Each thread loads a MICRO_TILE_ROWS x MICRO_TILE_COLS sub-tile from A.
        // Compute whether the A tile is fully inside (for vectorized loads).
        bool fullTileA = ((rowTile + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + MICRO_TILE_ROWS) <= M) &&
                         ((t * TILE_SIZE + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + MICRO_TILE_COLS) <= K);
        if (fullTileA && (MICRO_TILE_COLS % 2 == 0)) {
            // Fast (vectorized) load path.
            typedef float2 Vec;
            constexpr int vecWidth = 2; // 2 floats per vector load.
            int numVecs = MICRO_TILE_COLS / vecWidth;
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                // Global row index for A.
                int globalRow = rowTile + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                // Starting column within the tile for this thread.
                int aColStart = t * TILE_SIZE + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS;
                const Vec* aVecPtr = reinterpret_cast<const Vec*>(&A[globalRow * K + aColStart]);
                #pragma unroll
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = aVecPtr[v];
                    // Write into shared memory.
                    int sharedCol = warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + v * vecWidth;
                    int sharedRow = warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    As[sharedRow][sharedCol]     = vecVal.x;
                    As[sharedRow][sharedCol + 1] = vecVal.y;
                }
            }
        } else {
            // Slow (scalar) load path with bounds checking.
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                int aColStart = t * TILE_SIZE + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS;
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = aColStart + j;
                    int sharedRow = warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    int sharedCol = warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + j;
                    if (globalRow < M && globalCol < K)
                        As[sharedRow][sharedCol] = A[globalRow * K + globalCol];
                    else
                        As[sharedRow][sharedCol] = 0.0f;
                }
            }
        }

        // --- Load tile of B into shared memory ---
        bool fullTileB = ((t * TILE_SIZE + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + MICRO_TILE_ROWS) <= K) &&
                         ((colTile + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + MICRO_TILE_COLS) <= N);
        if (fullTileB && (MICRO_TILE_COLS % 2 == 0)) {
            typedef float2 Vec;
            constexpr int vecWidth = 2;
            int numVecs = MICRO_TILE_COLS / vecWidth;
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                int bColStart = colTile + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS;
                const Vec* bVecPtr = reinterpret_cast<const Vec*>(&B[globalRow * N + bColStart]);
                #pragma unroll
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = bVecPtr[v];
                    int sharedCol = warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + v * vecWidth;
                    int sharedRow = warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    Bs[sharedRow][sharedCol]     = vecVal.x;
                    Bs[sharedRow][sharedCol + 1] = vecVal.y;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                int bColStart = colTile + warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS;
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = bColStart + j;
                    int sharedRow = warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    int sharedCol = warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + j;
                    if (globalRow < K && globalCol < N)
                        Bs[sharedRow][sharedCol] = B[globalRow * N + globalCol];
                    else
                        Bs[sharedRow][sharedCol] = 0.0f;
                }
            }
        }

        __syncthreads(); // Ensure the full tile is loaded before computation

        // --- Multiply the two tiles ---
        // Each thread multiplies a row from As with a column from Bs.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                // Compute shared memory row index for A.
                int a_shared_row = warpRow * (TILE_SIZE / WARP_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                float aVal = As[a_shared_row][k];
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int b_shared_col = warpCol * (TILE_SIZE / WARP_COLS) + laneCol * MICRO_TILE_COLS + j;
                    accum[i][j] = __fmaf_rn(aVal, Bs[k][b_shared_col], accum[i][j]);
                }
            }
        }

        __syncthreads(); // Prepare for the next tile
    }

    // --- Write the accumulated sub-tile back to global memory ---
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int globalRow = threadTileRow + i;
        #pragma unroll
        for (int j = 0; j < MICRO_TILE_COLS; j++) {
            int globalCol = threadTileCol + j;
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
