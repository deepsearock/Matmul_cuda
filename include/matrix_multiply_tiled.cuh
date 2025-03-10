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
    // Total threads per block and warps per block.
    constexpr int THREADS_PER_BLOCK = BLOCK_DIM_X * BLOCK_DIM_Y;
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;

    // For both cases, we assume a warp grid of 2 rows.
    // For TILE_SIZE==16 we want 4 warp columns (2x4 = 8 warps total).
    constexpr int WARP_GRID_ROWS = 2;
    constexpr int WARP_GRID_COLS = 4;

    if constexpr (TILE_SIZE == 16) {
        // For 16x16 tile: set each warp’s internal layout to 8 rows x 4 cols.
        constexpr int WARP_SUB_ROWS = 8;
        constexpr int WARP_SUB_COLS = 4;
        // Micro-tile per thread: each thread computes 1 element.
        constexpr int MICRO_TILE_ROWS = TILE_SIZE / (WARP_GRID_ROWS * WARP_SUB_ROWS); // 16/(2*8)=1
        constexpr int MICRO_TILE_COLS = TILE_SIZE / (WARP_GRID_COLS * WARP_SUB_COLS); // 16/(4*4)=1

        // Compute the linear thread index.
        int linearId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        int warpId = linearId / 32;   // 32 threads per warp.
        int laneId = linearId % 32;

        // Determine warp position in the block (2 warp rows x 4 warp columns).
        int warpRow = warpId / WARP_GRID_COLS;
        int warpCol = warpId % WARP_GRID_COLS;

        // Within a warp, each thread's lane is arranged in an 8 (rows) x 4 (cols) grid.
        int laneRow = laneId / WARP_SUB_COLS; // 0..7
        int laneCol = laneId % WARP_SUB_COLS; // 0..3

        // Top-left of the tile in global C.
        int rowTile = blockIdx.y * TILE_SIZE;
        int colTile = blockIdx.x * TILE_SIZE;

        // Each thread computes one output element.
        int threadTileRow = rowTile + warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS;
        int threadTileCol = colTile + warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS;

        // Accumulator for the one output element.
        float accum = 0.0f;

        // Shared memory tiles for A and B.
        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        // Number of tiles along the K-dimension.
        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < numTiles; t++) {
            // --- Load tile of A into shared memory ---
            // Here each thread loads one element.
            int globalRowA = rowTile + warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow;
            int globalColA = t * TILE_SIZE + warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol;
            int sharedRowA = warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow;
            int sharedColA = warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol;
            if (globalRowA < M && globalColA < K)
                As[sharedRowA][sharedColA] = A[globalRowA * K + globalColA];
            else
                As[sharedRowA][sharedColA] = 0.0f;

            // --- Load tile of B into shared memory ---
            int globalRowB = t * TILE_SIZE + warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow;
            int globalColB = colTile + warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol;
            int sharedRowB = warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow;
            int sharedColB = warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol;
            if (globalRowB < K && globalColB < N)
                Bs[sharedRowB][sharedColB] = B[globalRowB * N + globalColB];
            else
                Bs[sharedRowB][sharedColB] = 0.0f;

            __syncthreads(); // Ensure full tile loaded.

            // --- Multiply the tile ---
            for (int k = 0; k < TILE_SIZE; k++) {
                // Each thread multiplies one element from A row with one element from B column.
                accum = __fmaf_rn(As[warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow][k],
                                  Bs[k][warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol],
                                  accum);
            }
            __syncthreads(); // Prepare for next tile.
        }

        // --- Write the computed element back to global memory ---
        if (threadTileRow < M && threadTileCol < N)
            C[threadTileRow * N + threadTileCol] = accum;
    }
    else {
        // Default path for TILE_SIZE not equal to 16.
        // Here we use the original warp tiling parameters: 2 warp rows x 4 warp columns
        // with each warp internally arranged as 4 rows x 8 cols.
        constexpr int WARP_SUB_ROWS = 16;
        constexpr int WARP_SUB_COLS = 2;
        constexpr int MICRO_TILE_ROWS = TILE_SIZE / (WARP_GRID_ROWS * WARP_SUB_ROWS);
        constexpr int MICRO_TILE_COLS = TILE_SIZE / (WARP_GRID_COLS * WARP_SUB_COLS);

        int linearId = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
        int warpId = linearId / 32;
        int laneId = linearId % 32;

        int warpRow = warpId / WARP_GRID_COLS;
        int warpCol = warpId % WARP_GRID_COLS;

        int laneRow = laneId / WARP_SUB_COLS;
        int laneCol = laneId % WARP_SUB_COLS;

        int rowTile = blockIdx.y * TILE_SIZE;
        int colTile = blockIdx.x * TILE_SIZE;

        int threadTileRow = rowTile + warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS;
        int threadTileCol = colTile + warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS;

        float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

        __shared__ float As[TILE_SIZE][TILE_SIZE];
        __shared__ float Bs[TILE_SIZE][TILE_SIZE];

        int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
        for (int t = 0; t < numTiles; t++) {
            // --- Load A into shared memory ---
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRowA = rowTile + warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                int aColStart = t * TILE_SIZE + warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS;
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalColA = aColStart + j;
                    int sharedRow = warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    int sharedCol = warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS + j;
                    if (globalRowA < M && globalColA < K)
                        As[sharedRow][sharedCol] = A[globalRowA * K + globalColA];
                    else
                        As[sharedRow][sharedCol] = 0.0f;
                }
            }
            // --- Load B into shared memory ---
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRowB = t * TILE_SIZE + warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                int bColStart = colTile + warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS;
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalColB = bColStart + j;
                    int sharedRow = warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    int sharedCol = warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS + j;
                    if (globalRowB < K && globalColB < N)
                        Bs[sharedRow][sharedCol] = B[globalRowB * N + globalColB];
                    else
                        Bs[sharedRow][sharedCol] = 0.0f;
                }
            }
            __syncthreads();

            // --- Multiply the tiles ---
            for (int k = 0; k < TILE_SIZE; k++) {
                for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                    int a_shared_row = warpRow * (TILE_SIZE / WARP_GRID_ROWS) + laneRow * MICRO_TILE_ROWS + i;
                    float aVal = As[a_shared_row][k];
                    for (int j = 0; j < MICRO_TILE_COLS; j++) {
                        int b_shared_col = warpCol * (TILE_SIZE / WARP_GRID_COLS) + laneCol * MICRO_TILE_COLS + j;
                        accum[i][j] = __fmaf_rn(aVal, Bs[k][b_shared_col], accum[i][j]);
                    }
                }
            }
            __syncthreads();
        }

        // --- Write the results back ---
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int globalRow = threadTileRow + i;
            for (int j = 0; j < MICRO_TILE_COLS; j++) {
                int globalCol = threadTileCol + j;
                if (globalRow < M && globalCol < N)
                    C[globalRow * N + globalCol] = accum[i][j];
            }
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
