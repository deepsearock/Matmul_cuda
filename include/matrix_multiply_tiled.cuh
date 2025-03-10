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

template <int TILE_SIZE>
__global__ void matrixMulTiled(const float * __restrict__ A,
                                   const float * __restrict__ B,
                                   float * __restrict__ C,
                                   int M, int N, int K)
{
    // We assume each block has 256 threads (8 warps).
    // Compute a flattened thread ID.
    int globalThreadId = threadIdx.y * blockDim.x + threadIdx.x;
    // Compute warp id and lane id.
    int warpId = globalThreadId / 32;    // 0 to 7
    int laneId = globalThreadId % 32;      // 0 to 31

    // Choose warp grid dimensions based on block shape.
    // For 8 warps, we have a few possible 2D arrangements.
    // Here we choose 4 warps along X and 2 along Y if blockDim.x >= blockDim.y,
    // otherwise we choose 2 warps along X and 4 along Y.
    int numWarpsX, numWarpsY;
    if (blockDim.x >= blockDim.y) {
        numWarpsX = 4;
        numWarpsY = 2;
    } else {
        numWarpsX = 2;
        numWarpsY = 4;
    }
    // Warp coordinates (which sub-tile of the block tile this warp covers).
    int warpRow = warpId / numWarpsX;
    int warpCol = warpId % numWarpsX;

    // Within a warp, choose a 2D arrangement of 32 threads.
    // If block is wider, use 8 columns x 4 rows; otherwise use 4 columns x 8 rows.
    int warpThreadDimX, warpThreadDimY;
    if (blockDim.x >= blockDim.y) {
        warpThreadDimX = 8; // 8 columns
        warpThreadDimY = 4; // 4 rows -> 8*4 = 32 threads
    } else {
        warpThreadDimX = 4;
        warpThreadDimY = 8;
    }
    // Compute this thread’s coordinates within its warp.
    int warpLocalRow = laneId / warpThreadDimX;  // row in [0, warpThreadDimY)
    int warpLocalCol = laneId % warpThreadDimX;    // col in [0, warpThreadDimX)

    // The block tile of C is TILE_SIZE x TILE_SIZE.
    // We partition it among the 8 warps.
    int warpTileRows = TILE_SIZE / numWarpsY; // each warp covers these many rows
    int warpTileCols = TILE_SIZE / numWarpsX; // each warp covers these many columns

    // Within each warp tile, further partition among the warp’s threads.
    int microTileRows = warpTileRows / warpThreadDimY;
    int microTileCols = warpTileCols / warpThreadDimX;
    // (For simplicity, we assume TILE_SIZE is chosen so these divisions are exact.
    // Otherwise, extra bounds checking would be needed.)

    // Compute the global starting indices for the block’s tile.
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;

    // Each warp’s sub-tile of C starts at:
    int warpTileStartRow = rowTile + warpRow * warpTileRows;
    int warpTileStartCol = colTile + warpCol * warpTileCols;

    // Each thread’s micro-tile of C starts at:
    int threadStartRow = warpTileStartRow + warpLocalRow * microTileRows;
    int threadStartCol = warpTileStartCol + warpLocalCol * microTileCols;

    // Allocate a register accumulator for the micro-tile.
    float accum[microTileRows][microTileCols] = {0.0f};

    // Shared memory tiles for A and B.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        // --- Load tile of A into shared memory ---
        // Each thread loads its micro-tile from A.
        // For A, the block tile covers rows [rowTile, rowTile+TILE_SIZE)
        // and columns [t*TILE_SIZE, t*TILE_SIZE+TILE_SIZE).
        for (int i = 0; i < microTileRows; i++) {
            for (int j = 0; j < microTileCols; j++) {
                int aRow = threadStartRow + i;
                int aCol = t * TILE_SIZE + (warpTileStartCol - colTile) + (warpLocalCol * microTileCols) + j;
                float aVal = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
                // Compute where to store in shared memory.
                // We map the thread’s load to the same location as in the block tile.
                int sharedRow = threadStartRow - rowTile + i;
                int sharedCol = (aCol - t * TILE_SIZE);
                As[sharedRow][sharedCol] = aVal;
            }
        }

        // --- Load tile of B into shared memory ---
        // For B, the block tile covers rows [t*TILE_SIZE, t*TILE_SIZE+TILE_SIZE)
        // and columns [colTile, colTile+TILE_SIZE).
        for (int i = 0; i < microTileRows; i++) {
            for (int j = 0; j < microTileCols; j++) {
                int bRow = t * TILE_SIZE + (warpTileStartRow - rowTile) + warpLocalRow * microTileRows + i;
                int bCol = threadStartCol + j;
                float bVal = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
                int sharedRow = (bRow - t * TILE_SIZE);
                int sharedCol = threadStartCol - colTile + j;
                Bs[sharedRow][sharedCol] = bVal;
            }
        }

        __syncthreads();

        // --- Multiply the two tiles ---
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < microTileRows; i++) {
                float aVal = As[(threadStartRow - rowTile) + i][k];
                for (int j = 0; j < microTileCols; j++) {
                    accum[i][j] = __fmaf_rn(aVal, Bs[k][(threadStartCol - colTile) + j], accum[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // --- Write the computed micro-tile back to global memory ---
    for (int i = 0; i < microTileRows; i++) {
        for (int j = 0; j < microTileCols; j++) {
            int cRow = threadStartRow + i;
            int cCol = threadStartCol + j;
            if (cRow < M && cCol < N)
                C[cRow * N + cCol] = accum[i][j];
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
                matrixMulTiled<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
                matrixMulTiled<16s><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
