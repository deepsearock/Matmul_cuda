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

#include <cuda_pipeline.h>  // For __cp_async intrinsics.
#include <cstdint>
#include <iostream>
#include <utility>

// As before, we assume matrices are stored in row–major order.
// This version uses double–buffering for asynchronous global memory loads.

// Note: For simplicity, this version uses one contiguous asynchronous copy per row.
// You may wish to combine vectorized loads (if alignment/size permits) with __cp_async.

// Templated kernel using double–buffering with asynchronous copy.
template <int BLOCK_DIM_X, int BLOCK_DIM_Y,
          int TILE_SIZE_Y, int TILE_SIZE_X, int TILE_SIZE_K>
__global__ void matrixMulTiledRect_async(const float * __restrict__ A,
                                          const float * __restrict__ B,
                                          float * __restrict__ C,
                                          int M, int N, int K)
{
    // Define micro–tile dimensions (each thread computes a MICRO_TILE)
    constexpr int MICRO_TILE_ROWS = TILE_SIZE_Y / BLOCK_DIM_Y;
    constexpr int MICRO_TILE_COLS = TILE_SIZE_X / BLOCK_DIM_X;

    // Block and thread indices.
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Global tile offset for C.
    int rowTile = by * TILE_SIZE_Y;
    int colTile = bx * TILE_SIZE_X;

    // Allocate double–buffered shared memory.
    // Two buffers: index 0 and 1.
    // For matrix A: each buffer is [TILE_SIZE_Y][TILE_SIZE_K] elements.
    // For matrix B: each buffer is [TILE_SIZE_K][TILE_SIZE_X] elements.
    __shared__ float As[2][TILE_SIZE_Y][TILE_SIZE_K];
    __shared__ float Bs[2][TILE_SIZE_K][TILE_SIZE_X];

    // Each thread maintains a small register block for its output micro–tile.
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Number of tiles along the K dimension.
    int numTiles = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;

    // Preload the first tile (tile 0) into shared memory buffer 0.
    int currBuf = 0;
    // Load A tile (each thread loads multiple rows).
    for (int i = ty; i < TILE_SIZE_Y; i += BLOCK_DIM_Y) {
        int globalRow = rowTile + i;
        int globalCol = 0; // tile 0: offset 0 along K
        if (globalRow < M && (globalCol + TILE_SIZE_K) <= K) {
            // Copy TILE_SIZE_K floats for row i.
            // __cp_async requires the size (in bytes) to be a multiple of 16.
            __cp_async(&As[currBuf][i][0], &A[globalRow * K + globalCol],
                       TILE_SIZE_K * sizeof(float));
        } else {
            // Fallback to scalar loads if tile goes out-of-bound.
            for (int j = 0; j < TILE_SIZE_K; j++) {
                int globalCol_j = globalCol + j;
                As[currBuf][i][j] = (globalRow < M && globalCol_j < K) ?
                                    A[globalRow * K + globalCol_j] : 0.0f;
            }
        }
    }
    // Load B tile.
    for (int i = tx; i < TILE_SIZE_K; i += BLOCK_DIM_X) {
        int globalRow = 0; // tile 0 for B: row offset 0 along K
        int globalCol = colTile;
        if ((globalRow + TILE_SIZE_K) <= K && (globalCol + TILE_SIZE_X) <= N) {
            __cp_async(&Bs[currBuf][i][0], &B[(globalRow + i) * N + globalCol],
                       TILE_SIZE_X * sizeof(float));
        } else {
            for (int j = 0; j < TILE_SIZE_X; j++) {
                int globalCol_j = globalCol + j;
                Bs[currBuf][i][j] = ((globalRow + i) < K && globalCol_j < N) ?
                                    B[(globalRow + i) * N + globalCol_j] : 0.0f;
            }
        }
    }
    // Wait for the initial asynchronous copies to complete.
    __cp_async_wait();
    __syncthreads();

    // Main loop over tiles.
    for (int t = 0; t < numTiles; t++) {
        currBuf = t & 1;         // current tile buffer index.
        int nextTile = t + 1;      // next tile index.
        int nextBuf = nextTile & 1;  // alternate buffer index.
        
        // If there is a next tile, start prefetching it asynchronously.
        if (nextTile < numTiles) {
            // For matrix A: load tile nextTile into buffer "nextBuf".
            for (int i = ty; i < TILE_SIZE_Y; i += BLOCK_DIM_Y) {
                int globalRow = rowTile + i;
                int globalCol = nextTile * TILE_SIZE_K;
                if (globalRow < M && (globalCol + TILE_SIZE_K) <= K) {
                    __cp_async(&As[nextBuf][i][0],
                               &A[globalRow * K + globalCol],
                               TILE_SIZE_K * sizeof(float));
                } else {
                    for (int j = 0; j < TILE_SIZE_K; j++) {
                        int globalCol_j = globalCol + j;
                        As[nextBuf][i][j] = (globalRow < M && globalCol_j < K) ?
                                             A[globalRow * K + globalCol_j] : 0.0f;
                    }
                }
            }
            // For matrix B: load tile nextTile into buffer "nextBuf".
            for (int i = tx; i < TILE_SIZE_K; i += BLOCK_DIM_X) {
                int globalRow = nextTile * TILE_SIZE_K + i;
                int globalCol = colTile;
                if ((globalRow < K) && ((globalCol + TILE_SIZE_X) <= N)) {
                    __cp_async(&Bs[nextBuf][i][0],
                               &B[globalRow * N + globalCol],
                               TILE_SIZE_X * sizeof(float));
                } else {
                    for (int j = 0; j < TILE_SIZE_X; j++) {
                        int globalCol_j = globalCol + j;
                        Bs[nextBuf][i][j] = (globalRow < K && globalCol_j < N) ?
                                             B[globalRow * N + globalCol_j] : 0.0f;
                    }
                }
            }
        }
        
        // Ensure the current tile’s asynchronous copy (if any) is complete.
        __cp_async_wait();
        __syncthreads();

        // Multiply the current tile.
        // Here we use a simple three–nested loop. The inner loops iterate over the micro–tile.
        for (int k = 0; k < TILE_SIZE_K; k++) {
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                float aVal = As[currBuf][ty * MICRO_TILE_ROWS + i][k];
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    accum[i][j] = __fmaf_rn(aVal,
                                            Bs[currBuf][k][tx * MICRO_TILE_COLS + j],
                                            accum[i][j]);
                }
            }
        }
        __syncthreads();  // Ensure all threads finish this tile before moving on.
    }

    // Write the computed micro–tile back to global memory.
    int rowBase = rowTile + ty * MICRO_TILE_ROWS;
    int colBase = colTile + tx * MICRO_TILE_COLS;
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int globalRow = rowBase + i;
        if (globalRow < M) {
            for (int j = 0; j < MICRO_TILE_COLS; j++) {
                int globalCol = colBase + j;
                if (globalCol < N)
                    C[globalRow * N + globalCol] = accum[i][j];
            }
        }
    }
}



/// Wrapper function: selects tile/block parameters based on input sizes.
/// For large matrices such as 3000×4000 or 100×10000, we use the “large” configuration.
// Wrapper function: allocates memory, selects parameters, launches the asynchronous kernel, and measures performance.
inline std::pair<double, double> runMatrixMulTiledAsync(int M, int N, int K) {
    // Allocate host memory.
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    // Initialize host matrices with random values.
    for (int i = 0; i < M * K; ++i)
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i)
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate and copy device memory.
    float *d_A, *d_B, *d_C;
    // (Assume allocateDeviceMemory allocates d_A, d_B, d_C appropriately)
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Dynamically choose tile sizes and block dimensions.
    int tileSizeY, tileSizeX, tileSizeK;
    int blockDimX, blockDimY;
    if (M >= 64 && N >= 128 && K >= 16) {
        // Large configuration.
        tileSizeY = 64; tileSizeX = 128; tileSizeK = 16;
        blockDimX = 16;  // MICRO_TILE_COLS = 128/16 = 8 (allows vectorized stores)
        blockDimY = 256 / blockDimX; // 256/16 = 16, so MICRO_TILE_ROWS = 64/16 = 4.
    } else if (M >= 32 && N >= 64 && K >= 16) {
        // Medium configuration.
        tileSizeY = 32; tileSizeX = 64; tileSizeK = 16;
        blockDimX = 8;  // MICRO_TILE_COLS = 64/8 = 8.
        blockDimY = 256 / blockDimX; // 256/8 = 32.
    } else {
        // Fallback configuration for smaller matrices.
        tileSizeY = 16; tileSizeX = 32; tileSizeK = 16;
        blockDimX = 16;
        blockDimY = 256 / blockDimX;
    }

    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + tileSizeX - 1) / tileSizeX, (M + tileSizeY - 1) / tileSizeY);

    // Launch the asynchronous kernel and measure performance.
    auto result = measurePerformance([&]() {
        if (tileSizeY == 64 && tileSizeX == 128 && tileSizeK == 16) {
            matrixMulTiledRect_async<16, 16, 64, 128, 16>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else if (tileSizeY == 32 && tileSizeX == 64 && tileSizeK == 16) {
            matrixMulTiledRect_async<8, 32, 32, 64, 16>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else if (tileSizeY == 16 && tileSizeX == 32 && tileSizeK == 16) {
            matrixMulTiledRect_async<16, 16, 16, 32, 16>
                <<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else {
            std::cerr << "Unsupported tile configuration" << std::endl;
            exit(EXIT_FAILURE);
        }
    }, M, N, K);

    // Copy the result back to host.
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device and host memory.
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
    // Allocate and copy device memory.
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // --- Dynamically choose tile sizes and block dimensions ---
    // Here we dispatch three cases. For large matrices (e.g. 3000×4000, 100×10000)
    // we typically have M >= 64 and N >= 128.
    int tileSizeY, tileSizeX, tileSizeK;
    int blockDimX, blockDimY;
    if (M >= 64 && N >= 128 && K >= 16) {
        // Large configuration.
        tileSizeY = 64; tileSizeX = 128; tileSizeK = 16;
        blockDimX = 16; // MICRO_TILE_COLS = 128/16 = 8 (allows vectorized loads)
        blockDimY = 256 / blockDimX; // 256/16 = 16, so MICRO_TILE_ROWS = 64/16 = 4.
    } else if (M >= 32 && N >= 64 && K >= 16) {
        // Medium configuration.
        tileSizeY = 32; tileSizeX = 64; tileSizeK = 16;
        blockDimX = 8;  // MICRO_TILE_COLS = 64/8 = 8.
        blockDimY = 256 / blockDimX; // 256/8 = 32.
    } else {
        // Fallback configuration for smaller matrices.
        tileSizeY = 16; tileSizeX = 32; tileSizeK = 16;
        blockDimX = 16;
        blockDimY = 256 / blockDimX;
    }

    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + tileSizeX - 1) / tileSizeX, (M + tileSizeY - 1) / tileSizeY);

    //launch kernel using runtime determined grid and block sizes
    auto result = measurePerformance([&]() {
        if (tileSizeY == 64 && tileSizeX == 128 && tileSizeK == 16) {
            matrixMulTiledRect<16, 16, 64, 128, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else if (tileSizeY == 32 && tileSizeX == 64 && tileSizeK == 16) {
            matrixMulTiledRect<8, 32, 32, 64, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else if (tileSizeY == 16 && tileSizeX == 32 && tileSizeK == 16) {
            matrixMulTiledRect<16, 16, 16, 32, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
        } else {
            std::cerr << "Unsupported tile configuration" << std::endl;
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
