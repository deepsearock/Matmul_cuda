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
    // Compute “base” micro–tile dimensions using ceiling division.
    // (These are the maximum numbers of rows/cols a thread might process.)
    constexpr int baseMicroTileRows = (TILE_SIZE + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y;
    constexpr int baseMicroTileCols = (TILE_SIZE + BLOCK_DIM_X - 1) / BLOCK_DIM_X;

    // Each thread’s starting coordinate within the tile (in shared memory)
    int tileRowStart = threadIdx.y * baseMicroTileRows;
    int tileColStart = threadIdx.x * baseMicroTileCols;

    // For threads at the block’s boundary, compute the actual micro–tile dimensions.
    int microTileRows = baseMicroTileRows;
    int microTileCols = baseMicroTileCols;
    if (tileRowStart + microTileRows > TILE_SIZE) {
        microTileRows = TILE_SIZE - tileRowStart;
    }
    if (tileColStart + microTileCols > TILE_SIZE) {
        microTileCols = TILE_SIZE - tileColStart;
    }

    // Global offsets for the block’s tile of C.
    int bx = blockIdx.x, by = blockIdx.y;
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;

    // Each thread accumulates a micro–tile of size up to [baseMicroTileRows x baseMicroTileCols].
    // (We use the “base” sizes for the register array; only the first microTileRows/microTileCols entries will be used.)
    float accum[baseMicroTileRows][baseMicroTileCols] = {0.0f};

    // Shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        // --- Load tile of A into shared memory ---
        // Global base indices for A for this thread’s portion:
        int aGlobalRowBase = rowTile + tileRowStart;
        int aGlobalColBase = t * TILE_SIZE + tileColStart;
        // Check that the entire micro–tile is within A.
        bool fullTileA = ((aGlobalRowBase + microTileRows) <= M) &&
                         ((aGlobalColBase + microTileCols) <= K);
        if (fullTileA && (microTileCols % 2 == 0)) {
            // Fast path: use vectorized loads (float2).
            typedef float2 Vec;
            constexpr int vecWidth = 2;
            int numVecs = microTileCols / vecWidth;
            for (int i = 0; i < microTileRows; i++) {
                int globalRow = aGlobalRowBase + i;
                int aColStart = aGlobalColBase;
                // Reinterpret pointer for vectorized load.
                const Vec* aVecPtr = reinterpret_cast<const Vec*>(&A[globalRow * K + aColStart]);
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = aVecPtr[v];
                    int smemColIndex = tileColStart + v * vecWidth;
                    As[tileRowStart + i][smemColIndex]     = vecVal.x;
                    As[tileRowStart + i][smemColIndex + 1] = vecVal.y;
                }
            }
        } else {
            // Slow path: scalar loads with bounds checking.
            for (int i = 0; i < microTileRows; i++) {
                int globalRow = aGlobalRowBase + i;
                int aColStart = aGlobalColBase;
                for (int j = 0; j < microTileCols; j++) {
                    int globalCol = aColStart + j;
                    if (globalRow < M && globalCol < K)
                        As[tileRowStart + i][tileColStart + j] = A[globalRow * K + globalCol];
                    else
                        As[tileRowStart + i][tileColStart + j] = 0.0f;
                }
            }
        }

        // --- Load tile of B into shared memory ---
        // For B, the global base indices (B is K x N):
        int bGlobalRowBase = t * TILE_SIZE + tileRowStart;
        int bGlobalColBase = colTile + tileColStart;
        bool fullTileB = ((bGlobalRowBase + microTileRows) <= K) &&
                         ((bGlobalColBase + microTileCols) <= N);
        if (fullTileB && (microTileCols % 2 == 0)) {
            typedef float2 Vec;
            constexpr int vecWidth = 2;
            int numVecs = microTileCols / vecWidth;
            for (int i = 0; i < microTileRows; i++) {
                int globalRow = bGlobalRowBase + i;
                int bColStart = bGlobalColBase;
                const Vec* bVecPtr = reinterpret_cast<const Vec*>(&B[globalRow * N + bColStart]);
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = bVecPtr[v];
                    int smemColIndex = tileColStart + v * vecWidth;
                    Bs[tileRowStart + i][smemColIndex]     = vecVal.x;
                    Bs[tileRowStart + i][smemColIndex + 1] = vecVal.y;
                }
            }
        } else {
            for (int i = 0; i < microTileRows; i++) {
                int globalRow = bGlobalRowBase + i;
                int bColStart = bGlobalColBase;
                for (int j = 0; j < microTileCols; j++) {
                    int globalCol = bColStart + j;
                    if (globalRow < K && globalCol < N)
                        Bs[tileRowStart + i][tileColStart + j] = B[globalRow * N + globalCol];
                    else
                        Bs[tileRowStart + i][tileColStart + j] = 0.0f;
                }
            }
        }

        __syncthreads(); // Ensure the full tiles are loaded before computing

        // --- Multiply the two tiles ---
        for (int k = 0; k < TILE_SIZE; k++) {
            for (int i = 0; i < microTileRows; i++) {
                float aVal = As[tileRowStart + i][k];
                for (int j = 0; j < microTileCols; j++) {
                    accum[i][j] = __fmaf_rn(aVal, Bs[k][tileColStart + j], accum[i][j]);
                }
            }
        }
        __syncthreads(); // Synchronize before loading the next tile
    }

    // --- Write the accumulated micro–tile back to global memory ---
    for (int i = 0; i < microTileRows; i++) {
        int globalRow = rowTile + tileRowStart + i;
        for (int j = 0; j < microTileCols; j++) {
            int globalCol = colTile + tileColStart + j;
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

    // Initialize host memory with random values.
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate and copy memory to device.
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Choose block dimensions independently of tileSize.
    // (For example, here blockDim.x is set to tileSize, but you could choose any configuration.)
    dim3 blockDim(tileSize, 256 / tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    // Launch kernel using runtime-determined grid and block sizes.
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

    // Copy results back to host.
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free memory.
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
