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
    // Each block computes a TILE_SIZE x TILE_SIZE tile of C.
    // Each thread computes a micro-tile of size:
    constexpr int MICRO_TILE_ROWS = TILE_SIZE / BLOCK_DIM_Y; // vertical sub-tile per thread
    constexpr int MICRO_TILE_COLS = TILE_SIZE / BLOCK_DIM_X; // horizontal sub-tile per thread

    // Block and thread indices.
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Top-left of the block-tile in global C.
    int rowTile = by * TILE_SIZE;
    int colTile = bx * TILE_SIZE;

    // Each thread’s computed sub-tile will be accumulated in registers.
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Shared memory tiles for A and B.
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along the K-dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {

        // --- Load tile of A into shared memory ---
        // Each thread loads a MICRO_TILE_ROWS x MICRO_TILE_COLS sub-tile from A.
        // Compute whether the A tile is fully inside (so we can use a vectorized load).
        bool fullTileA = ((rowTile + ty * MICRO_TILE_ROWS + MICRO_TILE_ROWS) <= M) &&
                         ((t * TILE_SIZE + tx * MICRO_TILE_COLS + MICRO_TILE_COLS) <= K);
        if (fullTileA && (MICRO_TILE_COLS % 2 == 0)) {
            // Fast (vectorized) load path.
            typedef float2 Vec;
            constexpr int vecWidth = 2; // number of floats per vector load
            int numVecs = MICRO_TILE_COLS / vecWidth;
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                // Global row index for A.
                int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
                // Starting column within the tile for this thread.
                int aColStart = t * TILE_SIZE + tx * MICRO_TILE_COLS;
                // Reinterpret the A pointer as a vector pointer.
                const Vec* aVecPtr = reinterpret_cast<const Vec*>(&A[globalRow * K + aColStart]);
                #pragma unroll
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = aVecPtr[v];
                    // Write into shared memory at row (ty*MICRO_TILE_ROWS + i) and the appropriate columns.
                    int colIndex = tx * MICRO_TILE_COLS + v * vecWidth;
                    As[ty * MICRO_TILE_ROWS + i][colIndex]     = vecVal.x;
                    As[ty * MICRO_TILE_ROWS + i][colIndex + 1] = vecVal.y;
                }
            }
        } else {
            // Slow (scalar) load path with bounds checking.
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
                int aColStart = t * TILE_SIZE + tx * MICRO_TILE_COLS;
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = aColStart + j;
                    if (globalRow < M && globalCol < K)
                        As[ty * MICRO_TILE_ROWS + i][tx * MICRO_TILE_COLS + j] = A[globalRow * K + globalCol];
                    else
                        As[ty * MICRO_TILE_ROWS + i][tx * MICRO_TILE_COLS + j] = 0.0f;
                }
            }
        }

        // --- Load tile of B into shared memory ---
        // Each thread loads a MICRO_TILE_ROWS x MICRO_TILE_COLS sub-tile from B.
        bool fullTileB = ((t * TILE_SIZE + ty * MICRO_TILE_ROWS + MICRO_TILE_ROWS) <= K) &&
                         ((colTile + tx * MICRO_TILE_COLS + MICRO_TILE_COLS) <= N);
        if (fullTileB && (MICRO_TILE_COLS % 2 == 0)) {
            typedef float2 Vec;
            constexpr int vecWidth = 2;
            int numVecs = MICRO_TILE_COLS / vecWidth;
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + ty * MICRO_TILE_ROWS + i;
                int bColStart = colTile + tx * MICRO_TILE_COLS;
                const Vec* bVecPtr = reinterpret_cast<const Vec*>(&B[globalRow * N + bColStart]);
                #pragma unroll
                for (int v = 0; v < numVecs; v++) {
                    Vec vecVal = bVecPtr[v];
                    int colIndex = tx * MICRO_TILE_COLS + v * vecWidth;
                    Bs[ty * MICRO_TILE_ROWS + i][colIndex]     = vecVal.x;
                    Bs[ty * MICRO_TILE_ROWS + i][colIndex + 1] = vecVal.y;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = t * TILE_SIZE + ty * MICRO_TILE_ROWS + i;
                int bColStart = colTile + tx * MICRO_TILE_COLS;
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = bColStart + j;
                    if (globalRow < K && globalCol < N)
                        Bs[ty * MICRO_TILE_ROWS + i][tx * MICRO_TILE_COLS + j] = B[globalRow * N + globalCol];
                    else
                        Bs[ty * MICRO_TILE_ROWS + i][tx * MICRO_TILE_COLS + j] = 0.0f;
                }
            }
        }

        __syncthreads(); // Ensure the full tile is loaded before computation

        // --- Multiply the two tiles ---
        // Each thread multiplies the row(s) of As with the column(s) of Bs and accumulates into its micro tile.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                float aVal = As[ty * MICRO_TILE_ROWS + i][k];
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    accum[i][j] = __fmaf_rn(aVal, Bs[k][tx * MICRO_TILE_COLS + j], accum[i][j]);
                }
            }
        }

        __syncthreads(); // Prepare for the next tile
    }

    // --- Write the accumulated sub-tile back to global memory ---
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++) {
        int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
        #pragma unroll
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

    dim3 blockDim(tileSize, 1024 / tileSize);
    dim3 gridDim((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);

    //launch kernel using runtime determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 16:
                matrixMulTiled<64, 16, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 32, 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 64:
                matrixMulTiled<64, 16, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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

    dim3 blockDim(tileSize, 512 / tileSize);
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
                matrixMulTiled<64, 8, 64><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
