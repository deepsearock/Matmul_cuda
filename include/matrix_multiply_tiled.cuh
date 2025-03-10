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

// Rectangular-tiled matrix multiplication kernel.
// Computes C = A * B, where A is MxK, B is KxN, and C is MxN.
// Template parameters:
//   BLOCK_DIM_X, BLOCK_DIM_Y : thread block dimensions.
//   TILE_SIZE_Y, TILE_SIZE_X : dimensions (rows x columns) of the output tile (C).
//   TILE_SIZE_K            : reduction tile size along the K dimension.
// Helper function for vectorized loads.
// 'leadingDim' is the pitch (number of elements per row) of the global matrix.
// 'globalRow' and 'globalColStart' determine the starting element in the global memory row.
// 'sharedColStart' is the offset in the destination row in shared memory.
template <typename Vec, int vecWidth>
__device__ inline void vectorizedLoad(const float * __restrict__ src,
                                        float * __restrict__ dst,
                                        int globalRow,
                                        int globalColStart,
                                        int sharedColStart,
                                        int numVecs,
                                        int leadingDim)
{
    const Vec* srcVec = reinterpret_cast<const Vec*>(&src[globalRow * leadingDim + globalColStart]);
    #pragma unroll
    for (int v = 0; v < numVecs; v++) {
        Vec vecVal = srcVec[v];
        int offset = sharedColStart + v * vecWidth;
        if constexpr (vecWidth == 4) {
            dst[offset    ] = vecVal.x;
            dst[offset + 1] = vecVal.y;
            dst[offset + 2] = vecVal.z;
            dst[offset + 3] = vecVal.w;
        } else if constexpr (vecWidth == 3) {
            dst[offset    ] = vecVal.x;
            dst[offset + 1] = vecVal.y;
            dst[offset + 2] = vecVal.z;
        } else if constexpr (vecWidth == 2) {
            dst[offset    ] = vecVal.x;
            dst[offset + 1] = vecVal.y;
        }
    }
}


// Rectangular-tiled matrix multiplication kernel.
// Computes C = A * B, where A is MxK, B is KxN, and C is MxN.
// Template parameters:
//   BLOCK_DIM_X, BLOCK_DIM_Y : thread block dimensions.
//   TILE_SIZE_Y, TILE_SIZE_X : dimensions (rows x columns) of the output tile (C).
//   TILE_SIZE_K            : reduction tile size along the K dimension.
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE_Y, int TILE_SIZE_X, int TILE_SIZE_K>
__global__ void matrixMulTiledRect(const float * __restrict__ A,
                                   const float * __restrict__ B,
                                   float * __restrict__ C,
                                   int M, int N, int K)
{
    // Output tile for C is TILE_SIZE_Y x TILE_SIZE_X.
    // For A, the tile is TILE_SIZE_Y x TILE_SIZE_K.
    // For B, the tile is TILE_SIZE_K x TILE_SIZE_X.
    // Each thread computes a micro-tile of C:
    constexpr int MICRO_TILE_ROWS = TILE_SIZE_Y / BLOCK_DIM_Y;
    constexpr int MICRO_TILE_COLS = TILE_SIZE_X / BLOCK_DIM_X;
    // For loading A, compute micro-tile width based on TILE_SIZE_K.
    constexpr int MICRO_TILE_COLS_A = TILE_SIZE_K / BLOCK_DIM_X;
    // For loading B, compute micro-tile height based on TILE_SIZE_K.
    constexpr int MICRO_TILE_ROWS_B = TILE_SIZE_K / BLOCK_DIM_Y;

    // Block and thread indices.
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Global tile offsets for C.
    int rowTile = by * TILE_SIZE_Y;
    int colTile = bx * TILE_SIZE_X;

    // Shared memory allocations.
    // For A: a tile of size [TILE_SIZE_Y][TILE_SIZE_K] (+1 padding in the K dimension).
    __shared__ float As[TILE_SIZE_Y][TILE_SIZE_K + 1];
    // For B: a tile of size [TILE_SIZE_K][TILE_SIZE_X] (+1 padding in the X dimension).
    __shared__ float Bs[TILE_SIZE_K][TILE_SIZE_X + 1];

    // Each thread accumulates a micro-tile of C in registers.
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Compute per-thread base offsets for its micro-tile load.
    // For A (which uses TILE_SIZE_K as width).
    int baseRowA = ty * MICRO_TILE_ROWS;
    int baseColA = tx * MICRO_TILE_COLS_A;
    // For B and C (which use TILE_SIZE_X).
    int baseRowB = ty * MICRO_TILE_ROWS_B;
    int baseColB = tx * MICRO_TILE_COLS;

    // Loop over the K dimension tiles.
    int numTiles = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;
    for (int t = 0; t < numTiles; t++) {
        // --- Load a tile of A into shared memory ---
        // Global A indices: rows [rowTile, rowTile+TILE_SIZE_Y)
        // and columns [t*TILE_SIZE_K, t*TILE_SIZE_K+TILE_SIZE_K).
        bool fullTileA = ((rowTile + baseRowA + MICRO_TILE_ROWS) <= M) &&
                         ((t * TILE_SIZE_K + baseColA + MICRO_TILE_COLS_A) <= K);
        if (fullTileA) {
            // Choose vectorized load based on MICRO_TILE_COLS_A.
            if constexpr (MICRO_TILE_COLS_A % 4 == 0) {
                constexpr int vecWidth = 4;
                typedef float4 Vec;
                int numVecs = MICRO_TILE_COLS_A / vecWidth;
                int aGlobalColStart = t * TILE_SIZE_K + baseColA;
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                    int globalRow = rowTile + baseRowA + i;
                    // Load into the shared memory row starting at column baseColA.
                    vectorizedLoad<Vec, vecWidth>(A, &As[baseRowA + i][0],
                                                  globalRow, aGlobalColStart,
                                                  baseColA, numVecs, K);
                }
            } else if constexpr (MICRO_TILE_COLS_A % 3 == 0) {
                constexpr int vecWidth = 3;
                typedef float3 Vec;
                int numVecs = MICRO_TILE_COLS_A / vecWidth;
                int aGlobalColStart = t * TILE_SIZE_K + baseColA;
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                    int globalRow = rowTile + baseRowA + i;
                    vectorizedLoad<Vec, vecWidth>(A, &As[baseRowA + i][0],
                                                  globalRow, aGlobalColStart,
                                                  baseColA, numVecs, K);
                }
            } else if constexpr (MICRO_TILE_COLS_A % 2 == 0) {
                constexpr int vecWidth = 2;
                typedef float2 Vec;
                int numVecs = MICRO_TILE_COLS_A / vecWidth;
                int aGlobalColStart = t * TILE_SIZE_K + baseColA;
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                    int globalRow = rowTile + baseRowA + i;
                    vectorizedLoad<Vec, vecWidth>(A, &As[baseRowA + i][0],
                                                  globalRow, aGlobalColStart,
                                                  baseColA, numVecs, K);
                }
            } else {
                // Fallback to scalar loads for A.
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                    int globalRow = rowTile + baseRowA + i;
                    int aGlobalColStart = t * TILE_SIZE_K + baseColA;
                    #pragma unroll
                    for (int j = 0; j < MICRO_TILE_COLS_A; j++) {
                        int globalCol = aGlobalColStart + j;
                        int sharedCol = baseColA + j;
                        As[baseRowA + i][sharedCol] =
                            (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
                    }
                }
            }
        } else {
            // Out-of-bound loads for A.
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                int globalRow = rowTile + baseRowA + i;
                int aGlobalColStart = t * TILE_SIZE_K + baseColA;
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS_A; j++) {
                    int globalCol = aGlobalColStart + j;
                    int sharedCol = baseColA + j;
                    As[baseRowA + i][sharedCol] =
                        (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
                }
            }
        }

        // --- Load a tile of B into shared memory ---
        // Global B indices: rows [t*TILE_SIZE_K, t*TILE_SIZE_K+TILE_SIZE_K)
        // and columns [colTile, colTile+TILE_SIZE_X).
        bool fullTileB = ((t * TILE_SIZE_K + baseRowB + MICRO_TILE_ROWS_B) <= K) &&
                         ((colTile + baseColB + MICRO_TILE_COLS) <= N);
        if (fullTileB) {
            if constexpr (MICRO_TILE_COLS % 4 == 0) {
                constexpr int vecWidth = 4;
                typedef float4 Vec;
                int numVecs = MICRO_TILE_COLS / vecWidth;
                int bGlobalColStart = colTile + baseColB;
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS_B; i++) {
                    int globalRow = t * TILE_SIZE_K + baseRowB + i;
                    vectorizedLoad<Vec, vecWidth>(B, &Bs[baseRowB + i][0],
                                                  globalRow, bGlobalColStart,
                                                  baseColB, numVecs, N);
                }
            } else if constexpr (MICRO_TILE_COLS % 3 == 0) {
                constexpr int vecWidth = 3;
                typedef float3 Vec;
                int numVecs = MICRO_TILE_COLS / vecWidth;
                int bGlobalColStart = colTile + baseColB;
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS_B; i++) {
                    int globalRow = t * TILE_SIZE_K + baseRowB + i;
                    vectorizedLoad<Vec, vecWidth>(B, &Bs[baseRowB + i][0],
                                                  globalRow, bGlobalColStart,
                                                  baseColB, numVecs, N);
                }
            } else if constexpr (MICRO_TILE_COLS % 2 == 0) {
                constexpr int vecWidth = 2;
                typedef float2 Vec;
                int numVecs = MICRO_TILE_COLS / vecWidth;
                int bGlobalColStart = colTile + baseColB;
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS_B; i++) {
                    int globalRow = t * TILE_SIZE_K + baseRowB + i;
                    vectorizedLoad<Vec, vecWidth>(B, &Bs[baseRowB + i][0],
                                                  globalRow, bGlobalColStart,
                                                  baseColB, numVecs, N);
                }
            } else {
                // Fallback to scalar loads for B.
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS_B; i++) {
                    int globalRow = t * TILE_SIZE_K + baseRowB + i;
                    int bGlobalColStart = colTile + baseColB;
                    #pragma unroll
                    for (int j = 0; j < MICRO_TILE_COLS; j++) {
                        int globalCol = bGlobalColStart + j;
                        int sharedCol = baseColB + j;
                        Bs[baseRowB + i][sharedCol] =
                            (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
                    }
                }
            }
        } else {
            // Out-of-bound loads for B.
            #pragma unroll
            for (int i = 0; i < MICRO_TILE_ROWS_B; i++) {
                int globalRow = t * TILE_SIZE_K + baseRowB + i;
                int bGlobalColStart = colTile + baseColB;
                #pragma unroll
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    int globalCol = bGlobalColStart + j;
                    int sharedCol = baseColB + j;
                    Bs[baseRowB + i][sharedCol] =
                        (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
                }
            }
        }

        __syncthreads(); // Ensure shared tiles are loaded.

        // --- Multiply the two tiles ---
        for (int k = 0; k < TILE_SIZE_K; k++) {
            for (int i = 0; i < MICRO_TILE_ROWS; i++) {
                float aVal = As[baseRowA + i][k];
                for (int j = 0; j < MICRO_TILE_COLS; j++) {
                    accum[i][j] = __fmaf_rn(aVal, Bs[k][baseColB + j], accum[i][j]);
                }
            }
        }
        __syncthreads(); // Prepare for next tile.
    }

    // --- Write the accumulated sub-tile back to C ---
    if constexpr (MICRO_TILE_COLS % 4 == 0) {
        typedef float4 Vec;
        constexpr int vecWidth = 4;
        int numVecs = MICRO_TILE_COLS / vecWidth;
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
            if (globalRow < M) {
                int colStart = colTile + tx * MICRO_TILE_COLS;
                bool fullStore = (colStart + MICRO_TILE_COLS) <= N;
                if (fullStore) {
                    Vec* cVecPtr = reinterpret_cast<Vec*>(&C[globalRow * N + colStart]);
                    for (int v = 0; v < numVecs; v++) {
                        Vec vecVal;
                        int base = v * vecWidth;
                        vecVal.x = accum[i][base];
                        vecVal.y = accum[i][base + 1];
                        vecVal.z = accum[i][base + 2];
                        vecVal.w = accum[i][base + 3];
                        cVecPtr[v] = vecVal;
                    }
                } else {
                    for (int j = 0; j < MICRO_TILE_COLS; j++) {
                        int globalCol = colStart + j;
                        if (globalCol < N)
                            C[globalRow * N + globalCol] = accum[i][j];
                    }
                }
            }
        }
    } else {
        for (int i = 0; i < MICRO_TILE_ROWS; i++) {
            int globalRow = rowTile + ty * MICRO_TILE_ROWS + i;
            for (int j = 0; j < MICRO_TILE_COLS; j++) {
                int globalCol = colTile + tx * MICRO_TILE_COLS + j;
                if (globalRow < M && globalCol < N)
                    C[globalRow * N + globalCol] = accum[i][j];
            }
        }
    }
}


/// Wrapper function: selects tile/block parameters based on input sizes.
/// For large matrices such as 3000×4000 or 100×10000, we use the “large” configuration.
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K) {
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

    // Allocate device memory.
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);

    // Create a CUDA stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy data to the device.
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Dynamically choose tile sizes and block dimensions based on the output matrix shape.
    // Here we use a simple heuristic based on the ratio of M (rows) to N (columns) in the output.
    int tileSizeY, tileSizeX, tileSizeK = 16;  // Fixed tileSizeK in this example.
    int blockDimX, blockDimY;
    if (M >= 2 * N) {
        // Tall output matrix: many more rows than columns.
        tileSizeY = 64;  // Larger tile in Y.
        tileSizeX = 32;  // Smaller tile in X.
        blockDimX = 16;
        blockDimY = 256 / blockDimX; // e.g. 16.
    } else if (N >= 2 * M) {
        // Wide output matrix: many more columns than rows.
        tileSizeY = 32;  // Smaller tile in Y.
        tileSizeX = 128; // Larger tile in X.
        blockDimX = 16;
        blockDimY = 256 / blockDimX; // e.g. 16.
    } else {
        // Nearly square output matrix.
        tileSizeY = 64;
        tileSizeX = 64;
        blockDimX = 16;
        blockDimY = 256 / blockDimX; // e.g. 16.
    }

    // Compute grid dimensions based on chosen tile sizes.
    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + tileSizeX - 1) / tileSizeX, (M + tileSizeY - 1) / tileSizeY);

    // Launch the kernel with the configuration corresponding to the shape.
    auto result = measurePerformance([&]() {
        if (M >= 2 * N) {
            // Tall configuration: instantiate kernel with parameters <BLOCK_DIM_X, BLOCK_DIM_Y, TILE_SIZE_Y, TILE_SIZE_X, TILE_SIZE_K>
            // Here: <16, 16, 64, 32, 16>
            matrixMulTiledRect<16, 16, 64, 32, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        } else if (N >= 2 * M) {
            // Wide configuration: e.g., <16, 16, 32, 128, 16>
            matrixMulTiledRect<16, 16, 32, 128, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        } else {
            // Nearly square configuration: e.g., <16, 16, 64, 64, 16>
            matrixMulTiledRect<16, 16, 64, 64, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        }
        // Synchronize the stream to ensure kernel completion.
        cudaStreamSynchronize(stream);
    }, M, N, K);

    // Asynchronously copy the result back to the host.
    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Ensure all operations are complete.

    // Clean up.
    cudaStreamDestroy(stream);
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
