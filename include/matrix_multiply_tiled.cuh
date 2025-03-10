#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "utils.cuh"
#include <cassert>

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
#include <cuda_runtime.h>
#include <cassert>

// Example: compile-time parameters for tile/block sizes
// BLOCK_DIM_X, BLOCK_DIM_Y: Threads along X, Y in a block
// TILE_SIZE_Y, TILE_SIZE_X: Output tile dims in rows, columns
// TILE_SIZE_K             : Reduction tile size
template <int BLOCK_DIM_X,
          int BLOCK_DIM_Y,
          int TILE_SIZE_Y,
          int TILE_SIZE_X,
          int TILE_SIZE_K>
__global__
void matrixMulTiledRect(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float*       __restrict__ C,
                              int M, int N, int K)
{
    // Compute micro-tile sizes
    constexpr int MICRO_TILE_ROWS  = TILE_SIZE_Y / BLOCK_DIM_Y;  // Must be integer
    constexpr int MICRO_TILE_COLS  = TILE_SIZE_X / BLOCK_DIM_X;  // Must be integer

    // For A loads (tile is TILE_SIZE_Y x TILE_SIZE_K)
    // we chunk along BLOCK_DIM_X => micro-tile in "width"
    constexpr int MICRO_TILE_COLS_A = TILE_SIZE_K / BLOCK_DIM_X; // Must be integer

    // For B loads (tile is TILE_SIZE_K x TILE_SIZE_X)
    // we chunk along BLOCK_DIM_Y => micro-tile in "height"
    constexpr int MICRO_TILE_ROWS_B = TILE_SIZE_K / BLOCK_DIM_Y; // Must be integer

    // Compute block indices
    int bx = blockIdx.x; 
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global tile offsets for C
    int rowTile = by * TILE_SIZE_Y;
    int colTile = bx * TILE_SIZE_X;

    // Shared memory
    __shared__ float As[TILE_SIZE_Y][TILE_SIZE_K + 1];
    __shared__ float Bs[TILE_SIZE_K][TILE_SIZE_X + 1];

    // Each thread accumulates a micro-tile of C in registers
    float accum[MICRO_TILE_ROWS][MICRO_TILE_COLS] = {0.0f};

    // Base offsets within the tile for A, B loads
    int baseRowA = ty * MICRO_TILE_ROWS;    // how many row-chunks each thread covers in tile
    int baseColA = tx * MICRO_TILE_COLS_A;  // how many col-chunks each thread covers in tile

    int baseRowB = ty * MICRO_TILE_ROWS_B;  // how many row-chunks each thread covers in tile (for B)
    int baseColB = tx * MICRO_TILE_COLS;    // how many col-chunks each thread covers in tile (for B, C)

    // Number of horizontal slices in K dimension
    int numTilesK = (K + TILE_SIZE_K - 1) / TILE_SIZE_K;

    // Loop over slices in K dimension
    for (int tileK = 0; tileK < numTilesK; tileK++) 
    {
        // -- Load partial tile of A into shared memory (scalar fallback for safety) --
        // Global offset in A:
        //  - Row: from rowTile..(rowTile+TILE_SIZE_Y-1)
        //  - Col: from tileK*TILE_SIZE_K..(tileK*TILE_SIZE_K + TILE_SIZE_K - 1)
        for (int i = 0; i < MICRO_TILE_ROWS; i++) 
        {
            int globalRow = rowTile + (baseRowA + i);
            for (int j = 0; j < MICRO_TILE_COLS_A; j++)
            {
                int globalCol = tileK * TILE_SIZE_K + (baseColA + j);

                // Shared memory indices
                int sharedRow = (baseRowA + i);
                int sharedCol = (baseColA + j);

                // Check: within allocated shared memory?
                //   sharedRow < TILE_SIZE_Y
                //   sharedCol < TILE_SIZE_K+1
                if (sharedRow < TILE_SIZE_Y && sharedCol < (TILE_SIZE_K + 1))
                {
                    // Check: within actual matrix A?
                    float val = 0.0f;
                    if ((globalRow < M) && (globalCol < K)) {
                        val = A[globalRow * K + globalCol];
                    }
                    As[sharedRow][sharedCol] = val;
                }
            }
        }

        // -- Load partial tile of B into shared memory (scalar fallback for safety) --
        // Global offset in B:
        //  - Row: from tileK*TILE_SIZE_K..(tileK*TILE_SIZE_K + TILE_SIZE_K - 1)
        //  - Col: from colTile..(colTile + TILE_SIZE_X - 1)
        for (int i = 0; i < MICRO_TILE_ROWS_B; i++)
        {
            int globalRow = tileK * TILE_SIZE_K + (baseRowB + i);
            for (int j = 0; j < MICRO_TILE_COLS; j++)
            {
                int globalCol = colTile + (baseColB + j);

                // Shared memory indices
                int sharedRow = (baseRowB + i);
                int sharedCol = (baseColB + j);

                // Check: within allocated shared memory?
                //   sharedRow < TILE_SIZE_K
                //   sharedCol < TILE_SIZE_X+1
                if (sharedRow < TILE_SIZE_K && sharedCol < (TILE_SIZE_X + 1))
                {
                    // Check: within actual matrix B?
                    float val = 0.0f;
                    if ((globalRow < K) && (globalCol < N)) {
                        val = B[globalRow * N + globalCol];
                    }
                    Bs[sharedRow][sharedCol] = val;
                }
            }
        }

        __syncthreads(); // Make sure As[] and Bs[] are loaded

        // -- Multiply partial tile of A & B, accumulate in registers --
        for (int kk = 0; kk < TILE_SIZE_K; kk++)
        {
            // If tileK*TILE_SIZE_K + kk >= K, then we are beyond the matrix dimension,
            // but As/Bs store 0.0f for out-of-bounds => no harm in multiply
            if (kk < TILE_SIZE_K) 
            {
                float* rowAs = As[baseRowA];  // pointer to the top row for this thread
                float* rowBs = Bs[kk];        // we will vary row from As, col from Bs

                // Each thread does micro-rows x micro-cols
                #pragma unroll
                for (int i = 0; i < MICRO_TILE_ROWS; i++)
                {
                    float aVal = As[baseRowA + i][kk];
                    #pragma unroll
                    for (int j = 0; j < MICRO_TILE_COLS; j++)
                    {
                        accum[i][j] = __fmaf_rn(aVal, Bs[kk][baseColB + j], accum[i][j]);
                    }
                }
            }
        }
        __syncthreads(); // Next tile load
    }

    // -- Store accum back to C in partial fashion --
    #pragma unroll
    for (int i = 0; i < MICRO_TILE_ROWS; i++)
    {
        int globalRow = rowTile + (ty * MICRO_TILE_ROWS + i);
        #pragma unroll
        for (int j = 0; j < MICRO_TILE_COLS; j++)
        {
            int globalCol = colTile + (tx * MICRO_TILE_COLS + j);

            if (globalRow < M && globalCol < N) {
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
    // We use a fixed threads-per-block budget (e.g. 256) and pick block dimensions
    // such that BLOCK_DIM_X divides the chosen tile size in X and similarly for Y.
    int tileSizeY, tileSizeX, tileSizeK = 16;
    int blockDimX, blockDimY;
    const int threadsPerBlock = 256;

    if (M >= 2 * N) {
        // Tall output matrix: many more rows than columns.
        // Favor more threads in the Y-dimension to cover extra rows.
        blockDimX = 8;                     // Fewer threads along X.
        blockDimY = threadsPerBlock / blockDimX; // 256/8 = 32 threads along Y.
        tileSizeY = 64;                    // Tile covers more rows.
        tileSizeX = 32;                    // Tile covers fewer columns.
    } else if (N >= 2 * M) {
        // Wide output matrix: many more columns than rows.
        // Favor more threads in the X-dimension.
        blockDimX = 32;                    // More threads along X.
        blockDimY = threadsPerBlock / blockDimX; // 256/32 = 8 threads along Y.
        tileSizeY = 32;                    // Tile covers fewer rows.
        tileSizeX = 128;                   // Tile covers more columns.
    } else {
        // Nearly square output matrix.
        blockDimX = 16;
        blockDimY = threadsPerBlock / blockDimX; // 256/16 = 16.
        tileSizeY = 64;
        tileSizeX = 64;
    }

    // Verify that our chosen tile sizes are evenly divisible by the block dimensions.
    // (i.e. MICRO_TILE_ROWS = tileSizeY / blockDimY and MICRO_TILE_COLS = tileSizeX / blockDimX)
    // This is assumed by the kernel.
    assert(tileSizeY % blockDimY == 0 && "tileSizeY must be divisible by blockDimY");
    assert(tileSizeX % blockDimX == 0 && "tileSizeX must be divisible by blockDimX");

    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + tileSizeX - 1) / tileSizeX, (M + tileSizeY - 1) / tileSizeY);

    // Launch the kernel using a configuration that matches the shape.
    auto result = measurePerformance([&]() {
        if (M >= 2 * N) {
            // Tall configuration: instantiate kernel with parameters:
            // <BLOCK_DIM_X=8, BLOCK_DIM_Y=32, TILE_SIZE_Y=64, TILE_SIZE_X=32, TILE_SIZE_K=16>
            matrixMulTiledRect<8, 32, 64, 32, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        } else if (N >= 2 * M) {
            // Wide configuration: instantiate kernel with parameters:
            // <BLOCK_DIM_X=32, BLOCK_DIM_Y=8, TILE_SIZE_Y=32, TILE_SIZE_X=128, TILE_SIZE_K=16>
            matrixMulTiledRect<32, 8, 32, 128, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        } else {
            // Nearly square configuration: instantiate kernel with parameters:
            // <BLOCK_DIM_X=16, BLOCK_DIM_Y=16, TILE_SIZE_Y=64, TILE_SIZE_X=64, TILE_SIZE_K=16>
            matrixMulTiledRect<16, 16, 64, 64, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        }
        // Synchronize to ensure the kernel finishes.
        cudaStreamSynchronize(stream);
    }, M, N, K);

    // Asynchronously copy the result back to host.
    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Wait for all operations to complete.

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
    // Allocate and copy device memory.`
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    // Create a CUDA stream.
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Asynchronously copy data to the device.
    cudaMemcpyAsync(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice, stream);

    // --- Dynamically choose tile sizes and block dimensions ---
    // Here we dispatch three cases. For large matrices (e.g. 3000×4000, 100×10000)
    // we typically have M >= 64 and N >= 128.
    int tileSizeY, tileSizeX, tileSizeK = 16;
    int blockDimX, blockDimY;
    const int threadsPerBlock = 256;

    if (M >= 2 * N) {
        // Tall output matrix: many more rows than columns.
        // Favor more threads in the Y-dimension to cover extra rows.
        blockDimX = 8;                     // Fewer threads along X.
        blockDimY = threadsPerBlock / blockDimX; // 256/8 = 32 threads along Y.
        tileSizeY = 64;                    // Tile covers more rows.
        tileSizeX = 32;                    // Tile covers fewer columns.
    } else if (N >= 2 * M) {
        // Wide output matrix: many more columns than rows.
        // Favor more threads in the X-dimension.
        blockDimX = 32;                    // More threads along X.
        blockDimY = threadsPerBlock / blockDimX; // 256/32 = 8 threads along Y.
        tileSizeY = 32;                    // Tile covers fewer rows.
        tileSizeX = 128;                   // Tile covers more columns.
    } else {
        // Nearly square output matrix.
        blockDimX = 16;
        blockDimY = threadsPerBlock / blockDimX; // 256/16 = 16.
        tileSizeY = 64;
        tileSizeX = 64;
    }

    // Verify that our chosen tile sizes are evenly divisible by the block dimensions.
    // (i.e. MICRO_TILE_ROWS = tileSizeY / blockDimY and MICRO_TILE_COLS = tileSizeX / blockDimX)
    // This is assumed by the kernel.
    assert(tileSizeY % blockDimY == 0 && "tileSizeY must be divisible by blockDimY");
    assert(tileSizeX % blockDimX == 0 && "tileSizeX must be divisible by blockDimX");

    dim3 blockDim(blockDimX, blockDimY);
    dim3 gridDim((N + tileSizeX - 1) / tileSizeX, (M + tileSizeY - 1) / tileSizeY);

    // Launch the kernel using a configuration that matches the shape.
    auto result = measurePerformance([&]() {
        if (M >= 2 * N) {
            // Tall configuration: instantiate kernel with parameters:
            // <BLOCK_DIM_X=8, BLOCK_DIM_Y=32, TILE_SIZE_Y=64, TILE_SIZE_X=32, TILE_SIZE_K=16>
            matrixMulTiledRect<8, 32, 64, 32, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        } else if (N >= 2 * M) {
            // Wide configuration: instantiate kernel with parameters:
            // <BLOCK_DIM_X=32, BLOCK_DIM_Y=8, TILE_SIZE_Y=32, TILE_SIZE_X=128, TILE_SIZE_K=16>
            matrixMulTiledRect<32, 8, 32, 128, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        } else {
            // Nearly square configuration: instantiate kernel with parameters:
            // <BLOCK_DIM_X=16, BLOCK_DIM_Y=16, TILE_SIZE_Y=64, TILE_SIZE_X=64, TILE_SIZE_K=16>
            matrixMulTiledRect<16, 16, 64, 64, 16>
                <<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
        }
        // Synchronize to ensure the kernel finishes.
        cudaStreamSynchronize(stream);
    }, M, N, K);
    //copy results back to host
    cudaMemcpyAsync(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream); // Wait for all operations to complete.

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
    cudaStreamDestroy(stream);
    freeDeviceMemory(d_A, d_B, d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return result;
}


#endif
