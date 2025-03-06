#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include "utils.cuh"

template <int TILE_SIZE>
__global__ void matrixMulTiled(const float *__restrict__ A,
                               const float *__restrict__ B,
                               float *__restrict__ C,
                               int M, int N, int K) {
    // Each block computes a TILE_SIZE x TILE_SIZE sub-matrix of C.
    // Block row offset in C.
    int blockRow = blockIdx.y * TILE_SIZE;
    // Global column index for C; each thread handles one column in the tile.
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Determine how many output rows in the tile this thread computes.
    // Maximum possible is TILE_SIZE (when blockDim.y==1), but if more threads in y, then fewer rows per thread.
    int numRows = (TILE_SIZE + blockDim.y - 1) / blockDim.y;
    
    // Allocate a local accumulation array.
    // We allocate TILE_SIZE elements (the maximum needed) but only use the first 'numRows' entries.
    float accum[TILE_SIZE];
    #pragma unroll
    for (int i = 0; i < TILE_SIZE; i++) {
        accum[i] = 0.0f;
    }
    
    // Shared memory tiles.
    // For tileA, a TILE_SIZE x TILE_SIZE block.
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    // For tileB, pad by one column to reduce shared memory bank conflicts.
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];
    
    // Number of tiles in the K dimension.
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // --- Load tileA ---
        // Each thread loads multiple rows from A into shared memory.
        for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
            int row = blockRow + i;
            int colA = t * TILE_SIZE + threadIdx.x;
            tileA[i][threadIdx.x] = (row < M && colA < K) ? A[row * K + colA] : 0.0f;
        }
        
        // --- Load tileB ---
        // Each thread loads multiple rows from B into shared memory.
        for (int i = threadIdx.y; i < TILE_SIZE; i += blockDim.y) {
            int rowB = t * TILE_SIZE + i;
            tileB[i][threadIdx.x] = (rowB < K && col < N) ? B[rowB * N + col] : 0.0f;
        }
        
        __syncthreads();
        
        // --- Multiply the two tiles ---
        // Unroll the loop over the tile dimension.
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            float bVal = tileB[k][threadIdx.x];
            // Each thread computes 'numRows' output rows.
            for (int i = 0; i < numRows; i++) {
                int rowInTile = threadIdx.y + i * blockDim.y;
                if (rowInTile < TILE_SIZE) {
                    accum[i] += tileA[rowInTile][k] * bVal;
                }
            }
        }
        
        __syncthreads();
    }
    
    // --- Write results back to global memory ---
    for (int i = 0; i < numRows; i++) {
        int row = blockRow + threadIdx.y + i * blockDim.y;
        if (row < M && col < N) {
            C[row * N + col] = accum[i];
        }
    }
}



// wrapper function that measures performance and does memory management
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);

    int minGridSize, blockSize;

    // Determine block size dynamically based on tile size
    switch (tileSize) {
        case 8:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(tileSize, tileSize);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<gridDim, dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            default:
                std::cerr << "Unsupported tile size" << std::endl;
                exit(EXIT_FAILURE);
        }
    }, M, N, K);

    freeDeviceMemory(d_A, d_B, d_C);
    return result;
}

inline std::pair<double, double> runMatrixMulTiledWithErrorCheck(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    // Initialize host memory with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate and copy memory to device
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    int minGridSize, blockSize;

    // Determine block size dynamically based on tile size
    switch (tileSize) {
        case 8:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(tileSize, tileSize);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<gridDim, dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            default:
                std::cerr << "Unsupported tile size" << std::endl;
                exit(EXIT_FAILURE);
        }
    }, M, N, K);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute CPU reference result
    matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Compute error metrics
    double mse = 0.0, max_error = 0.0;
    int error_count = 0;
    double error_threshold = 1e-3; // Acceptable error threshold

    for (int i = 0; i < M * N; ++i) {
        double diff = fabs(h_C[i] - h_C_ref[i]);
        mse += diff * diff;
        max_error = std::max(max_error, diff);
        if (diff > error_threshold) error_count++;
    }
    mse /= (M * N);
    double error_percentage = (error_count * 100.0) / (M * N);

    // Print error results
    std::cout << "Error Percentage: " << error_percentage << "%" << std::endl;
    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Max Absolute Error: " << max_error << std::endl;

    // Clean up
    freeDeviceMemory(d_A, d_B, d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return result;
}


#endif
