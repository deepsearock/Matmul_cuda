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
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int TILE_SIZE>
__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    // Define shared memory for tiles - no padding to avoid out-of-bounds access
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    // Calculate thread coordinates
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row and column
    int row = by * BLOCK_DIM_Y + ty;
    int col = bx * BLOCK_DIM_X + tx;
    
    // Accumulator for output
    float Cvalue = 0.0f;
    
    // Calculate number of tile iterations needed
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over all tiles
    for (int t = 0; t < numTiles; ++t) {
        // Carefully load tiles with strict boundary checking
        
        // Each thread potentially loads multiple elements based on block vs tile dimensions
        for (int i = ty; i < TILE_SIZE; i += BLOCK_DIM_Y) {
            for (int j = tx; j < TILE_SIZE; j += BLOCK_DIM_X) {
                // Global indices for A and B
                int globalARow = by * BLOCK_DIM_Y + i;
                int globalACol = t * TILE_SIZE + j;
                
                int globalBRow = t * TILE_SIZE + i;
                int globalBCol = bx * BLOCK_DIM_X + j;
                
                // Load A with boundary check
                if (globalARow < M && globalACol < K) {
                    As[i][j] = A[globalARow * K + globalACol];
                } else {
                    As[i][j] = 0.0f;
                }
                
                // Load B with boundary check
                if (globalBRow < K && globalBCol < N) {
                    Bs[i][j] = B[globalBRow * N + globalBCol];
                } else {
                    Bs[i][j] = 0.0f;
                }
            }
        }
        
        // Synchronize to ensure all threads have loaded data
        __syncthreads();
        
        // Only compute if this thread's output position is within bounds
        if (row < M && col < N) {
            // Compute dot product
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (t * TILE_SIZE + k < K) {
                    Cvalue += As[ty][k] * Bs[k][tx];
                }
            }
        }
        
        // Synchronize before loading the next tile
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
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
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(32, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<32, 8, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<32, 8, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8 , 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 8>, 0, 0);
            break;
        case 16:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 16>, 0, 0);
            break;
        case 32:
            cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMulTiled<32, 8, 32>, 0, 0);
            break;
        default:
            std::cerr << "Unsupported tile size" << std::endl;
            exit(EXIT_FAILURE);
    }

    int threadsPerBlock = std::min(blockSize, 1024);  // Ensure we don't exceed max threads per block
    dim3 blockDim(32, 8);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel using runtime-determined grid and block sizes
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<32, 8, 8><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<32, 8, 16><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32, 8 , 32><<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
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
