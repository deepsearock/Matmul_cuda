#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"

// Tiled CUDA kernel for matrix multiplication using shared memory
template <int TILE_SIZE>
__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    
    // Shared memory tiles
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    // Compute row and column index for the output matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Initialize accumulator
    float sum = 0.0f;

    // Iterate over tiles
    for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {

        // Compute indices for loading tiles
        int tiledRowA = row;
        int tiledColA = tileIdx * TILE_SIZE + threadIdx.x;

        int tiledRowB = tileIdx * TILE_SIZE + threadIdx.y;
        int tiledColB = col;

        // Load tiles from global memory into shared memory with boundary checks
        if (tiledRowA < M && tiledColA < K)
            tileA[threadIdx.y][threadIdx.x] = A[tiledRowA * K + tiledColA];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f; // Prevents undefined behavior

        if (tiledRowB < K && tiledColB < N)
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + tiledColB];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        // Synchronize to ensure tiles are loaded before computation
        __syncthreads();

        // Compute partial sum using only valid elements
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            if ((tileIdx * TILE_SIZE + k) < K) // Ensures valid multiplication
                sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading new tiles
        __syncthreads();
    }

    // Store the computed value in global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// wrapper function that measures performance and does memory management
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K, int tileSize) {
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);

    // launch kernel
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
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

    // Launch the kernel
    auto result = measurePerformance([&]() {
        switch (tileSize) {
            case 8:
                matrixMulTiled<8><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                matrixMulTiled<16><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                matrixMulTiled<32><<<dim3((N + 31) / 32, (M + 31) / 32), dim3(32, 8)>>>(d_A, d_B, d_C, M, N, K);
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
    for (int i = 0; i < M * N; ++i) {
        double diff = fabs(h_C[i] - h_C_ref[i]);
        mse += diff * diff;
        max_error = std::max(max_error, diff);
    }
    mse /= (M * N);
    

    // Print error results
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
