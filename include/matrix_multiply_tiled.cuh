#ifndef MATRIX_MULTIPLY_TILED_CUH
#define MATRIX_MULTIPLY_TILED_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"
#include <cuda_fp16.h>
#include <mma.h>

template <int TILE_SIZE>
__global__ void matrixMulTiled(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  
    __shared__ float tileB[TILE_SIZE][TILE_SIZE + 1];

    // Compute global thread coordinates
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Register accumulation for better performance
    float sum = 0.0f;

    // Iterate over tiles
    for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; ++tileIdx) {
        // Load tiles into shared memory using vectorized loads for coalesced memory access
        int tiledRowA = row;
        int tiledColA = tileIdx * TILE_SIZE + threadIdx.x;
        int tiledRowB = tileIdx * TILE_SIZE + threadIdx.y;
        int tiledColB = col;

        if (tiledRowA < M && tiledColA < K) {
            tileA[threadIdx.y][threadIdx.x] = A[tiledRowA * K + tiledColA];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tiledRowB < K && tiledColB < N) {
            tileB[threadIdx.y][threadIdx.x] = B[tiledRowB * N + tiledColB];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // Synchronize before computation

        // Use loop unrolling for TILE_SIZE = 32
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
            sum += tileA[threadIdx.y][k+1] * tileB[k+1][threadIdx.x];
            sum += tileA[threadIdx.y][k+2] * tileB[k+2][threadIdx.x];
            sum += tileA[threadIdx.y][k+3] * tileB[k+3][threadIdx.x];
        }

        __syncthreads(); // Synchronize before loading next tiles
    }

    // Store results back in global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }

}

template <int TILE_SIZE>
__global__ void matrixMulTensorCore(half *A, half *B, float *C, int M, int N, int K) {
    // Define WMMA tile size
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    // Compute tile position
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / 16;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / 16;

    // Shared memory for input matrices
    __shared__ half tileA[TILE_SIZE][TILE_SIZE];
    __shared__ half tileB[TILE_SIZE][TILE_SIZE];

    // Accumulator register
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // Iterate over tiles
    for (int tileIdx = 0; tileIdx < (K + TILE_SIZE - 1) / TILE_SIZE; tileIdx++) {
        // Load tiles into shared memory
        int row = warpM * TILE_SIZE + threadIdx.y;
        int col = warpN * TILE_SIZE + threadIdx.x;

        if (row < M && (tileIdx * TILE_SIZE + threadIdx.x) < K) {
            tileA[threadIdx.y][threadIdx.x] = A[row * K + tileIdx * TILE_SIZE + threadIdx.x];
        } else {
            tileA[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        if ((tileIdx * TILE_SIZE + threadIdx.y) < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = __float2half(0.0f);
        }

        __syncthreads();

        // Load fragments from shared memory to registers
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;

        wmma::load_matrix_sync(a_frag, &tileA[threadIdx.y][0], TILE_SIZE);
        wmma::load_matrix_sync(b_frag, &tileB[0][threadIdx.x], TILE_SIZE);

        // Matrix multiply and accumulate
        wmma::mma_sync(acc, a_frag, b_frag, acc);
        __syncthreads();
    }

    // Store result back to global memory
    if (warpM * TILE_SIZE + threadIdx.y < M && warpN * TILE_SIZE + threadIdx.x < N) {
        wmma::store_matrix_sync(&C[(warpM * TILE_SIZE + threadIdx.y) * N + (warpN * TILE_SIZE + threadIdx.x)], acc, TILE_SIZE, wmma::mem_row_major);
    }
}


// wrapper function that measures performance and does memory management
inline std::pair<double, double> runMatrixMulTiled(int M, int N, int K, int TILE_SIZE) {
    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    dim3 blockSize;
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    // launch kernel
    auto result = measurePerformance([&]() {
        switch (TILE_SIZE) {
            case 8:
                blockSize = dim3(8, 8);
                matrixMulTiled<8><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 16:
                blockSize = dim3(16, 16);
                matrixMulTiled<16><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
                break;
            case 32:
                blockSize = dim3(32, 8);
                matrixMulTiled<32><<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
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
