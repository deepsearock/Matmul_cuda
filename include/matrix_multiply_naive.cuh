#ifndef MATRIX_MULTIPLY_NAIVE_CUH
#define MATRIX_MULTIPLY_NAIVE_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"
#include <random>

// naive 
__global__ void matrixMulGlobalNaive(float *A, float *B, float *C, int M, int N, int K) {
    
    //calculate index
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // check if they are multiples of block
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// wrapper function that measures performance and does memory management
inline std::pair<double, double> runMatrixMulNaive(int M, int N, int K, int blockWidth, int blockHeight) {

    float *d_A, *d_B, *d_C;
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    
    dim3 blockDim(blockWidth, blockHeight, 1);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y, 1);

    auto result = measurePerformance([&]() { matrixMulGlobalNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K); }, M, N, K);
    
    freeDeviceMemory(d_A, d_B, d_C);

    return result;
}

inline std::pair<double, double> runMatrixMulNaiveWithErrorCheck(int M, int N, int K, int blockWidth, int blockHeight) {
    float *d_A, *d_B, *d_C;
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    // Initialize host matrices with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    allocateDeviceMemory(&d_A, &d_B, &d_C, M, N, K);
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(blockWidth, blockHeight);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch kernel and measure performance
    auto result = measurePerformance([&]() { matrixMulGlobalNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K); }, M, N, K);

    // Copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Compute reference result on CPU
    matrixMulCPU(h_A, h_B, h_C_ref, M, N, K);

    // Compute error metrics
    double mse = 0.0, max_error = 0.0;
    for (int i = 0; i < M * N; ++i) {
        double diff = fabs(h_C[i] - h_C_ref[i]);
        mse += diff * diff;
        max_error = std::max(max_error, diff);
    }
    mse /= (M * N);
    double error_threshold = 1e-5; // Define an acceptable error threshold
int error_count = 0;

for (int i = 0; i < M * N; ++i) {
    double diff = fabs(h_C[i] - h_C_ref[i]);
    if (diff > error_threshold) { // Consider it an error if the difference is too large
        error_count++;
    }
}

double error_percentage = (error_count * 100.0) / (M * N);

std::cout << "Error Percentage: " << error_percentage << "%" << std::endl;
    
    // Print error results
    std::cout << "Mean Squared Error: " << mse << std::endl;
    std::cout << "Max Absolute Error: " << max_error << std::endl;

    // Free memory
    freeDeviceMemory(d_A, d_B, d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    return result;
}

#endif

