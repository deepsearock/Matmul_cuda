#ifndef MATRIX_MULTIPLY_CPU_CUH
#define MATRIX_MULTIPLY_CPU_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include "utils.cuh"

// Naive CUDA kernel for matrix multiplication using only global memory
void matrixMulCPU(float *A, float *B, float *C, int M, int N, int K) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to allocate memory, launch the naive kernel, and measure performance
inline std::pair<double, double> runMatrixMulCPU(int M, int N, int K) {
    std::vector<std::vector<float>> A(M, std::vector<float>(N));
    std::vector<std::vector<float>> B(N, std::vector<float>(K));
    std::vector<std::vector<float>> C(M, std::vector<float>(K, 0));

    generateMatrix(A, M, N);
    generateMatrix(B, N, K);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double time_sec = duration.count();
    double num_operations = 2.0 * M * N * K; // Total FLOPs in matrix multiplication
    double tflops = (num_operations / (time_sec * 1e12)); // Convert to TFLOPS

    return {time_sec, tflops};
}


#endif // MATRIX_MULTIPLY_CPU_CUH

