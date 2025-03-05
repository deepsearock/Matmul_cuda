#ifndef MATRIX_MULTIPLY_CPU_CUH
#define MATRIX_MULTIPLY_CPU_CUH

#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <random>

// CPU implementation of matrix multiplication
inline void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            C[i * K + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * K + j] += A[i * N + k] * B[k * K + j];
            }
        }
    }
}

// Function to generate a random matrix

inline void generateMatrix(float* matrix, int rows, int cols) {
    std::random_device rd;  
    std::mt19937 gen(rd()); 
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            matrix[i * cols + j] = dis(gen); // Generates a float in range [0,1]
        }
    }
}

// Function to run matrix multiplication and return execution time and TFLOPS
inline std::pair<double, double> runMatrixMulCPU(int M, int N, int K) {
    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    generateMatrix(A, M, N);
    generateMatrix(B, N, K);

    auto start = std::chrono::high_resolution_clock::now();
    matrixMulCPU(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double time_sec = duration.count();
    double num_operations = 2.0 * M * N * K; // Total FLOPs in matrix multiplication
    double tflops = (num_operations / (time_sec * 1e12)); // Convert to TFLOPS

    delete[] A;
    delete[] B;
    delete[] C;

    return {time_sec, tflops};
}

#endif
