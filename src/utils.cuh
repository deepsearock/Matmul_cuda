#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <functional>
#include <random>

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)


void populateMatrix(float *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

inline void checkCuda(cudaError_t result, const char *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " in " << func << " (" << cudaGetErrorString(result) << ")\n";
        exit(EXIT_FAILURE);
    }
}

// Function to allocate memory for matrices
inline void allocateDeviceMemory(float **d_A, float **d_B, float **d_C, int M, int N, int K) {
    checkCudaErrors(cudaMalloc((void**)d_A, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)d_B, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)d_C, M * N * sizeof(float)));
}

// Function to free allocated memory
inline void freeDeviceMemory(float *d_A, float *d_B, float *d_C) {
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}

// Function to measure execution time and calculate TFLOPS
inline std::pair<double, double> measurePerformance(std::function<void()> kernelLaunch, int M, int N, int K) {

    // Record start time
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch the kernel
    kernelLaunch();
    
    // Synchronize device and record end time
    checkCudaErrors(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    std::chrono::duration<double, std::milli> duration = end - start;
    double timeMs = duration.count();

    // Calculate floating-point operations and performance in TFLOPS
    double numOps = 2.0 * M * N * K; // Floating point operations (for matrix multiplication)
    double tflops = (numOps / (timeMs * 1e9));

    return {tflops, timeMs};  // Return the performance and execution time
}

#endif // UTILS_CUH
