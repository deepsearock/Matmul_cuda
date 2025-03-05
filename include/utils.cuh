#ifndef UTILS_CUH
#define UTILS_CUH

#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <random>

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

// unused could do random 0.0 to 1.0
void populateMatrix(float *matrix, int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0, 1.0);
    
    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = dist(gen);
    }
}

// error checking for cuda i see this a lot
inline void checkCuda(cudaError_t result, const char *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << " in " << func << " (" << cudaGetErrorString(result) << ")\n";
        exit(EXIT_FAILURE);
    }
}


void matrixMulCPU(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// print to terminal gpu specs
void printGpuSpecs() {
    cudaDeviceProp mygpu;
    cudaGetDeviceProperties(&mygpu, 0);

    std::cout << "GPU Specifications:" << std::endl;
    std::cout << "  Name: " << mygpu.name << std::endl;
    std::cout << "  CUDA Cores per SM: " << mygpu.multiProcessorCount << std::endl;
    std::cout << "  Number of SMs: " << mygpu.multiProcessorCount << std::endl;
    std::cout << "  GPU Clock Rate (MHz): " << mygpu.clockRate / 1000.0 << std::endl;
    std::cout << "  GPU Memory Clock Rate (Mhz): " <<mygpu.memoryClockRate / 1000.0 << std::endl;
    std::cout << "  Memory Bus Width: " << mygpu.memoryBusWidth << std::endl;
    std::cout << "  Memory Bandwidth (GB/s): " << mygpu.memoryBusWidth * mygpu.memoryClockRate * 2 / 1.0e6 << std::endl;
}


void gpuselect(int device) {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable device found!" << std::endl;
        exit(1);
    }

    std::cout << "Available CUDA devices:\n";
    cudaDeviceProp prop;
    for (int i = 0; i < deviceCount; i++) {
        cudaGetDeviceProperties(&prop, i);
        std::cout << "  GPU " << i << ": " << prop.name << " (Compute Capability: " 
                  << prop.major << "." << prop.minor << ")\n";
    }

    // Select GPU
    int selectedDevice = device;  // Change this value if needed
    cudaSetDevice(selectedDevice);
    cudaGetDeviceProperties(&prop, selectedDevice);
    std::cout << "Using GPU " << selectedDevice << ": " << prop.name << "\n";
}

// memory allocation function on gpu
inline void allocateDeviceMemory(float **d_A, float **d_B, float **d_C, int M, int N, int K) {
    checkCudaErrors(cudaMalloc((void**)d_A, M * K * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)d_B, K * N * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)d_C, M * N * sizeof(float)));
}

// function to free memory on gpu
inline void freeDeviceMemory(float *d_A, float *d_B, float *d_C) {
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
}

// performance function mainly calcs tflops
inline std::pair<double, double> measurePerformance(std::function<void()> kernelLaunch, int M, int N, int K) {
    cudaEvent_t start, stop;
    float timeMs = 0.0f;

    // create some cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    kernelLaunch();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // calculate the elapsed time
    cudaEventElapsedTime(&timeMs, start, stop);

    // destroy the events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // tflops calc
    double numOps = 2.0 * M * N * K;
    double tflops = (numOps / (timeMs * 1e9));  // Convert to TFLOPS (1e6 for milliseconds)

    return {tflops, timeMs};
}

#endif
