#ifndef MATRIX_MULTIPLY_NAIVE_CUH
#define MATRIX_MULTIPLY_NAIVE_CUH

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// Kernel for naive matrix multiplication using only global memory (no tiling)
__global__ void matrixMulGlobalNaive(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float Cvalue = 0.0f;
        for (int k = 0; k < K; k++) {
            Cvalue += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = Cvalue;
    }
}

// Host function to manage memory and execute naive global memory matrix multiplication
double matrixMultiplyNaive(float *A, float *B, float *C, int M, int N, int K, int BLOCK_SIZE, float *executionTime) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    checkCudaError(cudaMalloc((void **)&d_A, sizeA), "cudaMalloc d_A failed");
    checkCudaError(cudaMalloc((void **)&d_B, sizeB), "cudaMalloc d_B failed");
    checkCudaError(cudaMalloc((void **)&d_C, sizeC), "cudaMalloc d_C failed");
    
    checkCudaError(cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice), "cudaMemcpy A->d_A failed");
    checkCudaError(cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice), "cudaMemcpy B->d_B failed");
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixMulGlobalNaive<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    cudaEventElapsedTime(executionTime, start, stop);
    
    double ops = 2.0 * M * N * K;
    double tflops = (ops / (*executionTime / 1000.0)) / 1e12;
    
    checkCudaError(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy d_C->C failed");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return tflops;
}

#endif // MATRIX_MULTIPLY_NAIVE_CUH
