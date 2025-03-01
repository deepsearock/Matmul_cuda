#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Kernel for matrix multiplication using shared memory and tiling
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K, int BLOCK_SIZE) {
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    extern __shared__ float sharedMem[];
    float *As = sharedMem;
    float *Bs = &sharedMem[BLOCK_SIZE * BLOCK_SIZE];
    
    float Cvalue = 0.0f;
    
    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        int globalArow = blockRow * BLOCK_SIZE + threadRow;
        int globalAcol = m * BLOCK_SIZE + threadCol;
        int globalBrow = m * BLOCK_SIZE + threadRow;
        int globalBcol = blockCol * BLOCK_SIZE + threadCol;
        
        // Handle boundary conditions to prevent out-of-bounds memory access
        if (globalArow < M && globalAcol < K)
            As[threadRow * BLOCK_SIZE + threadCol] = A[globalArow * K + globalAcol];
        else
            As[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        
        if (globalBrow < K && globalBcol < N)
            Bs[threadRow * BLOCK_SIZE + threadCol] = B[globalBrow * N + globalBcol];
        else
            Bs[threadRow * BLOCK_SIZE + threadCol] = 0.0f;
        
        __syncthreads();
        
        // Perform matrix multiplication for the tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadRow * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + threadCol];
        }
        
        __syncthreads();
    }
    
    int globalCrow = blockRow * BLOCK_SIZE + threadRow;
    int globalCcol = blockCol * BLOCK_SIZE + threadCol;
    
    // Ensure the output is written only within valid matrix bounds
    if (globalCrow < M && globalCcol < N) {
        C[globalCrow * N + globalCcol] = Cvalue;
    }
}

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

// Host function to manage memory and invoke the kernel
double matrixMultiply(float *A, float *B, float *C, int M, int N, int K, int BLOCK_SIZE) {
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
    int sharedMemSize = 2 * BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    matrixMulShared<<<dimGrid, dimBlock, sharedMemSize>>>(d_A, d_B, d_C, M, N, K, BLOCK_SIZE);
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    double ops = 2.0 * M * N * K;
    double tflops = (ops / (milliseconds / 1000.0)) / 1e12;
    
    checkCudaError(cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy d_C->C failed");
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return tflops;
}

int main(int argc, char *argv[]) {
    if (argc != 5 || strcmp(argv[1], "-i") != 0) {
        fprintf(stderr, "Usage: %s -i <rowDimA> <colDimA> <colDimB>\n", argv[0]);
        return -1;
    }
    
    int M = atoi(argv[2]); // Rows of A
    int K = atoi(argv[3]); // Columns of A, rows of B
    int N = atoi(argv[4]); // Columns of B
    
    int BLOCK_SIZE = 32; // Default block size
    
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)(rand() % 100) / 100.0f;
    }
    
    printf("Running matrix multiplication with dimensions A(%d x %d), B(%d x %d)\n", M, K, K, N);
    double tflops = matrixMultiply(A, B, C, M, N, K, BLOCK_SIZE);
    printf("Execution Time: %f ms, Performance: %f TFLOPS\n", tflops * 1e12 / (2.0 * M * N * K), tflops);
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
