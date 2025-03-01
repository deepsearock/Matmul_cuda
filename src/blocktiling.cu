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

int main(int argc, char *argv[]) {
    if (argc != 5 || strcmp(argv[1], "-i") != 0) {
        fprintf(stderr, "Usage: %s -i <rowDimA> <colDimA> <colDimB>\n", argv[0]);
        return -1;
    }
    
    int M = atoi(argv[2]); // Rows of A
    int K = atoi(argv[3]); // Columns of A, rows of B
    int N = atoi(argv[4]); // Columns of B
    
    int blockSizes[] = {8, 16, 32};
    int numBlocks = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M * K; i++) {
        A[i] = (float)(rand() % 100) / 100.0f;
    }
    for (int i = 0; i < K * N; i++) {
        B[i] = (float)(rand() % 100) / 100.0f;
    }
    
    for (int b = 0; b < numBlocks; b++) {
        int BLOCK_SIZE = blockSizes[b];
        double totalTime = 0.0;
        double totalTflops = 0.0;
        int runs = 100;
        
        for (int i = 0; i < runs; i++) {
            cudaEvent_t start, stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            
            double tflops = matrixMultiply(A, B, C, M, N, K, BLOCK_SIZE);
            
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            
            float milliseconds = 0;
            cudaEventElapsedTime(&milliseconds, start, stop);
            
            totalTime += milliseconds;
            totalTflops += tflops;
            
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
        
        double avgTime = totalTime / runs;
        double avgTflops = totalTflops / runs;
        
        printf("Block Size: %d, Average Execution Time: %f ms, Average Performance: %f TFLOPS\n", BLOCK_SIZE, avgTime, avgTflops);
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
