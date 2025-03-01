#include <stdio.h>
#include <stdlib.h>
#include "matrix_multiply_shared.cuh"
#include "matrix_multiply_naive.cuh"
#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    if (argc != 5 || strcmp(argv[1], "-i") != 0) {
        fprintf(stderr, "Usage: %s -i <rowDimA> <colDimA> <colDimB>\n", argv[0]);
        return -1;
    }
    
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    int blockSizes[] = {8, 16, 32};
    int numBlocks = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;
    
    printf("\nMatrix Multiplication Performance Comparison:\n");
    printf("%-12s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s\n", "Block Size", "Shared TFLOPS", "Shared Time (ms)", "Naive TFLOPS", "Naive Time (ms)", "Theoretical Warps", "Achieved Warps", "Theoretical Occupancy", "Achieved Occupancy");
    
    for (int i = 0; i < numBlocks; i++) {
        int blockSize = blockSizes[i];
        int numBlocksPerSM;
        
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, matrixMultiplyShared, blockSize * blockSize, 0);
        
        int maxThreadsPerSM;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;
        int warpSize = prop.warpSize;
        
        int theoreticalWarps = maxThreadsPerSM / warpSize;
        int achievedWarps = (blockSize * blockSize * numBlocksPerSM) / warpSize;
        float theoreticalOccupancy = (float)achievedWarps / theoreticalWarps * 100.0f;
        float achievedOccupancy = (float)numBlocksPerSM / (prop.maxThreadsPerMultiProcessor / (blockSize * blockSize)) * 100.0f;

        // Shared Memory Kernel Execution
        float sharedTime = 0.0f;
        dim3 gridDim((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
        dim3 blockDim(blockSize, blockSize);
        matrixMultiplyShared<<<gridDim, blockDim>>>(A, B, C, M, K, N, blockSize, &sharedTime);
        cudaDeviceSynchronize();
        float sharedTflops = (2.0f * M * K * N) / (sharedTime * 1.0e6);
        
        // Naive Kernel Execution
        float naiveTime = 0.0f;
        dim3 gridDim_naive((N + blockSize - 1) / blockSize, (M + blockSize - 1) / blockSize);
        dim3 blockDim_naive(blockSize, blockSize);
        matrixMultiplyNaive<<<gridDim_naive, blockDim_naive>>>(A, B, C, M, K, N, blockSize, &naiveTime);
        cudaDeviceSynchronize();
        float naiveTflops = (2.0f * M * K * N) / (naiveTime * 1.0e6);
        
        printf("%-12d %-20.3f %-20.3f %-20.3f %-20.3f %-20d %-20d %-20.2f %-20.2f\n", blockSize, sharedTflops, sharedTime, naiveTflops, naiveTime, theoreticalWarps, achievedWarps, theoreticalOccupancy, achievedOccupancy);
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
