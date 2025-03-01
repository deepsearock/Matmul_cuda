#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime_api.h>
#include "matrix_multiply_shared.cuh"
#include "matrix_multiply_naive.cuh"

#include <cuda_runtime.h>

int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s, Compute Capability: %d.%d, Global Memory: %.2f GB, Shared Memory per Block: %d KB, Max Threads per Block: %d \n\n", prop.name, prop.major, prop.minor, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0), (int)(prop.sharedMemPerBlock / 1024), (int)prop.maxThreadsPerBlock);
    
    int M = atoi(argv[2]);
    int K = atoi(argv[3]);
    int N = atoi(argv[4]);
    int blockSizes[] = {8, 16, 32};
    int numBlocks = sizeof(blockSizes) / sizeof(blockSizes[0]);
    
    float *A = (float *)malloc(M * K * sizeof(float));
    float *B = (float *)malloc(K * N * sizeof(float));
    float *C = (float *)malloc(M * N * sizeof(float));
    
    for (int i = 0; i < M * K; i++) A[i] = (float)(rand() % 100) / 100.0f;
    for (int i = 0; i < K * N; i++) B[i] = (float)(rand() % 100) / 100.0f;
    
    printf("Matrix Multiplication Performance Comparison:");
    printf("%-12s %-20s %-20s %-20s %-20s %-20s %-20s %-20s %-20s", "Block Size", "Shared TFLOPS", "Shared Time (ms)", "Naive TFLOPS", "Naive Time (ms)", "Theor. Warps", "Ach. Warps", "Theor. Occ. (%)", "Ach. Occ. (%)");
    printf("%-12s %-20s %-20s %-20s %-20s\n", "Block Size", "Shared TFLOPS", "Shared Time (ms)", "Naive TFLOPS", "Naive Time (ms)");
    
    for (int i = 0; i < numBlocks; i++) {
        int BLOCK_SIZE = blockSizes[i];
        float execTimeShared = 0.0f, execTimeNaive = 0.0f;
        double tflopsShared = matrixMultiplyShared(A, B, C, M, N, K, BLOCK_SIZE, &execTimeShared);
        double tflopsNaive = matrixMultiplyNaive(A, B, C, M, N, K, BLOCK_SIZE, &execTimeNaive);
        
        int maxThreadsPerSM;
        int warpSize;
        cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
        cudaDeviceGetAttribute(&warpSize, cudaDevAttrWarpSize, 0);
        int maxWarpsPerSM = maxThreadsPerSM / warpSize;
        int theoreticalWarps = maxWarpsPerSM;
        
        int activeWarpsPerSM;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&activeWarpsPerSM, matrixMultiplyShared, BLOCK_SIZE * BLOCK_SIZE, 0);
        double achievedWarps = (double)activeWarpsPerSM;
        
        int minGridSize;
        int blockSize;
        cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, matrixMultiplyShared, 0, 0);
        int maxBlocksPerSM = minGridSize;
        double theoreticalOccupancy = ((double)maxBlocksPerSM * BLOCK_SIZE * BLOCK_SIZE) / maxThreadsPerSM * 100.0;
        
        int achievedActiveThreads;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&achievedActiveThreads, matrixMultiplyShared, BLOCK_SIZE * BLOCK_SIZE, 0);
        double achievedOccupancy = (double)achievedActiveThreads / maxBlocksPerSM * 100.0;
        
        
        
        printf("%-12d %-20.2f %-20.2f %-20.2f %-20.2f %-20d %-20.2f %-20.2f %-20.2f\n", BLOCK_SIZE, tflopsShared, execTimeShared, tflopsNaive, execTimeNaive, theoreticalWarps, achievedWarps, theoreticalOccupancy, achievedOccupancy);
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
