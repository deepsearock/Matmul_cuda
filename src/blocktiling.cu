#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Block size (tile size)
#define BLOCK_SIZE 16

// Kernel for matrix multiplication using block tiling
__global__ void matrixMulShared(float *A, float *B, float *C, int M, int N, int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    
    // Thread row and column within the block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    
    // Allocate shared memory for the sub-matrices of A and B
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Each thread computes one element of C
    float Cvalue = 0.0f;
    
    // Loop over all sub-matrices of A and B
    for (int m = 0; m < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++m) {
        // Load sub-matrices from global memory to shared memory
        
        // Global indices for the current sub-matrices
        int globalArow = blockRow * BLOCK_SIZE + threadRow;
        int globalAcol = m * BLOCK_SIZE + threadCol;
        int globalBrow = m * BLOCK_SIZE + threadRow;
        int globalBcol = blockCol * BLOCK_SIZE + threadCol;
        
        // Load the matrices from global memory to shared memory
        // Each thread loads one element of each sub-matrix
        if (globalArow < M && globalAcol < K) {
            As[threadRow][threadCol] = A[globalArow * K + globalAcol];
        } else {
            As[threadRow][threadCol] = 0.0f;
        }
        
        if (globalBrow < K && globalBcol < N) {
            Bs[threadRow][threadCol] = B[globalBrow * N + globalBcol];
        } else {
            Bs[threadRow][threadCol] = 0.0f;
        }
        
        // Synchronize to ensure all data is loaded
        __syncthreads();
        
        // Multiply the two sub-matrices together
        // Each thread computes one element of the sub-matrix
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            Cvalue += As[threadRow][k] * Bs[k][threadCol];
        }
        
        // Synchronize to ensure that the preceding computation is done
        // before loading new sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    
    // Write the computed value to global memory
    // Each thread writes one element
    int globalCrow = blockRow * BLOCK_SIZE + threadRow;
    int globalCcol = blockCol * BLOCK_SIZE + threadCol;
    
    if (globalCrow < M && globalCcol < N) {
        C[globalCrow * N + globalCcol] = Cvalue;
    }
}

// Helper function for checking CUDA errors
void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

// Host code for matrix multiplication with timing and performance metrics
double matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Allocate device memory
    cudaError_t error = cudaMalloc((void **)&d_A, sizeA);
    checkCudaError(error, "cudaMalloc d_A failed");
    
    error = cudaMalloc((void **)&d_B, sizeB);
    checkCudaError(error, "cudaMalloc d_B failed");
    
    error = cudaMalloc((void **)&d_C, sizeC);
    checkCudaError(error, "cudaMalloc d_C failed");
    
    // Copy input matrices from host to device
    error = cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy A->d_A failed");
    
    error = cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);
    checkCudaError(error, "cudaMemcpy B->d_B failed");
    
    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Record start event
    cudaEventRecord(start, 0);
    
    // Launch the kernel
    matrixMulShared<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    
    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    // Check for kernel launch errors
    error = cudaGetLastError();
    checkCudaError(error, "Kernel launch failed");
    
    // Calculate elapsed time
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    
    // Convert to seconds
    elapsedTime /= 1000.0f;
    
    // Calculate performance in TFLOPS
    // For matrix multiplication, we perform M*N*K*2 operations (MNK multiplications and MNK additions)
    double ops = 2.0 * M * N * K;  // Number of floating-point operations
    double tflops = (ops / elapsedTime) / 1e12;  // Convert to TFLOPS
    
    printf("Matrix multiplication performance:\n");
    printf("Matrix dimensions: A(%d x %d), B(%d x %d), C(%d x %d)\n", M, K, K, N, M, N);
    printf("Block size: %d x %d\n", BLOCK_SIZE, BLOCK_SIZE);
    printf("Execution time: %f seconds\n", elapsedTime);
    printf("Performance: %f TFLOPS\n", tflops);
    
    // Copy the result from device to host
    error = cudaMemcpy(C, d_C, sizeC, cudaMemcpyDeviceToHost);
    checkCudaError(error, "cudaMemcpy d_C->C failed");
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    // Destroy events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return tflops;
}

// Run multiple tests with different matrix sizes and compute average performance
void runPerformanceTests() {
    printf("Running performance tests for matrix multiplication...\n");
    
    // Array of matrix sizes to test
    int sizes[] = {1024, 2048, 4096};
    int numSizes = sizeof(sizes) / sizeof(sizes[0]);
    
    double totalTflops = 0.0;
    
    for (int i = 0; i < numSizes; i++) {
        int size = sizes[i];
        
        printf("\n======= Test with %d x %d matrices =======\n", size, size);
        
        // Allocate host memory
        float *A = (float *)malloc(size * size * sizeof(float));
        float *B = (float *)malloc(size * size * sizeof(float));
        float *C = (float *)malloc(size * size * sizeof(float));
        
        // Initialize matrices A and B with random values
        for (int j = 0; j < size * size; j++) {
            A[j] = (float)(rand() % 100) / 100.0f;
            B[j] = (float)(rand() % 100) / 100.0f;
        }
        
        // Run the test and measure performance
        double tflops = matrixMultiply(A, B, C, size, size, size);
        totalTflops += tflops;
        
        // Free host memory
        free(A);
        free(B);
        free(C);
    }
    
    // Report average performance
    printf("\n======= Performance Summary =======\n");
    printf("Average performance: %f TFLOPS\n", totalTflops / numSizes);
}

int main() {
    // Set random seed
    srand(42);
    
    // Run the performance tests
    runPerformanceTests();
    
    return 0;
}