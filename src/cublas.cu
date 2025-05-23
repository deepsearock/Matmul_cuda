#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cublas_v2.h>
#include <cuda_runtime.h>

void matrixMultiplyCUBLAS(float* A, float* B, float* C, int M, int N, int K) {
    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f;
    float beta = 0.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M * N * sizeof(float));
    cudaMalloc((void**)&d_B, N * K * sizeof(float));
    cudaMalloc((void**)&d_C, M * K * sizeof(float));

    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_ref = new float[M * N];

    //initialize host memory with random values
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

    //allocate and copy memory to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, d_B, K, d_A, N,  &beta, d_C, K);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    //copy results back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_ref;

    cublasDestroy(handle);
}


//quick cublas implementation to check for performance
int main(int argc, char* argv[]) {

    if (argc != 5 || std::string(argv[1]) != "-i") {
        std::cerr << "Usage: " << argv[0] << " -i <rowDimA> <colDimA> <colDimB>" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[2]);
    int N = std::atoi(argv[3]);
    int K = std::atoi(argv[4]);

    float* A = new float[M * N];
    float* B = new float[N * K];
    float* C = new float[M * K];

    for (int i = 0; i < M * N; ++i) A[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < N * K; ++i) B[i] = static_cast<float>(rand()) / RAND_MAX;

    auto start = std::chrono::high_resolution_clock::now();
    matrixMultiplyCUBLAS(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double time_sec = duration.count();
    double num_operations = 2.0 * M * N * K;
    double tflops = num_operations / (time_sec * 1e12);

    std::cout << "Matrix multiplication took " << time_sec << " seconds." << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS." << std::endl;

    delete[] A;
    delete[] B;
    delete[] C;
    
    return 0;
}
