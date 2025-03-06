#include <iostream>
#include <cstdlib>
#include <ctime>
#include "utils.cuh"
#include "matrix_multiply_naive.cuh"
#include "matrix_multiply_tiled.cuh"
#include <unistd.h>
int main() {
    // Set matrix dimensions
    int M = 1564;  // Number of rows in A and C
    int N = 1567;  // Number of columns in B and C
    int K = 1567;  // Number of columns in A and rows in B

    int tileSize = 32;  // Tile size for tiled matrix multiplication

    printGpuSpecs();
    gpuselect(4);

    std::cout << "==========================================" << std::endl;
    std::cout << "      MATRIX MULTIPLICATION TEST" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Matrix Dimensions: " << M << " x " << N << " x " << K << std::endl;
    // Run tiled matrix multiplication with error check
    std::cout << "\nRunning Tiled GPU Matrix Multiplication (Tile Size = " << tileSize << ")..." << std::endl;
    std::pair<double, double> tiled_result = runMatrixMulTiledWithErrorCheck(M, N, K, tileSize);
    sleep(3);
    // Run naive matrix multiplication with error check
    std::cout << "\nRunning Naïve GPU Matrix Multiplication..." << std::endl;
    auto naive_result = runMatrixMulNaiveWithErrorCheck(M, N, K, 32, 8);

    
    

    // Print performance results
    std::cout << "\n==========================================" << std::endl;
    std::cout << "           PERFORMANCE RESULTS" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Naïve Kernel Execution Time: " << naive_result.second << " ms" << std::endl;
    std::cout << "Tiled Kernel Execution Time: " << tiled_result.second << " ms" << std::endl;

    std::cout << "\n==========================================" << std::endl;
    std::cout << "             SPEEDUP FACTOR" << std::endl;
    std::cout << "==========================================" << std::endl;
    double speedup = tiled_result.first / naive_result.first;
    std::cout << "Tiled Speedup Over Naïve: " << speedup << "x" << std::endl;

    return 0;
}