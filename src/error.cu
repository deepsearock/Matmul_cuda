#include <iostream>
#include <cstdlib>
#include <ctime>
#include "utils.cuh"
#include "matrix_multiply_naive.cuh"
#include "matrix_multiply_tiled.cuh"
#include <unistd.h>
#include <random>
int main(int argc, char *argv[]) {
    // set matrix dimensions
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    std::uniform_int_distribution<int> distribution(1, 2000);
    int M = distribution(generator); //number of rows in A and C
    int N = distribution(generator); //number of columns in B and C
    int K = distribution(generator); //number of columns in A and rows in B
    
    if (argc != 3 || std::string(argv[1]) != "-i") {
        std::cerr << "Usage: " << argv[0] << " -i <tileSize>" << std::endl;
        return 1;
    }

    int tileSize = std::atoi(argv[2]);  //tile size for tiled matrix multiplication


    printGpuSpecs();
    gpuselect(0);

    std::cout << "==========================================" << std::endl;
    std::cout << "                ERROR CHECK" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Matrix Dimensions: " << M << " x " << N << " x " << K << std::endl;
    //run tiled matrix multiplication with error check
    std::cout << "\nRunning Tiled GPU Matrix Multiplication (Tile Size = " << tileSize << ")...\n" << std::endl;
    std::pair<double, double> tiled_result = runMatrixMulTiledWithErrorCheck(M, N, K, tileSize);
    //run naive matrix multiplication with error check
    std::cout << "\nRunning Naïve GPU Matrix Multiplication..." << std::endl;
    auto naive_result = runMatrixMulNaiveWithErrorCheck(M, N, K, 32, 8);

    
    

    //print performance results
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