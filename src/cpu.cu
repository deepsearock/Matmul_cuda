#include <iostream>
#include <cstdlib>
#include "matrix_multiply_cpu.cuh"

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <rowDimA> <colDimA> <colDimB>" << std::endl;
        return 1;
    }

    int M = std::atoi(argv[1]);
    int N = std::atoi(argv[2]);
    int K = std::atoi(argv[3]);

    // Store the result in a pair
    std::pair<double, double> result = runMatrixMulNaive(M, N, K);
    
    double time_sec = result.first;
    double tflops = result.second;

    std::cout << "Matrix multiplication took " << time_sec << " seconds." << std::endl;
    std::cout << "Performance: " << tflops << " TFLOPS." << std::endl;
    
    return 0;
}
