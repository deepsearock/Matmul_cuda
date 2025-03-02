#include <iostream>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include "matrix_multiply_tiled.cuh"
#include "matrix_multiply_naive.cuh"

void printUsage() {
    std::cout << "Usage: TiledMatrixMul -i <rowDimA> <colDimA> <colDimB>" << std::endl;
    std::cout << "  <rowDimA>: Number of rows in matrix A and matrix C" << std::endl;
    std::cout << "  <colDimA>: Number of columns in matrix A (and number of rows in matrix B)" << std::endl;
    std::cout << "  <colDimB>: Number of columns in matrix B and matrix C" << std::endl;
    exit(1);
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc != 5 || std::string(argv[1]) != "-i") {
        printUsage();
    }

    int rowDimA = std::atoi(argv[2]);
    int colDimA = std::atoi(argv[3]);
    int colDimB = std::atoi(argv[4]);

    // Print matrix dimensions
    std::cout << "Matrix dimensions: " << std::endl;
    std::cout << "  A (" << rowDimA << "x" << colDimA << ")" << std::endl;
    std::cout << "  B (" << colDimA << "x" << colDimB << ")" << std::endl;
    std::cout << "  C (" << rowDimA << "x" << colDimB << ")" << std::endl;

    // List of block sizes and tile sizes to compare
    int blockSizes[] = {8, 16, 32};
    int tileSizes[] = {8, 16, 32};

    // Compare performance for different block and tile sizes
    for (int blockSize : blockSizes) {
        for (int tileSize : tileSizes) {
            double totalNaiveTime = 0.0;
            double totalNaiveFlops = 0.0;
            double totalTiledTime = 0.0;
            double totalTiledFlops = 0.0;

            // Run the test 10 times
            for (int i = 0; i < 1; ++i) {
                std::cout << "\nRunning Naive Matrix Multiplication with Block Size: " << blockSize << " and Tile Size: " << tileSize << " (Run " << i + 1 << ")" << std::endl;
                auto naiveResult = runMatrixMulNaive(rowDimA, colDimB, colDimA, blockSize);
                totalNaiveTime += naiveResult.second;
                totalNaiveFlops += naiveResult.first;

                std::cout << "\nRunning Tiled Matrix Multiplication with Block Size: " << blockSize << " and Tile Size: " << tileSize << " (Run " << i + 1 << ")" << std::endl;
                auto tiledResult = runMatrixMulTiled(rowDimA, colDimB, colDimA, tileSize);
                totalTiledTime += tiledResult.second;
                totalTiledFlops += tiledResult.first;
            }

            // Calculate averages
            double avgNaiveTime = totalNaiveTime / 10.0;
            double avgNaiveFlops = totalNaiveFlops / 10.0;
            double avgTiledTime = totalTiledTime / 10.0;
            double avgTiledFlops = totalTiledFlops / 10.0;

            // Output average results
            std::cout << "\nAverage Performance over 10 Runs:" << std::endl;
            std::cout << "Naive Execution Time (ms): " << avgNaiveTime << std::endl;
            std::cout << "Tiled Execution Time (ms): " << avgTiledTime << std::endl;
            std::cout << "Naive Performance (TFLOPS): " << avgNaiveFlops << std::endl;
            std::cout << "Tiled Performance (TFLOPS): " << avgTiledFlops << std::endl;
        }
    }

    return 0;
}
