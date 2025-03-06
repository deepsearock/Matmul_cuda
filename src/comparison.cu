#include <iostream>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <cuda_runtime.h>
#include "matrix_multiply_tiled.cuh"
#include "matrix_multiply_naive.cuh"
#include "utils.cuh"

void printUsage() {
    std::cout << "Usage: TiledMatrixMul -i <rowDimA> <colDimA> <colDimB>" << std::endl;
    std::cout << "  <rowDimA>: Number of rows in matrix A and matrix C" << std::endl;
    std::cout << "  <colDimA>: Number of columns in matrix A (and number of rows in matrix B)" << std::endl;
    std::cout << "  <colDimB>: Number of columns in matrix B and matrix C" << std::endl;
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc != 5 || std::string(argv[1]) != "-i") {
        printUsage();
    }

    int rowDimA = std::atoi(argv[2]);
    int colDimA = std::atoi(argv[3]);
    int colDimB = std::atoi(argv[4]);

    printGpuSpecs();
    gpuselect(3);
    std::cout << "Matrix dimensions: " << std::endl;
    std::cout << "  A (" << rowDimA << "x" << colDimA << ")" << std::endl;
    std::cout << "  B (" << colDimA << "x" << colDimB << ")" << std::endl;
    std::cout << "  C (" << rowDimA << "x" << colDimB << ")" << std::endl;

    int tileSizes[] = {16, 32, 64};
    std::pair<int, int> blockConfigs[] = {{32, 8}, {16, 16}};

    for (int tileSize : tileSizes) {
        for (size_t i = 0; i < 2; i++) { // Fix structured bindings issue
            int blockWidth = blockConfigs[i].first;
            int blockHeight = blockConfigs[i].second;

            double totalNaiveTime = 0.0, totalNaiveFlops = 0.0;
            double totalTiledTime = 0.0, totalTiledFlops = 0.0;

            for (int run = 0; run < 2; ++run) {
                auto naiveResult = runMatrixMulNaive(rowDimA, colDimB, colDimA, blockWidth, blockHeight);
                totalNaiveTime += naiveResult.second;
                totalNaiveFlops += naiveResult.first;

                std::pair<double, double> tiledResult = runMatrixMulTiled(rowDimA, colDimB, colDimA, tileSize);
                totalTiledTime += tiledResult.second;
                totalTiledFlops += tiledResult.first;
            }

            double avgNaiveTime = totalNaiveTime / 2;
            double avgNaiveFlops = totalNaiveFlops / 2;
            double avgTiledTime = totalTiledTime / 2;
            double avgTiledFlops = totalTiledFlops / 2;

            std::cout << "\nPerformance Results:" << std::endl;
            std::cout << "Tile Block Size: tileSize^2 Tile Size: " << tileSize << std::endl;
            std::cout << "Naive Block Size: " << blockWidth << "x" << blockHeight << std::endl;
            std::cout << "Naive Execution Time (ms): " << avgNaiveTime << std::endl;
            std::cout << "Tiled Execution Time (ms): " << avgTiledTime << std::endl;
            std::cout << "Naive Performance (TFLOPS): " << avgNaiveFlops << std::endl;
            std::cout << "Tiled Performance (TFLOPS): " << avgTiledFlops << std::endl;
        }
    }
    
    return 0;
}
