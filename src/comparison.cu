#include <iostream>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <cuda_runtime.h>
#include "matrix_multiply_tiled.cuh"
#include "matrix_multiply_naive.cuh"
#include "utils.cuh"

void printUsage(char *argv) {
    std::cout << "Usage: "<< argv << " -i <rowDimA> <colDimA> <colDimB> <gpu>" << std::endl;
    std::cout << "  <rowDimA>: Number of rows in matrix A and matrix C" << std::endl;
    std::cout << "  <colDimA>: Number of columns in matrix A (and number of rows in matrix B)" << std::endl;
    std::cout << "  <colDimB>: Number of columns in matrix B and matrix C" << std::endl;
    std::cout << "  <gpu>: Select a GPU (0-4)" << std::endl;
    exit(1);
}

int main(int argc, char *argv[]) {
    if (argc != 6 || std::string(argv[1]) != "-i") {
        printUsage(argv[0]);
    }

    int rowDimA = std::atoi(argv[2]);
    int colDimA = std::atoi(argv[3]);
    int colDimB = std::atoi(argv[4]);
    int gpuIndex = std::atoi(argv[5]);

    printGpuSpecs();
    gpuselect(gpuIndex);
    std::cout << "Matrix dimensions: " << std::endl;
    std::cout << "  A (" << rowDimA << "x" << colDimA << ")" << std::endl;
    std::cout << "  B (" << colDimA << "x" << colDimB << ")" << std::endl;
    std::cout << "  C (" << rowDimA << "x" << colDimB << ")" << std::endl;

    int tileSizes[] = {16,32,64};
    std::pair<int, int> blockConfigs[] = {{16, 16}, {32, 8}, {64, 4}};
    int block_height_tile = 0;
    for (int i = 0; i < 3; ++i) {
        int tileSize = tileSizes[i];
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
        block_height_tile = 256/tileSize;
        std::cout << "\nPerformance Results:" << std::endl;
        std::cout << "Tile Block Size: "<< tileSize << "x" << block_height_tile << " Tile Size: " << tileSize << std::endl;
        std::cout << "Naive Block Size: " << blockWidth << "x" << blockHeight << std::endl;
        std::cout << "Naive Execution Time (ms): " << avgNaiveTime << std::endl;
        std::cout << "Tiled Execution Time (ms): " << avgTiledTime << std::endl;
        std::cout << "Naive Performance (TFLOPS): " << avgNaiveFlops << std::endl;
        std::cout << "Tiled Performance (TFLOPS): " << avgTiledFlops << std::endl;
        
    }
    
    return 0;
}
