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

    std::cout << "Matrix dimensions: " << std::endl;
    std::cout << "  A (" << rowDimA << "x" << colDimA << ")" << std::endl;
    std::cout << "  B (" << colDimA << "x" << colDimB << ")" << std::endl;
    std::cout << "  C (" << rowDimA << "x" << colDimB << ")" << std::endl;

    int blockSizes[] = {8, 16, 32};
    int tileSizes[] = {8, 16, 32};

    double memoryVolumeBytes = (rowDimA * colDimA + colDimA * colDimB + rowDimA * colDimB) * sizeof(float);
    double memoryVolumeGB = memoryVolumeBytes / 1e9;

    for (int blockSize : blockSizes) {
        for (int tileSize : tileSizes) {
            double totalNaiveTime = 0.0, totalNaiveFlops = 0.0;
            double totalTiledTime = 0.0, totalTiledFlops = 0.0;

            for (int run = 0; run < 10; ++run) {
                auto naiveResult = runMatrixMulNaive(rowDimA, colDimB, colDimA, blockSize);
                totalNaiveTime += naiveResult.second;
                totalNaiveFlops += naiveResult.first;
                //double naiveMemoryBandwidth = memoryVolumeGB / (totalNaiveTime / 1000.0);

                auto tiledResult = runMatrixMulTiled(rowDimA, colDimB, colDimA, tileSize);
                totalTiledTime += tiledResult.second;
                totalTiledFlops += tiledResult.first;
                //double tiledMemoryBandwidth = memoryVolumeGB / (totalTiledTime / 1000.0);
            }

            double avgNaiveTime = totalNaiveTime / 10.0;
            double avgNaiveFlops = totalNaiveFlops / 10.0;
            double avgTiledTime = totalTiledTime / 10.0;
            double avgTiledFlops = totalTiledFlops / 10.0;

            std::cout << "\nPerformance Results:" << std::endl;
            std::cout << "Block Size: " << blockSize << ", Tile Size: " << tileSize, ", Block Size: 8x" << tileSize << std::endl;
            std::cout << "Naive Execution Time (ms): " << avgNaiveTime << std::endl;
            std::cout << "Tiled Execution Time (ms): " << avgTiledTime << std::endl;
            std::cout << "Naive Performance (TFLOPS): " << avgNaiveFlops << std::endl;
            std::cout << "Tiled Performance (TFLOPS): " << avgTiledFlops << std::endl;
            //std::cout << "Naive Memory Bandwidth (GB/s): " << naiveMemoryBandwidth << std::endl;
            //std::cout << "Tiled Memory Bandwidth (GB/s): " << tiledMemoryBandwidth << std::endl;
        }
    }
    return 0;
}
