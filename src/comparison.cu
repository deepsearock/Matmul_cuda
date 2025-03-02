#include <iostream>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <cuda_runtime.h>
#include "matrix_multiply_tiled.cuh"
#include "matrix_multiply_naive.cuh"

void printUsage() {
    std::cout << "Usage: TiledMatrixMul -i <rowDimA> <colDimA> <colDimB>" << std::endl;
    std::cout << "  <rowDimA>: Number of rows in matrix A and matrix C" << std::endl;
    std::cout << "  <colDimA>: Number of columns in matrix A (and number of rows in matrix B)" << std::endl;
    std::cout << "  <colDimB>: Number of columns in matrix B and matrix C" << std::endl;
    exit(1);
}

void printGpuSpecs() {
    cudaDeviceProp mygpu;
    cudaGetDeviceProperties(&mygpu, 0);

    std::cout << "GPU Specifications:" << std::endl;
    std::cout << "  Name: " << mygpu.name << std::endl;
    std::cout << "  CUDA Cores per SM: " << mygpu.multiProcessorCount << std::endl;
    std::cout << "  Number of SMs: " << mygpu.multiProcessorCount << std::endl;
    std::cout << "  GPU Clock Rate (MHz): " << mygpu.clockRate / 1000.0 << std::endl;
    std::cout << "  GPU Memory Clock Rate (Mhz): " <<mygpu.memoryClockRate / 1000.0 << std::endl;
    std::cout << "  Memory Bus Width: " << mygpu.memoryBusWidth << std::endl;
    std::cout << "  Memory Bandwidth (GB/s): " << mygpu.memoryBusWidth * mygpu.memoryClockRate * 2 / 1.0e6 << std::endl;
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

    double numOps = 2.0 * rowDimA * colDimB * colDimA;
    double memoryVolumeBytes = (rowDimA * colDimA + colDimA * colDimB + rowDimA * colDimB) * sizeof(float);
    double memoryVolumeGB = memoryVolumeBytes / 1e9;

    for (int blockSize : blockSizes) {
        for (int tileSize : tileSizes) {
            //auto naiveResult = runMatrixMulNaive(rowDimA, colDimB, colDimA, blockSize);
            //double totalNaiveTime = naiveResult.second;
            //double totalNaiveFlops = naiveResult.first;
            //double naiveMemoryBandwidth = memoryVolumeGB / (totalNaiveTime / 1000.0);

            auto tiledResult = runMatrixMulTiled(rowDimA, colDimB, colDimA, tileSize);
            double totalTiledTime = tiledResult.second;
            double totalTiledFlops = tiledResult.first;
            double tiledMemoryBandwidth = memoryVolumeGB / (totalTiledTime / 1000.0);

            std::cout << "\nPerformance Results:" << std::endl;
            std::cout << "Block Size: " << blockSize << ", Tile Size: " << tileSize << std::endl;
            //std::cout << "Naive Execution Time (ms): " << totalNaiveTime << std::endl;
            std::cout << "Tiled Execution Time (ms): " << totalTiledTime << std::endl;
            //std::cout << "Naive Performance (TFLOPS): " << totalNaiveFlops << std::endl;
            std::cout << "Tiled Performance (TFLOPS): " << totalTiledFlops << std::endl;
            //std::cout << "Naive Memory Bandwidth (GB/s): " << naiveMemoryBandwidth << std::endl;
            std::cout << "Tiled Memory Bandwidth (GB/s): " << tiledMemoryBandwidth << std::endl;
        }
    }
    return 0;
}
