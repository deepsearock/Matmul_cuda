# cs811_lab1
cs811_lab1
Matrix multiplication with arbitrary size in cuda

In order to build you must run

mkdir build && cd build
cmake ..
cmake --build .

This will yield 4 programs
./comparison
./error
./cublas
./cpu

./comparison utilization:
Usage: ./comparison -i <rowDimA> <colDimA> <colDimB> <gpu> this runs the main program which compares tiled vs naive at different hardcoded block sizes and tile sizes, you also have to select a gpu. use nvidia-smi to find your gpu
  <rowDimA>: Number of rows in matrix A and matrix C
  <colDimA>: Number of columns in matrix A (and number of rows in matrix B)
  <colDimB>: Number of columns in matrix B and matrix C
  <gpu>: Select a GPU (0-4)

./error utilization: this generates random matrices for A and B and computes error rate between naive and tiled measured against cpu. tile size can be selected for tiled algorithm by giving it a tilesize, you have to anyway
    Usage: ./error -i <tileSize>

./cublas utilization: this program runs cublas based on input matrix sizes. its just a wrapper 
    Usage: ./cublas -i <rowDimA> <colDimA> <colDimB>
    its best to use cublas program with nvprof to see how fast the actual matmul algorithm is because i couldnt write a wrapper for it to measure perf. 
    nvprof ./cublas -i <rowDimA> <colDimA> <colDimB>

./cpu utilization: computes cpu speed, note this doesnt require -i to begin make sure you dont use -i. also this program doesn't have much of a use other than if you want to see single thread perf
    Usage: ./cpu <rowDimA> <colDimA> <colDimB>
