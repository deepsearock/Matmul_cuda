cmake_minimum_required(VERSION 3.18)
project(CUDA_Executables LANGUAGES CUDA CXX)

# Set CUDA architecture (modify based on your GPU, or use all supported architectures)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find and link CUDA libraries if necessary
find_package(CUDAToolkit REQUIRED)

# Create an executable for comparison.cu
add_executable(comparison comparison.cu)
set_target_properties(comparison PROPERTIES 
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(comparison PRIVATE CUDA::cudart)

# Create an executable for cpu.cu
add_executable(cpu cpu.cu)
set_target_properties(cpu PROPERTIES 
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(cpu PRIVATE CUDA::cudart)

# Create an executable for cublas.cu (requires cuBLAS library)
add_executable(cublas cublas.cu)
set_target_properties(cublas PROPERTIES 
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(cublas PRIVATE CUDA::cudart CUDA::cublas)
