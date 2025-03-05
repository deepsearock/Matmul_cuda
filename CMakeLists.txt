cmake_minimum_required(VERSION 3.18)
project(CUDA_Project LANGUAGES CUDA CXX)

# Set CUDA architecture (modify based on your GPU, or use all supported architectures)
set(CMAKE_CUDA_ARCHITECTURES 70)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -Xptxas -dlcm=ca --use_fast_math -lineinfo")

# Enable CUDA compilation with NVCC
enable_language(CUDA)
 
# Find CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# Include directories (modify if headers are elsewhere)

set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)

# Include directories
include_directories(${INCLUDE_DIR})

# Define source files
set(SOURCES
    ${SRC_DIR}/comparison.cu
    ${SRC_DIR}/cpu.cu
    ${SRC_DIR}/cublas.cu
    ${SRC_DIR}/error.cu
)

# Create separate executables for each CUDA file
add_executable(comparison ./src/comparison.cu)
set_target_properties(comparison PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(comparison PRIVATE CUDA::cudart)

add_executable(cpu ./src/cpu.cu)
set_target_properties(cpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cpu PRIVATE CUDA::cudart)

add_executable(cublas ./src/cublas.cu)
set_target_properties(cublas PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(cublas PRIVATE CUDA::cudart CUDA::cublas)

add_executable(error ./src/error.cu)
set_target_properties(error PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(error PRIVATE CUDA::cudart)