cmake_minimum_required(VERSION 3.18)
project(CUDA_Executables LANGUAGES CUDA CXX)

# Set CUDA architecture (modify based on your GPU, or use all supported architectures)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find and link CUDA libraries if necessary
find_package(CUDAToolkit REQUIRED)

#find src
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/src)

#comparsion.cu
add_executable(comparison comparison.cu)
set_target_properties(comparison PROPERTIES 
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(comparison PRIVATE CUDA::cudart)

#cpu.cu
add_executable(cpu cpu.cu)
set_target_properties(cpu PROPERTIES 
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(cpu PRIVATE CUDA::cudart)

#cublas
add_executable(cublas cublas.cu)
set_target_properties(cublas PROPERTIES 
    CUDA_STANDARD 14
    CUDA_STANDARD_REQUIRED ON
)
target_link_libraries(cublas PRIVATE CUDA::cudart CUDA::cublas)
