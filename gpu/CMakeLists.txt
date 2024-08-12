cmake_minimum_required(VERSION 3.28)

set(CMAKE_C_COMPILER "/usr/bin/gcc-13")
set(CMAKE_CXX_COMPILER "/usr/bin/g++-13")
set(CMAKE_CXX_FLAGS "-fopenmp")
set(CMAKE_C_FLAGS "-fopenmp")
set(CMAKE_CUDA_FLAGS "-fopenmp")
enable_language(CXX)
find_package(OpenMP REQUIRED)


set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES 75)


project(vgg16_cuda CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(vgg16_cuda adibof.cu)
add_executable(vgg16_cuda_origin vgg_origin.cu)
add_executable(test_code test_code.cu)
add_executable(vgg16_cuda_custom vgg_custom.cu)


set_target_properties(vgg16_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vgg16_cuda_custom PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(vgg16_cuda_origin PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test_code PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(vgg16_cuda PRIVATE cudnn culibos cublas)
target_link_libraries(vgg16_cuda_origin PRIVATE cudnn culibos cublas)
target_link_libraries(vgg16_cuda_custom PRIVATE cudnn culibos cublas)
target_link_libraries(test_code PRIVATE cudnn culibos cublas)
