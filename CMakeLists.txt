cmake_minimum_required(VERSION 3.20)

project(matmul LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(CLI11 CONFIG REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets --expt-relaxed-constexpr --expt-extended-lambda")

set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_DEBUG "-G")

add_executable(matmul matmul.cu)
target_link_libraries(matmul PRIVATE Eigen3::Eigen CLI11::CLI11 cuda)