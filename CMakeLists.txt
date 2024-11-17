cmake_minimum_required(VERSION 3.26)
project(DeepBit)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

find_package(OpenMP)

if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

if (NOT MSVC)
    set(DISKANN_ASYNC_LIB aio)
endif()

add_definitions(-std=c++17 -Ofast -march=core-avx2 -fopenmp)
add_subdirectory(src)