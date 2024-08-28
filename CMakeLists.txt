cmake_minimum_required(VERSION 3.26)
project(DeepBit)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_definitions(-std=c++17 -Ofast -march=core-avx2)
add_subdirectory(src)