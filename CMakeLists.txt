cmake_minimum_required(VERSION 3.17)
project(heat)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(heat main.cpp Grid.cpp heat/heat.cpp output/grid_to_bmp.cpp output/grid_to_bmp.h heat/heat.h)