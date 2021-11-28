//
// Created by qaze on 01.11.2021.
//

#ifndef HEAT_GPU_FUNCTIONS_CUH
#define HEAT_GPU_FUNCTIONS_CUH

#include <grid.cuh>

template<typename T>
__global__ void copy(int n, T *source, T *destination);

template<typename T>
__global__ void step(int n, T *current, T *previous, int wrap, double epsilon);

#endif //HEAT_GPU_FUNCTIONS_CUH
