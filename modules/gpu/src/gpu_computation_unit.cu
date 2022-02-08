//
// Created by qaze on 01.11.2021.
//

#include <iostream>
#include "../include/gpu_computation_unit.cuh"
#include "consts.h"
#include "grid_operations.h"
#include <cuda_runtime_api.h>

__device__
bool d_finished;  // TODO perf

template<typename T>
__global__ void copy(int n, T *source, T *destination) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        destination[i] = source[i];
    }
}

template<typename T>
__global__ void step(int n, Grid<T> *current_grid, Grid<T> *previous_grid, int wrap, int start, double epsilon) { // TODO perf
    int index = start + blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    auto current = current_grid->raw();
    auto previous = previous_grid->raw();

    for (int i = index; i < n; i += stride) {
        if (previous[i] != 0.0 && previous[i] != 100.0) { // TODO correctness, perf
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        }
        if (fabs(current[i] - previous[i]) > epsilon) d_finished = false; // TODO perf
    }
}

template<typename T>
GpuComputationUnit<T>::GpuComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, int chunkStart, int chunkSize, bool leader)
        :ComputationUnit<T>(grid, previous, barrier, chunkStart, chunkSize, leader) {

    bool h_finished = true;

    int copyBlockCount = (grid.totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int stepBlockCount = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "copy block count: " << copyBlockCount << std::endl;
    std::cout << "step block count: " << stepBlockCount << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    std::cout << "Device name: " << prop.name << std::endl;


    for (int i = 0; i < 1000; ++i) {
        if (leader) {
            // czemu to nie dziala??
            //grid.swapBuffers(previous);
            copy<<<copyBlockCount, BLOCK_SIZE>>>(grid.totalSize, grid.raw(), previous.raw());
        }

        cudaDeviceSynchronize();
        barrier.synchronise();

        h_finished = true;
        cudaMemcpyToSymbol(d_finished, &h_finished, sizeof(bool));

        step<<<stepBlockCount, BLOCK_SIZE>>>(chunkStart + chunkSize, &grid, &previous, grid.sizeY, chunkStart, EPSILON);

        cudaMemcpyFromSymbol(&h_finished, d_finished, sizeof(bool));

        cudaDeviceSynchronize();
        barrier.synchronise();
    }
}

template<typename T>
GpuComputationUnit<T>::~GpuComputationUnit() {

}

template class GpuComputationUnit<float>;
template class GpuComputationUnit<double>;
