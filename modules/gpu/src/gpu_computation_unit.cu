//
// Created by qaze on 01.11.2021.
//

#include <util.h>
#include <iostream>
#include "../include/gpu_computation_unit.cuh"
#include <cuda_runtime_api.h>
#include <cuda_device_runtime_api.h>
#include <iomanip>

#define BLOCK_SIZE 256
#define EPSILON 0.01

__device__
bool d_finished;  // TODO perf

template<typename T>
GpuComputationUnit<T>::GpuComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, int chunkStart, int chunkSize, bool leader)
        :ComputationUnit<T>(grid, previous, barrier, chunkStart, chunkSize, leader) {

    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();
    bool h_finished = true;

    int copyBlockCount = (grid.totalSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int stepBlockCount = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "copy block count: " << copyBlockCount << std::endl;
    std::cout << "step block count: " << stepBlockCount << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device name: %s\n", prop.name);


    for (int i = 0; i < 1000; ++i) {
        if (leader) {
            copy<<<copyBlockCount, BLOCK_SIZE>>>(grid.totalSize, grid.raw(), previous.raw());
        }

        cudaDeviceSynchronize();
        barrier.synchronise();

        h_finished = true;
        cudaMemcpyToSymbol(d_finished, &h_finished, sizeof(bool));

        step<<<stepBlockCount, BLOCK_SIZE>>>(chunkStart + chunkSize, grid.raw(), previous.raw(), grid.sizeY, chunkStart, EPSILON);

        iterations++;
        if (iterations == iterations_print) {
            std::cout << "  " << std::setw(8) << iterations << "\n";
            iterations_print = 2 * iterations_print;
        }

        cudaMemcpyFromSymbol(&h_finished, d_finished, sizeof(bool));

        cudaDeviceSynchronize();
        barrier.synchronise();
    }

    std::cout << "total time " << timeMs() - startTime;
}

template<typename T>
__global__ void copy(int n, T *source, T *destination) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        destination[i] = source[i];
    }
}

template<typename T>
__global__ void step(int n, T *current, T *previous, int wrap, int start, double epsilon) { // TODO perf
    int index = start + blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        if (previous[i] != 0.0 && previous[i] != 100.0) { // TODO correctness, perf
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        }
        if (fabs(current[i] - previous[i]) > epsilon) d_finished = false; // TODO perf
    }
}

template<typename T>
GpuComputationUnit<T>::~GpuComputationUnit() {

}

template class GpuComputationUnit<float>;
template class GpuComputationUnit<double>;
