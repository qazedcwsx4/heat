//
// Created by qaze on 01.11.2021.
//

#include "../include/gpu_computation_unit.cuh"
#include "consts.h"
#include "grid_operations.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

template<typename T>
__global__ void step(const int n, T **current, T **previous,
                     const int sizeY,
                     const int wrap, const int start) {
    const long long index = start + blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = blockDim.x * gridDim.x;

    for (int repeat = 0; repeat < GPU_POWERCAP; repeat++) {
        for (long long i = index; i < n; i += stride) {
            *current[i] = (*(previous[i] - 1) + *(previous[i] + 1) + *(previous[i] - sizeY) + *(previous[i] + sizeY)) / 4.0;
        }
    }
}

template<typename T>
GpuComputationUnit<T>::GpuComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, const long long chunkStart, const long long chunkSize, bool leader)
        :ComputationUnit<T>(grid, previous, barrier, chunkStart, chunkSize, leader) {

    const long long stepBlockCount = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < ITERATION_COUNT; ++i) {
        if (leader) {
            grid.swapBuffers(previous);
        }

        cudaDeviceSynchronize();
        barrier.synchronise();

        step<<<stepBlockCount, BLOCK_SIZE>>>(chunkStart + chunkSize,
                                             grid.borderlessRaw(), previous.borderlessRaw(),
                                             grid.sizeY,
                                             grid.sizeY, chunkStart);

        cudaDeviceSynchronize();
        barrier.synchronise();
    }
}

template<typename T>
GpuComputationUnit<T>::~GpuComputationUnit() {

}

template class GpuComputationUnit<float>;
template class GpuComputationUnit<double>;
