//
// Created by qaze on 01.11.2021.
//

#include "../include/gpu_computation_unit.cuh"
#include "consts.h"
#include "grid_operations.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime_api.h>

__device__ bool isBorder(const int i, const int sizeX, const int sizeY, const int totalSize)
{
    if (i < sizeX || i >= totalSize - sizeX) return true;
    if (i % sizeY == 0 || (i + 1) % sizeY == 0) return true;
    return false;
}

template<typename T>
__global__ void step(const int n, T *current, const T *previous,
                     const int sizeX, const int sizeY, const int totalSize,
                     const int wrap, const int start, const double epsilon) {
    int index = start + blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        if (!isBorder(i, sizeX, sizeY, totalSize)) {
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        } else {
            current[i] = previous[i];
        }
    }
}

template<typename T>
GpuComputationUnit<T>::GpuComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, int chunkStart, int chunkSize, bool leader)
        :ComputationUnit<T>(grid, previous, barrier, chunkStart, chunkSize, leader) {

    const int stepBlockCount = (chunkSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int i = 0; i < 1000; ++i) {
        if (leader) {
            grid.swapBuffers(previous);
        }

        cudaDeviceSynchronize();
        barrier.synchronise();

        step<<<stepBlockCount, BLOCK_SIZE>>>(chunkStart + chunkSize,
                                             grid.raw(), previous.raw(),
                                             grid.sizeX, grid.sizeY, grid.totalSize,
                                             grid.sizeY, chunkStart, EPSILON);

        cudaDeviceSynchronize();
        barrier.synchronise();
    }
}

template<typename T>
GpuComputationUnit<T>::~GpuComputationUnit() {

}

template class GpuComputationUnit<float>;
template class GpuComputationUnit<double>;
