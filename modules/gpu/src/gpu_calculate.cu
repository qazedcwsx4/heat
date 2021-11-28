//
// Created by qaze on 12.05.2021.
//

#include "../include/gpu_calculate.cuh"
#include <iostream>
#include <cuda_device_runtime_api.h>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <util.h>

#define BLOCK_SIZE 256
#define EPSILON 0.0001

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wint-conversion"

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"

template<typename T>
__global__ void copy(int n, T *source, T *destination) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        destination[i] = source[i];
    }
}

#pragma clang diagnostic pop

__device__
bool d_finished;  // TODO perf

#pragma clang diagnostic push
#pragma ide diagnostic ignored "CannotResolve"

template<typename T>
__global__ void step(int n, T *current, T *previous, int wrap, double epsilon) { // TODO perf
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        if (previous[i] != 0.0 && previous[i] != 100.0) { // TODO correctness, perf
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        }
        if (fabs(current[i] - previous[i]) > epsilon) d_finished = false; // TODO perf
    }
}

#pragma clang diagnostic pop

template<typename T>
void gpuCalculate(Grid<T> &grid) {
    const int sizeX = grid.sizeX;
    const int sizeY = grid.sizeY;

    Grid<T> previous = Grid<T>::newManaged(sizeX, sizeY);

    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();
    bool h_finished = true;

    int blockCount = (sizeX * sizeY + BLOCK_SIZE - 1) / BLOCK_SIZE;
    std::cout << "block count: " << blockCount << std::endl;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Device name: %s\n", prop.name);


    do {
        copy<<<blockCount, BLOCK_SIZE>>>(sizeX * sizeY, grid.raw(), previous.raw());
        cudaDeviceSynchronize();
        h_finished = true;
        cudaMemcpyToSymbol(d_finished, &h_finished, sizeof(bool));

        step<<<blockCount, BLOCK_SIZE>>>(sizeX * sizeY, grid.raw(), previous.raw(), sizeY, EPSILON);

        iterations++;
        if (iterations == iterations_print) {
            std::cout << "  " << std::setw(8) << iterations << "\n";
            iterations_print = 2 * iterations_print;
        }

        cudaMemcpyFromSymbol(&h_finished, d_finished, sizeof(bool));
        cudaDeviceSynchronize();
    } while (!h_finished);

    std::cout << "total time " << timeMs() - startTime;
}

template void gpuCalculate<float>(Grid<float> &grid);
template void gpuCalculate<double>(Grid<double> &grid);

#pragma clang diagnostic pop