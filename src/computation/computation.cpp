//
// Created by qaze on 18.09.2021.
//

#include <gpu_computation_unit.cuh>
#include <iostream>
#include "computation.h"

template<typename T>
Computation<T>::Computation(Grid<T> &grid, std::vector<ComputationUnit<T>> computationUnits):
        computationUnits(computationUnits){}

template<typename T>
Computation<T> Computation<T>::newCpuComputation(Grid<T> &grid) {
    Grid<T> previous = Grid<T>::newCpu(grid.sizeX, grid.sizeY);

    int threads = THREAD_COUNT;
    Synchronisation barrier(threads);

    auto cu1 = CpuComputationUnit<T>(grid, previous, barrier, 0, grid.totalSize, true);

    cu1.await();

    return Computation<T>(grid, {cu1});
}

template<typename T>
Computation<T> Computation<T>::newGpuComputation(Grid<T> &grid) {
    Grid<T> previous = Grid<T>::newManaged(grid.sizeX, grid.sizeY);

    int threads = 1;
    Synchronisation barrier(threads);

    return Computation<T>(grid, {GpuComputationUnit<T>(grid, previous, barrier, 0, grid.totalSize, true)});
}

template<typename T>
Computation<T> Computation<T>::newHybridComputation(Grid<T> &grid) {
    Grid<T> previous = Grid<T>::newManaged(grid.sizeX, grid.sizeY);

    int threads = THREAD_COUNT + 1;
    Synchronisation barrier(threads);

    const int cpuChunkStart = 0;
    const int cpuChunkSize = grid.totalSize / 2;

    const int gpuChunkStart = cpuChunkStart + cpuChunkSize;
    const int gpuChunkSize = grid.totalSize - cpuChunkSize;

    auto cu1 = CpuComputationUnit<T>(grid, previous, barrier, cpuChunkStart, cpuChunkSize, false);
    auto cu2 = GpuComputationUnit<T>(grid, previous, barrier, gpuChunkStart, gpuChunkSize, true);
    cu1.await();

    return Computation<T>(grid, {cu1, cu2});
}

template class Computation<float>;
template class Computation<double>;
