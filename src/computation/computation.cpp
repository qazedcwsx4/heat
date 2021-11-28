//
// Created by qaze on 18.09.2021.
//

#include <gpu_computation_unit.cuh>
#include "computation.h"

template<typename T>
Computation<T>::Computation(Grid<T> &grid, std::vector<ComputationUnit<T>> computationUnits):
        computationUnits(computationUnits){}

template<typename T>
Computation<T> Computation<T>::newCpuComputation(Grid<T> &grid) {
    Grid<T> previous = Grid<T>::newCpu(grid.sizeX, grid.sizeY);

    int threads = 8;
    Synchronisation barrier(threads);

    return Computation<T>(grid, {CpuComputationUnit<T>(grid, previous, barrier, 0, 0)});
}

template<typename T>
Computation<T> Computation<T>::newGpuComputation(Grid<T> &grid) {
    Grid<T> previous = Grid<T>::newManaged(grid.sizeX, grid.sizeY);

    int threads = 1;
    Synchronisation barrier(threads);

    return Computation<T>(grid, {GpuComputationUnit<T>(grid, previous, barrier, 0, 0)});
}

template class Computation<float>;
template class Computation<double>;
