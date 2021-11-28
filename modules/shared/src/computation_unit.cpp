//
// Created by qaze on 19.09.2021.
//

#include "../include/computation_unit.h"

template<typename T>
ComputationUnit<T>::ComputationUnit(Grid<T> &grid, Grid<T> &previous,
                                    Synchronisation barrier,
                                    int chunkStart, int chunkSize):
        grid(grid), chunkStart(chunkStart),
        barrier(barrier),
        chunkSize(chunkSize), previous(previous) {}

template
class ComputationUnit<float>;

template
class ComputationUnit<double>;
