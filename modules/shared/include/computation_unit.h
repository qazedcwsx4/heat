//
// Created by qaze on 19.09.2021.
//

#ifndef HEAT_COMPUTATION_UNIT_H
#define HEAT_COMPUTATION_UNIT_H

#include "grid.cuh"
#include <synchronisation.h>

template<typename T>
class ComputationUnit {
private:

protected:
    Grid<T> &grid;
    Grid<T> &previous;
    int chunkStart;
    int chunkSize;
    Synchronisation barrier;

    ComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, int chunkStart, int chunkSize);

public:

};

#endif //HEAT_COMPUTATION_UNIT_H
