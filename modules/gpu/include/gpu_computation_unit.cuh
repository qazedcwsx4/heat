//
// Created by qaze on 01.11.2021.
//

#ifndef HEAT_GPU_COMPUTATION_UNIT_H
#define HEAT_GPU_COMPUTATION_UNIT_H

#include <computation_unit.h>
#include <barrier>

template<typename T>
class GpuComputationUnit : public ComputationUnit<T> {
private:
    bool finished = false;

public:
    GpuComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, int chunkStart, int chunkSize);

    ~GpuComputationUnit();
};


#endif //HEAT_GPU_COMPUTATION_UNIT_H
