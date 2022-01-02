//
// Created by qaze on 18.09.2021.
//

#ifndef HEAT_COMPUTATION_H
#define HEAT_COMPUTATION_H


#include <vector>
#include <computation_unit.h>
#include <cpu_computation_unit.h>

template <typename T>
class Computation {
private:
    std::vector<ComputationUnit<T>> computationUnits;

    Computation(Grid<T> &grid, std::vector<ComputationUnit<T>> computationUnits);

public:
    static Computation<T> newCpuComputation(Grid<T>& grid);

    static Computation<T> newGpuComputation(Grid<T>& grid);

    static Computation<T> newHybridComputation(Grid<T>& grid);
};

#endif //HEAT_COMPUTATION_H
