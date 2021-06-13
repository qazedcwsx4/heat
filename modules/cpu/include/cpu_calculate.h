//
// Created by qaze on 12.05.2021.
//

#ifndef HEAT_CPU_CALCULATE_H
#define HEAT_CPU_CALCULATE_H

#include <iostream>
#include "grid.cuh"

#define EPSILON 0.01
#define THREAD_COUNT 8

template <typename T>
void cpuCalculate(Grid<T> &grid);

template <typename T>
void cpuCalculateParallel(Grid<T> &grid);

template <typename T>
void cpuCalculateParallelProper(Grid<T> &grid);

#endif //HEAT_CPU_CALCULATE_H
