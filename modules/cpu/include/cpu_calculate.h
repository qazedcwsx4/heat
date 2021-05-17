//
// Created by qaze on 12.05.2021.
//

#ifndef HEAT_CPU_CALCULATE_H
#define HEAT_CPU_CALCULATE_H

#include <iostream>
#include "grid.cuh"

void cpuCalculate(Grid &grid);

void cpuCalculateParallel(Grid &grid);

#endif //HEAT_CPU_CALCULATE_H
