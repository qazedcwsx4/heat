//
// Created by qaze on 16.05.2021.
//

#ifndef HEAT_GRID_OPERATIONS_H
#define HEAT_GRID_OPERATIONS_H

#include "grid.cuh"

void saveToFile(Grid &grid, const char *filename);

void setupTest(Grid &grid);

#endif //HEAT_GRID_OPERATIONS_H
