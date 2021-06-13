//
// Created by qaze on 16.05.2021.
//

#ifndef HEAT_GRID_OPERATIONS_H
#define HEAT_GRID_OPERATIONS_H

#include "grid.cuh"

template <typename T>
void saveToFile(Grid<T> &grid, const char *filename);

template <typename T>
void setupTest(Grid<T> &grid);

#endif //HEAT_GRID_OPERATIONS_H
