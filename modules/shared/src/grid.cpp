//
// Created by qaze on 12.05.2021.
//

#include "../include/grid.cuh"

Grid::Grid(int sizeX, int sizeY) :
        sizeX{sizeX},
        sizeY{sizeY},
        field{new double[sizeX * sizeY]} {
}

Grid::Grid(int sizeX, int sizeY, double *field) :
        sizeX{sizeX},
        sizeY{sizeY},
        field{field} {
}

Grid::GridInner Grid::operator[](int x) {
    return {field, x, sizeY};
}
