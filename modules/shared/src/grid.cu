//
// Created by qaze on 12.05.2021.
//

#include <cuda_runtime_api.h>
#include "../include/grid.cuh"

Grid::Grid(bool isManaged, int sizeX, int sizeY) :
        isManaged{isManaged},
        sizeX{sizeX},
        sizeY{sizeY} {
    if (isManaged) {
        cudaMallocManaged((void **) (&field), sizeX * sizeY * sizeof(double));
    } else {
        field = new double[sizeX * sizeY];
    }
}

Grid::GridInner Grid::operator[](int x) {
    return {field, x, sizeY};
}

Grid::~Grid() {
    if (isManaged) {
        cudaFree(field);
    } else {
        delete field;
    }
}

Grid Grid::newCpu(int sizeX, int sizeY) {
    return {false, sizeX, sizeY};
}

Grid Grid::newManaged(int sizeX, int sizeY) {
    return {true, sizeX, sizeY};
}
