//
// Created by qaze on 12.05.2021.
//

#include <cuda_runtime_api.h>
#include "../include/grid.cuh"

template <typename T>
Grid<T>::Grid(bool isManaged, int sizeX, int sizeY) :
        isManaged{isManaged},
        sizeX{sizeX},
        sizeY{sizeY} {
    if (isManaged) {
        cudaMallocManaged((void **) (&field), sizeX * sizeY * sizeof(T));
    } else {
        field = new T[sizeX * sizeY];
    }
}

template <typename T>
Grid<T>::GridInner<T> Grid<T>::operator[](int x) {
    return {field, x, sizeY};
}

template <typename T>
Grid<T>::~Grid() {
    if (isManaged) {
        cudaFree(field);
    } else {
        delete field;
    }
}

template <typename T>
Grid<T> Grid<T>::newCpu(int sizeX, int sizeY) {
    return {false, sizeX, sizeY};
}

template <typename T>
Grid<T> Grid<T>::newManaged(int sizeX, int sizeY) {
    return {true, sizeX, sizeY};
}

template <typename T>
T *Grid<T>::raw() {
    return field;
}

template <typename T>
void Grid<T>::swapBuffers(Grid<T> &other) {
    T *tempField = this->field;
    this->field = other.field;
    other.field = tempField;
}

template class Grid<float>;
template class Grid<double>;
