//
// Created by qaze on 12.05.2021.
//

#include <cuda_runtime_api.h>
#include "../include/grid.cuh"

/**
 * 0 1 2
 * 3 4 5
 * 6 7 8
 *
 *
 * 0 1 2 3
 * 4 5 6 7
 * 8 . . .
 */


template <typename T>
Grid<T>::Grid(bool isManaged, int sizeX, int sizeY) :
        isManaged{isManaged},
        sizeX{sizeX},
        sizeY{sizeY},
        totalSize(sizeX * sizeY) {

    borderCache = new char[sizeX * sizeY]{0};
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
    delete[] borderCache;
    if (isManaged) {
        cudaFree(field);
    } else {
        delete[] field;
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
T* Grid<T>::raw() {
    return field;
}

template <typename T>
void Grid<T>::swapBuffers(Grid<T> &other) {
    T *tempField = this->field;
    this->field = other.field;
    other.field = tempField;
}

template<typename T>
char Grid<T>::isBorder(int i) {
    if (i < sizeX || i >= totalSize - sizeX) return 1;
    if (i % sizeY == 0 || (i + 1) % sizeY == 0) return 1;
    return -1;
}

template<typename T>
char Grid<T>::isBorderCached(int i) {
    if (this->borderCache[i] == 0)
    {
        this->borderCache[i] = this->isBorder(i);
    }
    return this->borderCache[i];
}

template class Grid<float>;
template class Grid<double>;
