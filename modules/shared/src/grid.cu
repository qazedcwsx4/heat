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
        totalSize(sizeX * sizeY),
        borderlessSize((sizeX - 2) * (sizeY - 2)){
    if (isManaged) {
        cudaMallocManaged((void **) (&field), totalSize * sizeof(T));
        cudaMallocManaged((void **) (&borderlessField), borderlessSize * sizeof(T*));
    } else {
        field = new T[totalSize];
        borderlessField = new T*[borderlessSize];
    }

    int j = 0;
    for(int i = 0; i < totalSize; i++) {
        if(!isBorder(i)) {
            borderlessField[j] = &field[i];
            j++;
        }
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
        cudaFree(borderlessField);
    } else {
        delete[] field;
        delete[] borderlessField;
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
T** Grid<T>::borderlessRaw() {
    return borderlessField;
}

template <typename T>
Grid<T> Grid<T>::copy(){
    Grid<T> copyGrid = Grid<T>(this->isManaged, this->sizeX, this->sizeY);

    for (int i = 0; i < this->totalSize; i++)
    {
        copyGrid.raw()[i] = this->field[i];
    }

    return copyGrid;
}

template <typename T>
void Grid<T>::swapBuffers(Grid<T> &other) {
    T *tempField = this->field;
    this->field = other.field;
    other.field = tempField;

    T** tempBorderlessField = this->borderlessField;
    this->borderlessField = other.borderlessField;
    other.borderlessField = tempBorderlessField;
}

template<typename T>
bool Grid<T>::isBorder(int i) {
    if (i < sizeX || i >= totalSize - sizeX) return true;
    if (i % sizeY == 0 || (i + 1) % sizeY == 0) return true;
    return false;
}

template class Grid<float>;
template class Grid<double>;
