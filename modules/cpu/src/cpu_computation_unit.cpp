//
// Created by qaze on 31.10.2021.
//

#include "../include/cpu_computation_unit.h"

template<typename T>
CpuComputationUnit<T>::CpuComputationUnit(Grid<T> &grid, Grid<T> &previous,
                                          Synchronisation barrier,
                                          int chunkStart, int chunkSize, bool leader):
        ComputationUnit<T>(grid, previous, barrier, chunkStart, chunkSize, leader) {
    for (int i = 0; i < THREAD_COUNT; i++) {
        threads[i] = std::thread([=]() { doWork(i); });
    }
}

template<typename T>
CpuComputationUnit<T>::~CpuComputationUnit() {

}

template<typename T>
void CpuComputationUnit<T>::doWork(int thread) {
    for (int i = 0; i < ITERATION_COUNT; ++i) {
        // buffers are swapped by overseer
        if (this->leader && thread == 0) {
            this->grid.swapBuffers(this->previous);
        }
        this->barrier.synchronise();

        internalStep(thread);
        this->barrier.synchronise();
    }
}

template<typename T>
void CpuComputationUnit<T>::internalStep(const int thread) {
    T** previous = this->previous.borderlessRaw();
    T** current = this->grid.borderlessRaw();

    const int start =  this->chunkStart + (this->chunkSize * thread) / THREAD_COUNT;
    const int finish = this->chunkStart + (this->chunkSize * (thread + 1)) / THREAD_COUNT;

    for (int i = start; i < finish; i += 1) {
        *current[i] = (*(previous[i] - 1) + *(previous[i] + 1) + *(previous[i] - this->grid.sizeY) + *(previous[i] + this->grid.sizeY)) / 4.0;
    }
}

template<typename T>
void CpuComputationUnit<T>::await() {
    for (int i = 0; i < THREAD_COUNT; i++) {
        threads[i].join();
    }
}

template class CpuComputationUnit<float>;
template class CpuComputationUnit<double>;
