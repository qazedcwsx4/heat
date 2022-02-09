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
    for (int i = 0; i < 1000; ++i) {
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
void CpuComputationUnit<T>::internalStep(int thread) {
    auto previous = this->previous.raw();
    auto current = this->grid.raw();

    int start = this->chunkStart + thread;
    int finish = this->chunkStart + this->chunkSize;

    for (int i = start; i < finish; i += THREAD_COUNT) {
        if (!this->grid.isBorder(i)) {
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - this->grid.sizeY] + previous[i + this->grid.sizeY]) / 4.0;
        } else {
            current[i] = previous[i];
        }
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
