//
// Created by qaze on 31.10.2021.
//

#include "../include/cpu_computation_unit.h"
#include <iomanip>
#include <iostream>
#include <util.h>

template<typename T>
CpuComputationUnit<T>::CpuComputationUnit(Grid<T> &grid, Grid<T> &previous,
                                          Synchronisation barrier,
                                          int chunkStart, int chunkSize):
        ComputationUnit<T>(grid, previous, barrier, chunkStart, chunkSize) {
    int total = (grid.sizeX * grid.sizeY);

    double startTime = timeMs();

    for (int i = 0; i < THREAD_COUNT; i++) {
        threads[i] = std::thread([=, &grid, &previous]() { doWork(i, total, grid, previous); });
    }
    for (int i = 0; i < THREAD_COUNT; i++) {
        threads[i].join();
    }
    std::cout << "total time " << timeMs() - startTime;
}

template<typename T>
CpuComputationUnit<T>::~CpuComputationUnit() {

}

template<typename T>
void CpuComputationUnit<T>::doWork(int thread, int total, Grid<T> &current, Grid<T> &previous) {
    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();

    finished = false;

    while (!finished) {
        if (thread == 0) {
            current.swapBuffers(previous);
        }
        this->barrier.synchronise();

        finished = true;

        internalStep(thread, total, current.raw(), previous.raw(), current.sizeY, EPSILON);
        this->barrier.synchronise();

        if (thread == 0) {
            iterations++;
            if (iterations == iterations_print) {
                std::cout << "  " << std::setw(8) << iterations << " " << timeMs() - startTime << "\n";
                iterations_print = 2 * iterations_print;
            }
        }
    }
}

template<typename T>
void CpuComputationUnit<T>::internalStep(int thread, int total, T *current, T *previous, int wrap, double epsilon) {
    for (int i = thread; i < total; i += THREAD_COUNT) {
        if (previous[i] != 0.0 && previous[i] != 100.0) {
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        } else {
            current[i] = previous[i];
        }
        if (fabs(current[i] - previous[i]) > epsilon) finished = false;
    }
}

template class CpuComputationUnit<float>;
template class CpuComputationUnit<double>;
