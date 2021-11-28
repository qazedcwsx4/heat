//
// Created by qaze on 31.10.2021.
//

#ifndef HEAT_CPU_COMPUTATION_UNIT_H
#define HEAT_CPU_COMPUTATION_UNIT_H

#include <computation_unit.h>
#include <barrier>
#include <thread>

#define EPSILON 0.01
#define THREAD_COUNT 8

template<typename T>
class CpuComputationUnit : public ComputationUnit<T> {
private:
    bool finished = false;
    std::thread threads[THREAD_COUNT];

    void doWork(int thread);

    void internalStep(int thread);

public:
    CpuComputationUnit(Grid<T> &grid, Grid<T> &previous, Synchronisation barrier, int chunkStart, int chunkSize, bool leader);

    void await();

    ~CpuComputationUnit();
};


#endif //HEAT_CPU_COMPUTATION_UNIT_H
