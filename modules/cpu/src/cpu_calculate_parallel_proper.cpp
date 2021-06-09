//
// Created by qaze on 20.05.2021.
//

#include <thread>
#include <mutex>
#include <iomanip>
#include "../include/cpu_calculate.h"
#include <barrier>

bool finished = false;
std::barrier barrier(THREAD_COUNT);

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

void copy(int thread, int total, double *source, double *destination) {
    for (int i = thread; i < total; i += THREAD_COUNT) {
        destination[i] = source[i];
    }
}

void step(int thread, int total, double *current, double *previous, int wrap, double epsilon) {
    for (int i = thread; i < total; i += THREAD_COUNT) {
        if (previous[i] != 0.0 && previous[i] != 100.0) {
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        }
        if (fabs(current[i] - previous[i]) > epsilon) finished = false;
    }
}

void doWork(int thread, int total, Grid &current, Grid &previous) {
    int iterations = 0;
    int iterations_print = 1;

    while (!finished) {
        finished = true;
        copy(thread, total, current.raw(), previous.raw());
        barrier.arrive_and_wait();
        step(thread, total, current.raw(), previous.raw(), current.sizeY, EPSILON);
        barrier.arrive_and_wait();

        if (thread == 0) {
            iterations++;
            if (iterations == iterations_print) {
                std::cout << "  " << std::setw(8) << iterations << "\n";
                iterations_print = 2 * iterations_print;
            }
        }
    }
}

void cpuCalculateParallelProper(Grid &grid) {
    int total = (grid.sizeX * grid.sizeY);
    Grid previous = Grid::newCpu(grid.sizeX, grid.sizeY);

    double startTime = timeMs();

    auto *workerThreads = new std::thread[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        workerThreads[i] = std::thread(doWork, i, total, std::ref(grid), std::ref(previous));
    }
    for (int i = 0; i < THREAD_COUNT; i++) {
        workerThreads[i].join();
    }

    std::cout << "total time " << timeMs() - startTime;
}