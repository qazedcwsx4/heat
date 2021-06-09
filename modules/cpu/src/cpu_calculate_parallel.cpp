//
// Created by qaze on 20.05.2021.
//

#include <thread>
#include <iomanip>
#include "../include/cpu_calculate.h"

bool finished;

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

void cpuCalculateParallel(Grid &grid) {
    int total = (grid.sizeX * grid.sizeY);
    Grid previous = Grid::newCpu(grid.sizeX, grid.sizeY);

    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();
    do {
        auto *copyThreads = new std::thread[THREAD_COUNT];
        for (int i = 0; i < THREAD_COUNT; i++) {
            copyThreads[i] = std::thread(copy, i, total, grid.raw(), previous.raw());
        }
        for (int i = 0; i < THREAD_COUNT; i++) {
            copyThreads[i].join();
        }
        delete[] copyThreads;

        finished = true;
        auto *stepThreads = new std::thread[THREAD_COUNT];
        for (int i = 0; i < THREAD_COUNT; i++) {
            stepThreads[i] = std::thread(step, i, total, grid.raw(), previous.raw(), grid.sizeY, EPSILON);
        }
        for (int i = 0; i < THREAD_COUNT; i++) {
            stepThreads[i].join();
        }
        delete[] stepThreads;


        iterations++;
        if (iterations == iterations_print) {
            std::cout << "  " << std::setw(8) << iterations << "\n";
            iterations_print = 2 * iterations_print;
        }
    } while (!finished);

    std::cout << "total time " << timeMs() - startTime;
}
