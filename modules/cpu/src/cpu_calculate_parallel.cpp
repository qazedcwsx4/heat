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

template<typename T>
void step(int thread, int total, T *current, T *previous, int wrap, double epsilon) {
    for (int i = thread; i < total; i += THREAD_COUNT) {
        if (previous[i] != 0.0 && previous[i] != 100.0) {
            current[i] = (previous[i - 1] + previous[i + 1] + previous[i - wrap] + previous[i + wrap]) / 4.0;
        } else {
            current[i] = previous[i];
        }
        if (fabs(current[i] - previous[i]) > epsilon) finished = false;
    }
}

template<typename T>
void cpuCalculateParallel(Grid<T> &grid) {
    int total = (grid.sizeX * grid.sizeY);
    Grid<T> previous = Grid<T>::newCpu(grid.sizeX, grid.sizeY);

    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();
    do {
        grid.swapBuffers(previous);

        finished = true;
        auto *stepThreads = new std::thread[THREAD_COUNT];
        for (int i = 0; i < THREAD_COUNT; i++) {
            stepThreads[i] = std::thread([=, &grid, &previous]() { step(i, total, grid.raw(), previous.raw(), grid.sizeY, EPSILON); });
        }
        for (int i = 0; i < THREAD_COUNT; i++) {
            stepThreads[i].join();
        }
        delete[] stepThreads;

        iterations++;
        if (iterations == iterations_print) {
            std::cout << "  " << std::setw(8) << iterations << " " << timeMs() - startTime << "\n";
            iterations_print = 2 * iterations_print;
        }
    } while (!finished);

    std::cout << "total time " << timeMs() - startTime;
}

template void cpuCalculateParallel<float>(Grid<float> &grid);

template void cpuCalculateParallel<double>(Grid<double> &grid);
