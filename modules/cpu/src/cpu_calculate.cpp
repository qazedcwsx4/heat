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
void doWork(int thread, int total, Grid<T> &current, Grid<T> &previous) {
    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();

    while (!finished) {
        if (thread == 0) {
            current.swapBuffers(previous);
        }
        barrier.arrive_and_wait();

        finished = true;

        step(thread, total, current.raw(), previous.raw(), current.sizeY, EPSILON);
        barrier.arrive_and_wait();

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
void cpuCalculate(Grid<T> &grid) {
    int total = (grid.sizeX * grid.sizeY);
    Grid<T> previous = Grid<T>::newCpu(grid.sizeX, grid.sizeY);

    double startTime = timeMs();

    auto *workerThreads = new std::thread[THREAD_COUNT];
    for (int i = 0; i < THREAD_COUNT; i++) {
        workerThreads[i] = std::thread([=, &grid, &previous]() { doWork(i, total, grid, previous); });
    }
    for (int i = 0; i < THREAD_COUNT; i++) {
        workerThreads[i].join();
    }

    std::cout << "total time " << timeMs() - startTime;
}

template void cpuCalculate<float>(Grid<float> &grid);

template void cpuCalculate<double>(Grid<double> &grid);