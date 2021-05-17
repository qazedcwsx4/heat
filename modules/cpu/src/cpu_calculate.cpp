//
// Created by qaze on 12.05.2021.
//

#include <iomanip>
#include <thread>
#include "../include/cpu_calculate.h"

#define EPSILON 0.01
#define THREAD_COUNT 10

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

void cpuCalculate(Grid &grid) {
    const int sizeX = grid.sizeX;
    const int sizeY = grid.sizeY;

    Grid previous = Grid::newCpu(sizeX, sizeY);
//
//  iterate until the  new solution W differs from the old solution U
//  by no more than EPSILON.
//
    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();

    double current_difference = EPSILON;
    while (EPSILON <= current_difference) {
// copy current to previous
        for (int i = 0; i < sizeY; i++) {
            for (int j = 0; j < sizeY; j++) {
                previous[i][j] = grid[i][j];
            }
        }

        current_difference = 0.0;
        for (int i = 1; i < sizeX - 1; i++) {
            for (int j = 1; j < sizeY - 1; j++) {
                grid[i][j] = (previous[i - 1][j] + previous[i + 1][j] + previous[i][j - 1] + previous[i][j + 1]) / 4.0;

                if (current_difference < fabs(grid[i][j] - previous[i][j])) {
                    current_difference = fabs(grid[i][j] - previous[i][j]);
                }
            }
        }

        iterations++;
        if (iterations == iterations_print) {
            std::cout << "  " << std::setw(8) << iterations
                      << "  " << current_difference << "\n";
            iterations_print = 2 * iterations_print;
        }
    }
    std::cout << "total time " << timeMs() - startTime;
}

bool finished;

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
