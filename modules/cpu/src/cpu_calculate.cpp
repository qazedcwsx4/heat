//
// Created by qaze on 12.05.2021.
//

#include <iomanip>
#include <thread>
#include "../include/cpu_calculate.h"

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

template<typename T>
void cpuCalculate(Grid<T> &grid) {
    const int sizeX = grid.sizeX;
    const int sizeY = grid.sizeY;

    Grid<T> previous = Grid<T>::newCpu(sizeX, sizeY);
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

template void cpuCalculate<float>(Grid<float> &grid);
template void cpuCalculate<double>(Grid<double> &grid);
