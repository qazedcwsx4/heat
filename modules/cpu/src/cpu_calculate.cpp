//
// Created by qaze on 12.05.2021.
//

#include <iomanip>
#include "../include/cpu_calculate.h"
#include "grid_operations.h"

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

void calculate(Grid &grid){
    const int sizeX = grid.sizeX;
    const int sizeY = grid.sizeY;

    double epsilon = 0.01;

    Grid previous = Grid(sizeX, sizeY);
//
//  iterate until the  new solution W differs from the old solution U
//  by no more than EPSILON.
//
    int iterations = 0;
    int iterations_print = 1;
    double startTime = timeMs();

    double current_difference = epsilon;
    while (epsilon <= current_difference) {
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
