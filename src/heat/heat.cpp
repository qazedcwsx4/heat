#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <string>
#include "heat.h"

double timeMs();

void saveToFile(Grid &grid, const char *filename);

int heat(Grid &grid, const std::string& filename) {
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
    saveToFile(grid, filename.c_str());
    return 0;
}

void setupTest(Grid &grid) {
    // Set unchanging boundaries
    for (int i = 1; i < grid.sizeX - 1; i++) {
        grid[i][0] = 100.0;
        grid[i][grid.sizeY - 1] = 0.0;
    }
    for (int j = 0; j < grid.sizeY; j++) {
        grid[grid.sizeX - 1][j] = 0.0;
        grid[0][j] = 0.0;
    }

    // Calculate mean of boundaries
    double mean = 0.0;
    for (int i = 1; i < grid.sizeX - 1; i++) {
        mean = mean + grid[i][0];
        mean = mean + grid[i][grid.sizeY - 1];
    }
    for (int j = 0; j < grid.sizeY; j++) {
        mean = mean + grid[grid.sizeX - 1][j];
        mean = mean + grid[0][j];
    }
    mean = mean / (double) (2 * grid.sizeX + 2 * grid.sizeY - 4);

    // Set the rest of fields to mean value
    for (int i = 1; i < grid.sizeX - 1; i++) {
        for (int j = 1; j < grid.sizeY - 1; j++) {
            grid[i][j] = mean;
        }
    }
}

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}

void saveToFile(Grid &grid, const char *filename) {
    std::ofstream output;

    output.open(filename);

    output << grid.sizeX << "\n";
    output << grid.sizeY << "\n";

    for (int i = 0; i < grid.sizeX; i++) {
        for (int j = 0; j < grid.sizeY; j++) {
            output << "  " << grid[i][j];
        }
        output << "\n";
    }
    output.close();
}
