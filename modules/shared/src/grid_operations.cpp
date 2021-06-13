//
// Created by qaze on 16.05.2021.
//

#include <fstream>
#include "../include/grid_operations.h"

template <typename T>
void saveToFile(Grid<T> &grid, const char *filename) {
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

template <typename T>
void setupTest(Grid<T> &grid) {
// Set unchanging boundaries
    for (int i = 1; i < (grid.sizeX - 1) / 2; i++) {
        grid[i][0] = 100.0;
        grid[i][grid.sizeY - 1] = 0.0;
    }

    for (int i = (grid.sizeX - 1) / 2; i < (grid.sizeX - 1); i++) {
        grid[i][0] = 0.0;
        grid[i][grid.sizeY - 1] = 100.0;
    }
    for (int j = 0; j < grid.sizeY / 2; j++) {
        grid[grid.sizeX - 1][j] = 100.0;
        grid[0][j] = 0.0;
    }

    for (int j = grid.sizeY / 2; j < grid.sizeY; j++) {
        grid[grid.sizeX - 1][j] = 0.0;
        grid[0][j] = 100.0;
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

template void setupTest<float>(Grid<float> &grid);
template void setupTest<double>(Grid<double> &grid);

template void saveToFile<float>(Grid<float> &grid, const char *filename);
template void saveToFile<double>(Grid<double> &grid, const char *filename);
