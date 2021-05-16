//
// Created by qaze on 16.05.2021.
//

#include <fstream>
#include "../include/grid_operations.h"

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

