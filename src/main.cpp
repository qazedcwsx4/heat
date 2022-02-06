#include <grid.cuh>
#include <grid_operations.h>
#include <string>
#include <iostream>
#include "output/bmp_converter.h"
#include "computation/computation.h"
#include <chrono>


int main() {
    Grid<float> grid = Grid<float>::newManaged(SIZE_X, SIZE_Y);
    setupTest(grid);

    auto start = std::chrono::high_resolution_clock::now();
    Computation<float>::newCpuComputation(grid);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "TIME: " << std::chrono::duration<double, std::milli>(end - start).count();

    saveToFile(grid, "grid.txt");
    convert("grid.txt", "heatmap.bmp");

    return 0;
}