#include <grid.cuh>
#include <grid_operations.h>
#include <string>
#include <gpu_calculate.cuh>
#include <cpu_calculate.h>
#include "output/bmp_converter.h"

#define SIZE_X 100
#define SIZE_Y 100

int main() {
    Grid grid = Grid::newManaged(SIZE_X, SIZE_Y);
    setupTest(grid);

    gpuCalculate(grid);

    saveToFile(grid, "grid.txt");

    convert("grid.txt", "heatmap.bmp");

    return 0;
}