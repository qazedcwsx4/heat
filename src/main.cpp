#include <grid.cuh>
#include <grid_operations.h>
#include <string>
#include <gpu_calculate.cuh>
#include <cpu_calculate.h>
#include "output/bmp_converter.h"

#define SIZE_X 1000
#define SIZE_Y 1000

int main() {
    Grid grid = Grid<float>::newCpu(SIZE_X, SIZE_Y);
    setupTest(grid);

    cpuCalculateParallelProper(grid);

    saveToFile(grid, "grid.txt");

    convert("grid.txt", "heatmap.bmp");

    return 0;
}