#include <grid.cuh>
#include <grid_operations.h>
#include <string>
#include <cpu_calculate.h>
#include "output/grid_to_bmp.h"

#define SIZE_X 1000
#define SIZE_Y 1000

int main() {
    Grid grid = Grid::newManaged(SIZE_X, SIZE_Y);
    setupTest(grid);

    calculate(grid);

    saveToFile(grid, "grid.txt");

    convert("grid.txt", "heatmap.bmp");

    return 0;
}