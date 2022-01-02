#include <grid.cuh>
#include <grid_operations.h>
#include <string>
#include "output/bmp_converter.h"
#include "computation/computation.h"

#define SIZE_X 400
#define SIZE_Y 400

int main() {
    Grid grid = Grid<float>::newManaged(SIZE_X, SIZE_Y);
    setupTest(grid);

    Computation<float>::newHybridComputation(grid);
    //cpuCalculate(grid);

    saveToFile(grid, "grid.txt");

    convert("grid.txt", "heatmap.bmp");

    return 0;
}