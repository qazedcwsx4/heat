#include <string>
#include "heat/heat.h"
#include "output/grid_to_bmp.h"

#define SIZE_X 1000
#define SIZE_Y 1000

#define FILE_PATH "../resources/"

int main() {
    Grid grid = Grid(SIZE_X, SIZE_Y);
    setupTest(grid);

    heat(grid, FILE_PATH + std::string("grid.txt"));
    convert(FILE_PATH + std::string("grid.txt"), FILE_PATH + std::string("heatmap.bmp"));
}