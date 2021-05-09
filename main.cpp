#include "heat/heat.h"
#include "output/grid_to_bmp.h"

#define SIZE_X 1000
#define SIZE_Y 1000

int main() {
    Grid grid = Grid(SIZE_X, SIZE_Y);
    setupTest(grid);
    heat(grid, "jddd.txt");

    convert("jddd.txt", "jddd.bmp");
}