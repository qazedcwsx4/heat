#include "heat.h"
#include "bmp/grid_to_bmp.h"

double timeMs();

int main() {
    heat("jddd.txt");
    convert("jddd.txt", "jddd.bmp");
}