//
// Created by qaze on 12.05.2021.
//

#ifndef HEAT_GRID_CUH
#define HEAT_GRID_CUH

class Grid {
private:
    class GridInner {
    private:
        double *field;
        int x;
        int sizeY;

    public:
        GridInner(double *field, int x, int sizeY) :
                field{field}, x{x}, sizeY{sizeY} {}

        double &operator[](int y) {
            return field[x * sizeY + y];
        }
    };

    double *field;
public:
    const int sizeX;
    const int sizeY;

    Grid(int sizeX, int sizeY);
    Grid(int sizeX, int sizeY, double *field);

    GridInner operator[](int x);
};

#endif //HEAT_GRID_CUH
