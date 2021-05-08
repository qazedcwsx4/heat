//
// Created by qaze on 08.05.2021.
//

#ifndef HEAT_GRID_CPP
#define HEAT_GRID_CPP


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

    int sizeX;
    int sizeY;
    double *field;
public:
    Grid(int sizeX, int sizeY) :
            sizeX{sizeX},
            sizeY{sizeY},
            field{new double[sizeX * sizeY]} {
    }

    Grid(int sizeX, int sizeY, double *field) :
            sizeX{sizeX},
            sizeY{sizeY},
            field{field} {
    }
// [12][1234567]
    GridInner operator[](int x) {
        return {field, x, sizeY};
    }

    double &get(int x, int y) {
        return field[y * sizeX + x];
    }
};


#endif //HEAT_GRID_CPP
