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
        const int x;
        const int sizeY;

    public:
        GridInner(double *field, int x, int sizeY) :
                field{field}, x{x}, sizeY{sizeY} {}

        double &operator[](int y) {
            return field[x * sizeY + y];
        }
    };

    double *field;

    Grid(bool isManaged, int sizeX, int sizeY);
public:
    const int sizeX;
    const int sizeY;

    const bool isManaged;

    static Grid newCpu(int sizeX, int sizeY);

    static Grid newManaged(int sizeX, int sizeY);

    GridInner operator[](int x);

    ~Grid();
};

#endif //HEAT_GRID_CUH
