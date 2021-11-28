//
// Created by qaze on 12.05.2021.
//

#ifndef HEAT_GRID_CUH
#define HEAT_GRID_CUH

template <typename T>
class Grid {
private:
    template <typename T>
    class GridInner {
    private:
        T *field;
        const int x;
        const int sizeY;

    public:
        GridInner<T>(T *field, int x, int sizeY) :
                field{field}, x{x}, sizeY{sizeY} {}

        T &operator[](int y) {
            return field[x * sizeY + y];
        }
    };

    T *field;

    Grid<T>(bool isManaged, int sizeX, int sizeY);

public:
    const int sizeX;
    const int sizeY;
    const int totalSize;

    const bool isManaged;

    static Grid<T> newCpu(int sizeX, int sizeY);

    static Grid<T> newManaged(int sizeX, int sizeY);

    GridInner<T> operator[](int x);

    T *raw();

    void swapBuffers(Grid &other);

    bool isBorder(int i);

    ~Grid();
};

#endif //HEAT_GRID_CUH
