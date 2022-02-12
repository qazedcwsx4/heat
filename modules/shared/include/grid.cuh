//
// Created by qaze on 12.05.2021.
//

#ifndef HEAT_GRID_CUH
#define HEAT_GRID_CUH

template <typename T>
class Grid {
private:
    template <typename G>
    class GridInner {
    private:
        G *field;
        const int x;
        const int sizeY;

    public:
        GridInner<G>(G *field, int x, int sizeY) :
                field{field}, x{x}, sizeY{sizeY} {}

        G &operator[](int y) {
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

    char* borderCache;

    static Grid<T> newCpu(int sizeX, int sizeY);

    static Grid<T> newManaged(int sizeX, int sizeY);

    GridInner<T> operator[](int x);

    void swapBuffers(Grid &other);

    char isBorder(int i);
    char isBorderCached(int i);

    T* raw();

    ~Grid();
};

#endif //HEAT_GRID_CUH
