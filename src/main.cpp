#include <grid.cuh>
#include <grid_operations.h>
#include <string>
#include <iostream>
#include "output/bmp_converter.h"
#include "computation/computation.h"
#include <chrono>
#include <unordered_map>

template<typename T>
using ComputationFunctionPtr = Computation<T>(*)(Grid<T> &grid);

enum class ComputationType : int {
    CPU_COMPUTATION,
    GPU_COMPUTATION,
    HYBRID_COMPUTATION,

    // This must be the last element!
    COMPUTATION_TYPE_COUNT
};

template<typename T>
const std::unordered_map<ComputationType, std::pair<ComputationFunctionPtr<T>, std::string>> availableComputationTypes = {
        {ComputationType::CPU_COMPUTATION, std::make_pair(Computation<T>::newCpuComputation, "CPU_COMPUTATION")},
        {ComputationType::GPU_COMPUTATION, std::make_pair(Computation<T>::newGpuComputation, "GPU_COMPUTATION")},
        {ComputationType::HYBRID_COMPUTATION, std::make_pair(Computation<T>::newHybridComputation, "HYBRID_COMPUTATION")},
};

int main() {
    //////////////////// Modify only this line /////////////////////////////////////////////////////////////////////////
    const ComputationType computationType = ComputationType::GPU_COMPUTATION;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    const auto& computationPair = availableComputationTypes<float>.at(computationType);
    const auto computationFunction = computationPair.first;
    const auto computationName = computationPair.second;

    Grid<float> grid = Grid<float>::newManaged(SIZE_X, SIZE_Y);
    setupTest(grid);

    auto start = std::chrono::high_resolution_clock::now();
    computationFunction(grid);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "GRID: [" << grid.sizeX << ", " << grid.sizeY << "]" << std::endl;
    std::cout << computationName << " TIME: " << std::chrono::duration<double, std::milli>(end - start).count();

    saveToFile(grid, "grid.txt");
    convert("grid.txt", "heatmap.bmp");

    return 0;
}