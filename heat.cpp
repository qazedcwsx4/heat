#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <ctime>
#include <string>
#include "Grid.cpp"
#include "heat.h"

double timeMs();

int heat(char *filename) {
#define SIZE_X 1000
#define SIZE_Y 1000

    double start_time;
    double step_difference;
    double epsilon = 0.01;
    int iterations;
    int iterations_print;
    double mean;
    std::ofstream output;
    char* output_filename = filename;

    Grid previous = Grid(SIZE_X, SIZE_Y);
    Grid current = Grid(SIZE_X, SIZE_Y);

    step_difference = epsilon;
//
//  Set the boundary values, which don't change.
//
    for (int i = 1; i < SIZE_X - 1; i++) {
        current[i][0] = 100.0;
        current[i][SIZE_Y - 1] = 0.0;
    }
    for (int j = 0; j < SIZE_Y; j++) {
        current[SIZE_X - 1][j] = 0.0;
        current[0][j] = 0.0;
    }
//
//  Average the boundary values, to come up with a reasonable
//  initial value for the interior.
//
    mean = 0.0;
    for (int i = 1; i < SIZE_X - 1; i++) {
        mean = mean + current[i][0];
        mean = mean + current[i][SIZE_Y - 1];
    }
    for (int j = 0; j < SIZE_Y; j++) {
        mean = mean + current[SIZE_X - 1][j];
        mean = mean + current[0][j];
    }
    mean = mean / (double) (2 * SIZE_X + 2 * SIZE_Y - 4);
//
//  Initialize the interior solution to the mean value.
//
    for (int i = 1; i < SIZE_X - 1; i++) {
        for (int j = 1; j < SIZE_Y - 1; j++) {
            current[i][j] = mean;
        }
    }
//
//  iterate until the  new solution W differs from the old solution U
//  by no more than EPSILON.
//
    iterations = 0;
    iterations_print = 1;
    std::cout << "\n";
    std::cout << " Iteration  Change\n";
    std::cout << "\n";
    start_time = timeMs();

    while (epsilon <= step_difference) {
//
//  Save the old solution in U.
//
        for (int i = 0; i < SIZE_X; i++) {
            for (int j = 0; j < SIZE_Y; j++) {
                previous[i][j] = current[i][j];
            }
        }
//
//  Determine the new estimate of the solution at the interior points.
//  The new solution W is the average of north, south, east and west neighbors.
//
        step_difference = 0.0;
        for (int i = 1; i < SIZE_X - 1; i++) {
            for (int j = 1; j < SIZE_Y - 1; j++) {
                current[i][j] =
                        (previous[i - 1][j] + previous[i + 1][j] + previous[i][j - 1] + previous[i][j + 1]) / 4.0;

                if (step_difference < fabs(current[i][j] - previous[i][j])) {
                    step_difference = fabs(current[i][j] - previous[i][j]);
                }
            }
        }
        iterations++;
        if (iterations == iterations_print) {
            std::cout << "  " << std::setw(8) << iterations
                 << "  " << step_difference << "\n";
            iterations_print = 2 * iterations_print;
        }
    }

    std::cout << "\n";
    std::cout << "  " << std::setw(8) << iterations
         << "  " << step_difference << "\n";
    std::cout << "\n";
    std::cout << "  Error tolerance achieved.\n";
    std::cout << "  CPU time = " << timeMs() - start_time << "\n";


    output.open(output_filename);

    output << SIZE_X << "\n";
    output << SIZE_Y << "\n";

    for (int i = 0; i < SIZE_X; i++) {
        for (int j = 0; j < SIZE_Y; j++) {
            output << "  " << current[i][j];
        }
        output << "\n";
    }
    output.close();
    return 0;
}

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}
