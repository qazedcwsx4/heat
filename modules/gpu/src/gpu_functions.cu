//
// Created by qaze on 01.11.2021.
//

#include "../include/gpu_functions.cuh"
#include <iostream>
#include <cuda_device_runtime_api.h>
#include <iomanip>
#include <cuda_runtime_api.h>
#include <grid.cuh>
#include <util.h>

#define BLOCK_SIZE 256
#define EPSILON 0.0001

