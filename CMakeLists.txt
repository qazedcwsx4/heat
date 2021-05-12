cmake_minimum_required(VERSION 3.17)
project(heat CXX CUDA)

set(CMAKE_CXX_STANDARD 14)

message("compiler is ${CMAKE_CXX_COMPILER_ID}")
if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "-Ox -fp:fast")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

add_subdirectory(modules/shared)
add_subdirectory(modules/cpu)
add_subdirectory(modules/gpu)

add_executable(heat src/main.cpp src/Grid.cpp src/heat/heat.cpp src/output/grid_to_bmp.cpp src/output/grid_to_bmp.h src/heat/heat.h modules/gpu/main.cu)

target_include_directories(${PROJECT_NAME} PUBLIC ${shared_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${heat_cpu_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${heat_gpu_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} heat_shared)
target_link_libraries(${PROJECT_NAME} heat_cpu)
target_link_libraries(${PROJECT_NAME} heat_gpu)