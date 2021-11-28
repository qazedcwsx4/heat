cmake_minimum_required(VERSION 3.17)
project(heat CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

message("compiler is ${CMAKE_CXX_COMPILER_ID}")
message("default debug flags ${CMAKE_CXX_FLAGS_DEBUG}")
message("default release flags ${CMAKE_CXX_FLAGS_RELEASE}")

if (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

message("overridden default debug flags ${CMAKE_CXX_FLAGS_DEBUG}")
message("overridden default release flags ${CMAKE_CXX_FLAGS_RELEASE}")

if (WIN32)
    set(CMAKE_SHARED_LIBRARY_PREFIX "")
endif ()

add_subdirectory(modules/synchronisation)
add_subdirectory(modules/shared)
add_subdirectory(modules/cpu)
add_subdirectory(modules/gpu)

add_executable(heat src/main.cpp src/output/bmp_converter.cpp src/output/bmp_converter.h src/computation/computation.cpp src/computation/computation.h)

target_include_directories(${PROJECT_NAME} PUBLIC ${heat_synchronisation_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${heat_shared_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${heat_cpu_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${heat_gpu_INCLUDE_DIRS})

target_link_libraries(${PROJECT_NAME} heat_synchronisation)
target_link_libraries(${PROJECT_NAME} heat_shared)
target_link_libraries(${PROJECT_NAME} heat_cpu)
target_link_libraries(${PROJECT_NAME} heat_gpu)


#message("partial build commencing")
#message("looking for header files in ${CMAKE_SOURCE_DIR}/include")
#include_directories(${CMAKE_SOURCE_DIR}/include)
#message("looking for precompiled binaries in ${CMAKE_SOURCE_DIR}/lib")
#link_directories(${CMAKE_SOURCE_DIR}/lib)