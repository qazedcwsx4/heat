cmake_minimum_required(VERSION 3.17)

project(heat_gpu CXX CUDA)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu)
set(CMAKE_CUDA_STANDARD 14)

add_library(${PROJECT_NAME} SHARED ${SOURCE_FILES})
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE ${heat_shared_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} heat_shared)

set_target_properties(${PROJECT_NAME}
        PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME})

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "61")