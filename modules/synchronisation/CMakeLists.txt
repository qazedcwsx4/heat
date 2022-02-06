cmake_minimum_required(VERSION 3.17)

project(heat_synchronisation CXX)

file(GLOB SOURCE_FILES CONFIGURE_DEPENDS src/*.cpp src/*.cu src/*.h)
set(CMAKE_CXX_STANDARD 17)

add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

set_target_properties(${PROJECT_NAME}
        PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(${PROJECT_NAME})

set(${PROJECT_NAME}_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include
        CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)