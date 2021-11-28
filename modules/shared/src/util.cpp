//
// Created by qaze on 01.11.2021.
//

#include <ctime>
#include "../include/util.h"

double timeMs() {
    return (double) clock() / (double) CLOCKS_PER_SEC;
}