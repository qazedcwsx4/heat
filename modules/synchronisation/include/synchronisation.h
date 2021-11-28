//
// Created by qaze on 28.11.2021.
//

#ifndef HEAT_SYNCHRONISATION_H
#define HEAT_SYNCHRONISATION_H


#include <memory>
#include "../src/barrier.h"

class Synchronisation {
private:
    std::shared_ptr<Barrier> barrier;

public:
    Synchronisation(int threadCount);

    Synchronisation(const Synchronisation &a);

    void synchronise();
};


#endif //HEAT_SYNCHRONISATION_H
