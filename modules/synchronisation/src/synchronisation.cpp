//
// Created by qaze on 28.11.2021.
//

#include "../include/synchronisation.h"

Synchronisation::Synchronisation(int threadCount) : barrier(std::make_shared<Barrier>(threadCount)){}

Synchronisation::Synchronisation(const Synchronisation &a) :barrier(a.barrier) {}

void Synchronisation::synchronise() {
    barrier->wait();
}
