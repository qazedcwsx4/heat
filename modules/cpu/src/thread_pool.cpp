//
// Created by macie on 17/05/2021.
//
#include "../include/thread_pool.h"

ThreadPool::ThreadPool() : numThreads(std::thread::hardware_concurrency()) {
    for (int i = 0; i < this->numThreads; i++) {
        pool.emplace_back(std::thread(&ThreadPool::magicInfiniteMethod, this));
    }
}

ThreadPool::~ThreadPool() {
    if (!stopped) {
        std::unique_lock<std::mutex> lock(threadPoolMutex);
        terminatePool = true;

        condition.notify_all();
        for (std::thread &thread : pool) {
            thread.join();
        }
        pool.clear();
        stopped = true;
    }
}

void ThreadPool::magicInfiniteMethod() {
    std::function<void()> job;
    while (true) {
        {
            std::unique_lock<std::mutex> lock(queueMutex);

            condition.wait(lock, [this] { return !queue.empty() || terminatePool; });
            if (terminatePool) return;

            job = queue.front();
            queue.pop();
        }
        job();
    }
}

void ThreadPool::addJob(std::function<void()> job) {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        queue.push(job);
    }
    condition.notify_one();
}
