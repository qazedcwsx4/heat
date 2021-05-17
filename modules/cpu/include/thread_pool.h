//
// Created by macie on 17/05/2021.
//

#ifndef MAIN_CPP_THREAD_POOL_H
#define MAIN_CPP_THREAD_POOL_H

#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <queue>
#include <functional>

class ThreadPool {
public:
    const unsigned int numThreads;

    ThreadPool();

    ~ThreadPool();

    void addJob(std::function<void()> job);

private:
    volatile bool terminatePool = false;
    volatile bool stopped = false;

    std::vector<std::thread> pool{};
    std::mutex threadPoolMutex{};

    std::queue<std::function<void()>> queue{};
    std::mutex queueMutex{};

    std::condition_variable condition{};

    [[noreturn]] void magicInfiniteMethod();
};

#endif //MAIN_CPP_THREAD_POOL_H
