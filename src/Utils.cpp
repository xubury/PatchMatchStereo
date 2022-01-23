#include "Utils.hpp"

#include <random>

namespace Random {

static std::random_device s_random_device;
static std::mt19937 s_random_engine(s_random_device());
static std::uniform_real_distribution<double> s_distribution(0.0, 1.0);

void Seed(uint32_t seed) { s_random_engine.seed(seed); }

double Uniform() { return s_distribution(s_random_engine); }

float Uniform(float min, float max) { return Uniform() * (max - min) + min; }

double Uniform(double min, double max) { return Uniform() * (max - min) + min; }

int32_t Uniform(int32_t min, int32_t max) {
    return Uniform() * (max - min) + min;
}

}  // namespace Random

ThreadPool::ThreadPool(uint32_t threads) : m_stop(false) {
    for (uint32_t i = 0; i < threads; ++i) {
        m_workers.emplace_back([this] {
            while (true) {
                std::packaged_task<void()> task;
                {
                    std::unique_lock<std::mutex> lock(m_mutex);
                    m_condition.wait(
                        lock, [this] { return m_stop || !m_tasks.empty(); });
                    if (m_stop && m_tasks.empty()) {
                        return;
                    }
                    task = std::move(m_tasks.front());
                    m_tasks.pop();
                }

                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_stop = true;
    }
    m_condition.notify_all();
    for (auto &worker : m_workers) {
        worker.join();
    }
}
