#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <functional>
#include <future>
#include <queue>

using ClockType = std::chrono::high_resolution_clock;

class Timer {
   public:
    Timer() : m_last_ticks(ClockType::now()) {}

    uint64_t GetElapsedMS() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   ClockType::now() - m_last_ticks)
            .count();
    }

    uint64_t GetElapsedSec() const {
        return std::chrono::duration_cast<std::chrono::seconds>(
                   ClockType::now() - m_last_ticks)
            .count();
    }

    // Restart the clock, and return elapsed millisecond.
    uint64_t Restart() {
        uint32_t elapsed = GetElapsedMS();
        m_last_ticks = ClockType::now();
        return elapsed;
    }

   private:
    std::chrono::time_point<ClockType> m_last_ticks;
};

namespace Random {

// generate uniform random range in [0, 1]
double Uniform();

float Uniform(float min, float max);
double Uniform(double min, double max);
int32_t Uniform(int32_t min, int32_t max);

void Seed(uint32_t seed);

};  // namespace Random

class ThreadPool {
   public:
    ThreadPool(uint32_t threads);
    ~ThreadPool();

    template <typename F, typename... ARGS>
    decltype(auto) Queue(F&& f, ARGS&&... args);

   private:
    std::vector<std::thread> m_workers;
    std::queue<std::packaged_task<void()>> m_tasks;

    std::mutex m_mutex;
    bool m_stop;
    std::condition_variable m_condition;
};

template <class F, class... ARGS>
decltype(auto) ThreadPool::Queue(F&& f, ARGS&&... args) {
    using return_type = typename std::result_of<F(ARGS...)>::type;

    std::packaged_task<return_type()> task(
        std::bind(std::forward<F>(f), std::forward<ARGS>(args)...));

    std::future<return_type> res = task.get_future();
    {
        std::unique_lock<std::mutex> lock(m_mutex);

        // don't allow enqueueing after stopping the pool
        if (m_stop) throw std::runtime_error("enqueue on stopped ThreadPool");

        m_tasks.emplace(std::move(task));
    }
    m_condition.notify_one();
    return res;
}

#endif
