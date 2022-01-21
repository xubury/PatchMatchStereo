#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>

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

#endif
