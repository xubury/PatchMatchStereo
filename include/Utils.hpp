#ifndef UTILS_HPP
#define UTILS_HPP

#include <chrono>
#include <functional>
#include <future>
#include <filesystem>
#include <queue>

// TODO: Put this to another separate module and
//  make an option for no glm dependency in the future
#include <glm/glm.hpp>
using Color = glm::i8vec3;
using Vector2f = glm::vec2;
using Vector3f = glm::vec3;
using Vector2i = glm::i32vec2;

inline const auto& Dot =
    static_cast<float (*)(const Vector3f&, const Vector3f&)>(glm::dot);
inline const auto& Normalize =
    static_cast<Vector3f (*)(const Vector3f&)>(glm::normalize);
/////////////////////////////////////////////////////////////////////

#include "stb_image_write.h"

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

constexpr auto INVALID_FLOAT = std::numeric_limits<float>::infinity();

template <typename T>
inline uint8_t Hamming(const T x, const T y) {
    T dist = 0;
    T val = x ^ y;
    while (val) {
        ++dist;
        val &= val - 1;
    }
    return dist;
}

const std::filesystem::path DEBUG_PATH = "img/debug";

inline void OutputDebugImg(int32_t width, int32_t height, int32_t channels,
                           uint8_t* data, const std::string& name) {
    if (!std::filesystem::exists(DEBUG_PATH)) {
        std::filesystem::create_directories(DEBUG_PATH);
    }
    std::filesystem::path path = DEBUG_PATH / name;
    if (!path.has_extension()) {
        path += ".png";
    } else if (path.extension() != ".png") {
        path.replace_extension(".png");
    }
    stbi_write_png(path.generic_string().c_str(), width, height, channels, data,
                   0);
}

inline void OutputDebugImg(int32_t width, int32_t height, int32_t channels,
                           float* data, float min, float max,
                           const std::string& name) {
    std::vector<uint8_t> integer_img(width * height * channels);
    auto range =
        (std::numeric_limits<decltype(integer_img)::value_type>::max() -
         std::numeric_limits<decltype(integer_img)::value_type>::min()) /
        (max - min);
    for (size_t i = 0; i < integer_img.size(); ++i) {
        integer_img[i] = std::round((data[i] - min) * range);
    }
    OutputDebugImg(width, height, channels, integer_img.data(), name);
}

inline void ComputeGray(const uint8_t* img, uint8_t* gray, int32_t width,
                        int32_t height) {
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const auto b = img[y * width * 3 + 3 * x];
            const auto g = img[y * width * 3 + 3 * x + 1];
            const auto r = img[y * width * 3 + 3 * x + 2];
            gray[y * width + x] = r * 0.299 + g * 0.587 + b * 0.114;
        }
    }
}

#endif
