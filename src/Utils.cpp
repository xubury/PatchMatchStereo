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

void WeightMedianFilter(const uint8_t *img_ptr, int32_t width, int32_t height,
                        const std::vector<Vector2i> &mismatches,
                        float *disparity, float patch_size, float gamma) {
    const int32_t wnd_size2 = patch_size / 2;

    const auto color = [](const uint8_t *img_data, int32_t width, int32_t x,
                          int32_t y) -> Color {
        auto *pixel = img_data + y * width * 3 + 3 * x;
        return {pixel[0], pixel[1], pixel[2]};
    };
    // 带权视差集
    std::vector<std::pair<float, float>> disps;
    disps.reserve(patch_size * patch_size);

    for (auto &pix : mismatches) {
        const int32_t x = pix.x;
        const int32_t y = pix.y;
        // weighted median filter
        disps.clear();
        const auto &col_p = color(img_ptr, width, x, y);
        float total_w = 0.0f;
        for (int32_t r = -wnd_size2; r <= wnd_size2; r++) {
            for (int32_t c = -wnd_size2; c <= wnd_size2; c++) {
                const int32_t yr = y + r;
                const int32_t xc = x + c;
                if (yr < 0 || yr >= height || xc < 0 || xc >= width) {
                    continue;
                }
                const auto disp = disparity[yr * width + xc];
                if (disp == INVALID_FLOAT) {
                    continue;
                }
                // 计算权值
                const auto &col_q = color(img_ptr, width, xc, yr);
                const auto dc = abs(col_p.r - col_q.r) +
                                abs(col_p.g - col_q.g) + abs(col_p.b - col_q.b);
                const auto w = exp(-dc / gamma);
                total_w += w;

                // 存储带权视差
                disps.emplace_back(disp, w);
            }
        }

        // --- 取加权中值
        // 按视差值排序
        std::sort(disps.begin(), disps.end());
        const float median_w = total_w / 2;
        float w = 0.0f;
        for (const auto &wd : disps) {
            w += wd.second;
            if (w >= median_w) {
                disparity[y * width + x] = wd.first;
                break;
            }
        }
    }
}
