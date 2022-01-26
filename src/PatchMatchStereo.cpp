#include <cstring>
#include <iostream>
#include <random>
#include <filesystem>

#include "PMSPropagation.hpp"
#include "Utils.hpp"

#include "stb_image_write.h"

const std::filesystem::path DEBUG_PATH = "img/debug";

constexpr auto INVALID_FLOAT = std::numeric_limits<float>::infinity();

static void OutputDebugImg(int32_t width, int32_t height, int32_t channels,
                           uint8_t *data, const std::string &name) {
    std::filesystem::path path = DEBUG_PATH / name;
    if (!path.has_extension()) {
        path += ".png";
    } else if (path.extension() != ".png") {
        path.replace_extension(".png");
    }
    stbi_write_png(path.generic_string().c_str(), width, height, channels, data,
                   0);
}

static void OutputDebugImg(int32_t width, int32_t height, int32_t channels,
                           float *data, float min, float max,
                           const std::string &name) {
    std::vector<uint8_t> integer_img(width * height * channels);
    for (size_t i = 0; i < integer_img.size(); ++i) {
        integer_img[i] = std::round(
            data[i] / (max - min) *
            std::numeric_limits<decltype(integer_img)::value_type>::max());
    }
    OutputDebugImg(width, height, channels, integer_img.data(), name);
}

static void OutputDebugImg(int32_t width, int32_t height,
                           PatchMatchStereo::Gradient *data,
                           const std::string &name) {
    std::vector<float> combine_grad(height * width);
    float min = std::numeric_limits<float>::max();
    float max = std::numeric_limits<float>::min();
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const auto p = y * width + x;
            const auto grad = data[p];
            combine_grad[y * width + x] = 0.5 * grad.x + 0.5 * grad.y;
            min = std::min(min, combine_grad[p]);
            max = std::max(max, combine_grad[p]);
        }
    }
    OutputDebugImg(width, height, 1, combine_grad.data(), min, max, name);
}

PatchMatchStereo::PatchMatchStereo()
    : m_is_initialized(false), m_width(0), m_height(0) {}

PatchMatchStereo::~PatchMatchStereo() {
    Timer timer;
    if (m_option.is_debug) {
        std::cout << "writing debug images..." << std::endl;
        timer.Restart();
        if (!std::filesystem::exists(DEBUG_PATH)) {
            std::filesystem::create_directories(DEBUG_PATH);
        }
        OutputDebugImg(m_width, m_height, 1, m_left_gray.data(), "left_gray");
        OutputDebugImg(m_width, m_height, 1, m_right_gray.data(), "right_gray");
        OutputDebugImg(m_width, m_height, m_left_grad.data(), "left_gradient");
        OutputDebugImg(m_width, m_height, m_right_grad.data(),
                       "right_gradient");
        OutputDebugImg(m_width, m_height, 1, m_left_cost.data(), 0, 255,
                       "left_cost");
        OutputDebugImg(m_width, m_height, 1, m_right_cost.data(), 0, 255,
                       "right_cost");
        OutputDebugImg(m_width, m_height, 1, m_left_disparity.data(),
                       m_option.min_disparity, m_option.max_disparity,
                       "left_disparity");
        OutputDebugImg(m_width, m_height, 1, m_right_disparity.data(),
                       -m_option.max_disparity, -m_option.min_disparity,
                       "right_disparity");
        std::cout << "images writing took " << timer.GetElapsedMS() << " ms."
                  << std::endl;
    }
}

bool PatchMatchStereo::Init(int32_t width, int32_t height,
                            const Option &option) {
    if (width <= 0 || height <= 0) {
        return false;
    }
    m_width = width;
    m_height = height;
    m_option = option;
    std::cout << "initializing patch match stereo..." << std::endl;

    //··· 开辟内存空间
    const int32_t img_size = width * height;
    // const int32_t disp_range = option.max_disparity -
    // option.min_disparity; 灰度数据
    m_left_gray.resize(img_size);
    m_right_gray.resize(img_size);
    // 梯度数据
    m_left_grad.resize(img_size);
    m_right_grad.resize(img_size);
    // 代价数据
    m_left_cost.resize(img_size);
    m_right_cost.resize(img_size);
    // 视差图
    m_left_disparity.resize(img_size);
    m_right_disparity.resize(img_size);
    // 平面集
    m_left_plane.resize(img_size);
    m_right_plane.resize(img_size);

    m_is_initialized = true;

    return m_is_initialized;
}

bool PatchMatchStereo::Match(const uint8_t *left_img, const uint8_t *right_img,
                             float *left_disparity) {
    if (!m_is_initialized) {
        std::cout << "patch match stereo not init!" << std::endl;
        return false;
    }

    if (left_img == nullptr || right_img == nullptr) {
        return false;
    }
    m_left_img = left_img;
    m_right_img = right_img;

    PreCompute();

    Propagation();

    if (m_option.is_check_lr) {
        LRCheck();
        std::cout << "left mismatches: " << m_left_mismatches.size()
                  << " right mismatches: " << m_right_mismatches.size()
                  << std::endl;
    }

    if (m_option.is_fill_hole) {
        FillHole();
    }

    if (left_disparity) {
        memcpy(left_disparity, m_left_disparity.data(),
               m_width * m_height *
                   sizeof(decltype(m_left_disparity)::value_type));
    }

    return true;
}

void PatchMatchStereo::RandomInit(DisparityPlane *plane, float *disparity,
                                  int width, int height, const Option &option,
                                  int8_t sign) {
    using FloatRndFunc = float (*)(float, float);
    auto rand_d = std::bind<FloatRndFunc>(Random::Uniform, option.min_disparity,
                                          option.max_disparity);
    auto rand_n = std::bind<FloatRndFunc>(Random::Uniform, -1.f, 1.f);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const int32_t p = y * width + x;
            float disp = sign * rand_d();
            if (option.is_integer_disp) {
                disp = std::round(disp);
            }
            disparity[p] = disp;

            Vector3f norm;
            if (!option.is_force_fpw) {
                norm.x = rand_n();
                norm.y = rand_n();
                float z = rand_n();
                while (z == 0) {
                    z = rand_n();
                }
                norm.z = z;
                norm = Normalize(norm);
            } else {
                norm = Vector3f(0, 0, 1);
            }
            plane[p] = DisparityPlane(x, y, norm, disp);
        }
    }
}

void PatchMatchStereo::ComputeGray(const uint8_t *img, uint8_t *gray,
                                   int32_t width, int32_t height) {
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const auto b = img[y * width * 3 + 3 * x];
            const auto g = img[y * width * 3 + 3 * x + 1];
            const auto r = img[y * width * 3 + 3 * x + 2];
            gray[y * width + x] = r * 0.299 + g * 0.587 + b * 0.114;
        }
    }
}

void PatchMatchStereo::ComputeGradient(const uint8_t *gray, Gradient *grad,
                                       int32_t width, int32_t height) {
    // Sobel梯度算子
    for (int32_t y = 1; y < height - 1; ++y) {
        for (int32_t x = 1; x < width - 1; ++x) {
            const auto grad_x =
                (-gray[(y - 1) * width + x - 1] +
                 gray[(y - 1) * width + x + 1]) +
                (-2 * gray[y * width + x - 1] + 2 * gray[y * width + x + 1]) +
                (-gray[(y + 1) * width + x - 1] +
                 gray[(y + 1) * width + x + 1]);
            const auto grad_y =
                (-gray[(y - 1) * width + x - 1] -
                 2 * gray[(y - 1) * width + x] -
                 gray[(y - 1) * width + x + 1]) +
                (gray[(y + 1) * width + x - 1] + 2 * gray[(y + 1) * width + x] +
                 gray[(y + 1) * width + x + 1]);

            // clamp it to 0-255
            grad[y * width + x].x = grad_x / 8.f;
            grad[y * width + x].y = grad_y / 8.f;
        }
    }
}

void PatchMatchStereo::PreCompute() {
    ThreadPool pool(std::thread::hardware_concurrency());

    pool.Queue(PatchMatchStereo::RandomInit, m_left_plane.data(),
               m_left_disparity.data(), m_width, m_height, m_option, 1);
    pool.Queue(PatchMatchStereo::RandomInit, m_right_plane.data(),
               m_right_disparity.data(), m_width, m_height, m_option, -1);

    pool.Queue(PatchMatchStereo::ComputeGray, m_left_img, m_left_gray.data(),
               m_width, m_height);
    pool.Queue(PatchMatchStereo::ComputeGray, m_right_img, m_right_gray.data(),
               m_width, m_height);

    pool.Queue(PatchMatchStereo::ComputeGradient, m_left_gray.data(),
               m_left_grad.data(), m_width, m_height);
    pool.Queue(PatchMatchStereo::ComputeGradient, m_right_gray.data(),
               m_right_grad.data(), m_width, m_height);
}

void PatchMatchStereo::Propagation() {
    Timer timer;

    std::cout << "initializing cost propagation data" << std::endl;
    timer.Restart();

    Option option_right = m_option;
    option_right.min_disparity = -m_option.max_disparity;
    option_right.max_disparity = -m_option.min_disparity;

    PMSPropagation left_propagation(
        m_left_img, m_right_img, m_left_grad.data(), m_right_grad.data(),
        m_width, m_height, m_option, m_left_plane.data(), m_right_plane.data(),
        m_left_cost.data(), m_right_cost.data());
    PMSPropagation right_propagation(
        m_right_img, m_left_img, m_right_grad.data(), m_left_grad.data(),
        m_width, m_height, option_right, m_right_plane.data(),
        m_left_plane.data(), m_right_cost.data(), m_left_cost.data());
    std::cout << "cost data initialization took " << timer.GetElapsedMS()
              << " ms." << std::endl;

    for (int32_t i = 0; i < m_option.num_iters; ++i) {
        std::cout << "computing propagation " << i + 1 << " of "
                  << m_option.num_iters << "..." << std::endl;
        timer.Restart();
        left_propagation.DoPropagation();
        right_propagation.DoPropagation();
        std::cout << "propagation took " << timer.GetElapsedMS() << " ms."
                  << std::endl;
    }

    ThreadPool pool(2);

    pool.Queue(PatchMatchStereo::PlaneToDisparity, m_left_plane.data(),
               m_left_disparity.data(), m_width, m_height);
    pool.Queue(PatchMatchStereo::PlaneToDisparity, m_right_plane.data(),
               m_right_disparity.data(), m_width, m_height);
}

void PatchMatchStereo::LRCheck() {
    for (int k = 0; k < 2; ++k) {
        auto *disp_left =
            k == 0 ? m_left_disparity.data() : m_right_disparity.data();
        auto *disp_right =
            k == 0 ? m_right_disparity.data() : m_left_disparity.data();
        auto &mismatches = k == 0 ? m_left_mismatches : m_right_mismatches;

        mismatches.clear();
        for (int32_t y = 0; y < m_height; ++y) {
            for (int32_t x = 0; x < m_width; ++x) {
                // 左影像视差值
                auto &disp = disp_left[y * m_width + x];

                if (disp == INVALID_FLOAT) {
                    mismatches.emplace_back(x, y);
                    continue;
                }

                // 根据视差值找到右影像上对应的同名像素
                const auto x_right = lround(x - disp);

                if (x_right >= 0 && x_right < m_width) {
                    // 右影像上同名像素的视差值
                    auto &disp_r = disp_right[y * m_width + x_right];

                    // 判断两个视差值是否一致（差值在阈值内为一致）
                    // 在本代码里，左右视图的视差值符号相反
                    if (std::abs(disp + disp_r) > m_option.lrcheck_thresh) {
                        // 让视差值无效
                        disp = INVALID_FLOAT;
                        mismatches.emplace_back(x, y);
                    }
                } else {
                    // 通过视差值在右影像上找不到同名像素（超出影像范围）
                    disp = INVALID_FLOAT;
                    mismatches.emplace_back(x, y);
                }
            }
        }
    }
}

void PatchMatchStereo::FillHole() {
    const auto task = [this](float *disparity,
                             std::vector<Vector2i> &mismatches) {
        if (mismatches.empty()) {
            return;
        }
        // 存储每个待填充像素的视差
        std::vector<float> fill_disps(mismatches.size());
        for (size_t n = 0; n < mismatches.size(); ++n) {
            const auto x = mismatches[n].x;
            const auto y = mismatches[n].y;
            std::vector<Vector2i> candidates;

            // 向左向右各搜寻第一个有效像素，记录平面
            auto xs = x + 1;
            while (xs < m_width) {
                if (disparity[y * m_width + xs] != INVALID_FLOAT) {
                    candidates.emplace_back(xs, y);
                    break;
                }
                ++xs;
            }
            xs = x - 1;
            while (xs >= 0) {
                if (disparity[y * m_width + xs] != INVALID_FLOAT) {
                    candidates.emplace_back(xs, y);
                    break;
                }
                --xs;
            }

            if (candidates.empty()) {
                fill_disps[n] = disparity[y * m_width + x];
                continue;
            } else if (candidates.size() == 1) {
                fill_disps[n] =
                    disparity[m_width * candidates[0].y + candidates[0].x];
            } else {
                // 选择较小的视差
                auto dp1 =
                    disparity[m_width * candidates[0].y + candidates[0].x];
                auto dp2 =
                    disparity[m_width * candidates[1].y + candidates[1].x];
                fill_disps[n] = std::abs(dp1) < std::abs(dp2) ? dp1 : dp2;
            }
        }
        for (size_t n = 0; n < mismatches.size(); ++n) {
            const int32_t x = mismatches[n].x;
            const int32_t y = mismatches[n].y;
            disparity[y * m_width + x] = fill_disps[n];
        }
    };
    ThreadPool pool(std::thread::hardware_concurrency());
    auto left_ret =
        pool.Queue(task, m_left_disparity.data(), m_left_mismatches);
    auto right_ret =
        pool.Queue(task, m_right_disparity.data(), m_right_mismatches);

    left_ret.get();
    pool.Queue(&PatchMatchStereo::WeightMedianFilter, this, m_left_img,
               m_left_mismatches, m_left_disparity.data());
    right_ret.get();
    pool.Queue(&PatchMatchStereo::WeightMedianFilter, this, m_right_img,
               m_right_mismatches, m_right_disparity.data());
}

void PatchMatchStereo::WeightMedianFilter(
    const uint8_t *img_data, const std::vector<Vector2i> &mismatches,
    float *disparity) const {
    const int32_t wnd_size2 = m_option.patch_size / 2;

    const auto color = [](const uint8_t *img_data, int32_t width, int32_t x,
                          int32_t y) -> Color {
        auto *pixel = img_data + y * width * 3 + 3 * x;
        return {pixel[0], pixel[1], pixel[2]};
    };
    // 带权视差集
    std::vector<std::pair<float, float>> disps;
    disps.reserve(m_option.patch_size * m_option.patch_size);

    for (auto &pix : mismatches) {
        const int32_t x = pix.x;
        const int32_t y = pix.y;
        // weighted median filter
        disps.clear();
        const auto &col_p = color(img_data, m_width, x, y);
        float total_w = 0.0f;
        for (int32_t r = -wnd_size2; r <= wnd_size2; r++) {
            for (int32_t c = -wnd_size2; c <= wnd_size2; c++) {
                const int32_t yr = y + r;
                const int32_t xc = x + c;
                if (yr < 0 || yr >= m_height || xc < 0 || xc >= m_width) {
                    continue;
                }
                const auto &disp = disparity[yr * m_width + xc];
                if (disp == INVALID_FLOAT) {
                    continue;
                }
                // 计算权值
                const auto &col_q = color(img_data, m_width, xc, yr);
                const auto dc = abs(col_p.r - col_q.r) +
                                abs(col_p.g - col_q.g) + abs(col_p.b - col_q.b);
                const auto w = exp(-dc / m_option.gamma);
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
                disparity[y * m_width + x] = wd.first;
                break;
            }
        }
    }
}

void PatchMatchStereo::PlaneToDisparity(const DisparityPlane *plane,
                                        float *disparity, int32_t width,
                                        int32_t height) {
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const size_t p = x + y * width;
            disparity[p] = plane[p].GetDisparity(x, y);
        }
    }
}
