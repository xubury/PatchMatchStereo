#include <cstring>
#include <iostream>
#include <random>
#include <filesystem>
#include <opencv2/opencv.hpp>

#include "PMSPropagation.hpp"
#include "Utils.hpp"

const std::filesystem::path DEBUG_PATH = "img/debug";

static void OutputDebugImg(int rows, int cols, int channels, uint8_t *data,
                           const std::string &name) {
    cv::Mat mat(rows, cols, channels == 3 ? CV_8UC3 : CV_8UC1, data);
    cv::imwrite((DEBUG_PATH / name).generic_string(), mat);
}

static void OutputDebugImg(int rows, int cols, int channels, float *data,
                           const std::string &name) {
    cv::Mat mat(rows, cols, channels == 3 ? CV_32FC3 : CV_32FC1, data);
    cv::imwrite((DEBUG_PATH / name).generic_string(), mat);
}

static void OutputDebugImg(int rows, int cols, PatchMatchStereo::Gradient *data,
                           const std::string &name) {
    std::vector<float> combine_grad(rows * cols);
    for (int32_t y = 0; y < rows; ++y) {
        for (int32_t x = 0; x < cols; ++x) {
            const auto grad = data[y * cols + x];
            combine_grad[y * cols + x] = 0.5 * grad.x + 0.5 * grad.y;
        }
    }
    cv::Mat mat(rows, cols, CV_32FC1, combine_grad.data());
    cv::imwrite((DEBUG_PATH / name).generic_string(), mat);
}

PatchMatchStereo::PatchMatchStereo()
    : m_is_initialized(false), m_width(0), m_height(0) {}

PatchMatchStereo::~PatchMatchStereo() {
    if (m_option.is_debug) {
        if (!std::filesystem::exists(DEBUG_PATH)) {
            std::filesystem::create_directories(DEBUG_PATH);
        }
        OutputDebugImg(m_height, m_width, 1, m_left_gray.data(),
                       "left_gray.png");
        OutputDebugImg(m_height, m_width, 1, m_right_gray.data(),
                       "right_gray.png");
        OutputDebugImg(m_height, m_width, m_left_grad.data(),
                       "left_gradient.png");
        OutputDebugImg(m_height, m_width, m_right_grad.data(),
                       "right_gradient.png");
        OutputDebugImg(m_height, m_width, 1, m_left_cost.data(),
                       "left_cost.png");
        OutputDebugImg(m_height, m_width, 1, m_right_cost.data(),
                       "right_cost.png");
        OutputDebugImg(m_height, m_width, 1, m_left_disparity.data(),
                       "left_disparity.png");
        OutputDebugImg(m_height, m_width, 1, m_right_disparity.data(),
                       "right_disparity.png");
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
    // const int32_t disp_range = option.max_disparity - option.min_disparity;
    // 灰度数据
    m_left_gray.resize(img_size);
    ;
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

    Timer timer;

    std::cout << "initializing random plane..." << std::endl;
    timer.Restart();
    RandomInit();
    std::cout << "random plane init took " << timer.GetElapsedMS() << " ms. "
              << std::endl;

    std::cout << "computing gray scale image..." << std::endl;
    timer.Restart();
    ComputeGray();
    std::cout << "gray scale image compute took " << timer.GetElapsedMS()
              << " ms. " << std::endl;

    std::cout << "computing image gradient..." << std::endl;
    timer.Restart();
    ComputeGradient();
    std::cout << "image gradient compute took " << timer.GetElapsedMS()
              << " ms. " << std::endl;

    std::cout << "computing cost propagation..." << std::endl;
    timer.Restart();
    Propagation();
    std::cout << "cost propagation took " << timer.GetElapsedMS() << " ms. "
              << std::endl;

    std::cout << "computing plane to disparity..." << std::endl;
    timer.Restart();
    PlaneToDisparity();
    std::cout << "plane to disparity compute took " << timer.GetElapsedMS()
              << " ms. " << std::endl;

    if (left_disparity) {
        memcpy(left_disparity, m_left_disparity.data(),
               m_width * m_height * sizeof(float));
    }

    return true;
}

void PatchMatchStereo::RandomInit() {
    using FloatRndFunc = float (*)(float, float);
    auto rand_d = std::bind<FloatRndFunc>(
        Random::Uniform, m_option.min_disparity, m_option.max_disparity);
    auto rand_n = std::bind<FloatRndFunc>(Random::Uniform, -1.f, 1.f);

    for (int k = 0; k < 2; ++k) {
        auto disp_ptr =
            k == 0 ? m_left_disparity.data() : m_right_disparity.data();
        auto plane_ptr = k == 0 ? m_left_plane.data() : m_right_plane.data();
        int32_t sign = k == 0 ? 1 : -1;
        for (int32_t y = 0; y < m_height; ++y) {
            for (int32_t x = 0; x < m_width; ++x) {
                const int32_t p = y * m_width + x;
                float disp = sign * rand_d();
                if (m_option.is_integer_disp) {
                    disp = std::round(disp);
                }
                disp_ptr[p] = disp;

                Vector3f norm;
                if (!m_option.is_force_fpw) {
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
                plane_ptr[p] = DisparityPlane(x, y, norm, disp);
            }
        }
    }
}

void PatchMatchStereo::ComputeGray() {
    for (int k = 0; k < 2; ++k) {
        auto *color = k == 0 ? m_left_img : m_right_img;
        auto *gray = k == 0 ? m_left_gray.data() : m_right_gray.data();
        for (int32_t y = 0; y < m_height; ++y) {
            for (int32_t x = 0; x < m_width; ++x) {
                const auto b = color[y * m_width * 3 + 3 * x];
                const auto g = color[y * m_width * 3 + 3 * x + 1];
                const auto r = color[y * m_width * 3 + 3 * x + 2];
                gray[y * m_width + x] = r * 0.299 + g * 0.587 + b * 0.114;
            }
        }
    }
}

void PatchMatchStereo::ComputeGradient() {
    // Sobel梯度算子
    for (int32_t n = 0; n < 2; ++n) {
        auto gray = n == 0 ? m_left_gray.data() : m_right_gray.data();
        auto grad = n == 0 ? m_left_grad.data() : m_right_grad.data();
        for (int y = 1; y < m_height - 1; ++y) {
            for (int x = 1; x < m_width - 1; ++x) {
                const auto grad_x = (-gray[(y - 1) * m_width + x - 1] +
                                     gray[(y - 1) * m_width + x + 1]) +
                                    (-2 * gray[y * m_width + x - 1] +
                                     2 * gray[y * m_width + x + 1]) +
                                    (-gray[(y + 1) * m_width + x - 1] +
                                     gray[(y + 1) * m_width + x + 1]);
                const auto grad_y = (-gray[(y - 1) * m_width + x - 1] -
                                     2 * gray[(y - 1) * m_width + x] -
                                     gray[(y - 1) * m_width + x + 1]) +
                                    (gray[(y + 1) * m_width + x - 1] +
                                     2 * gray[(y + 1) * m_width + x] +
                                     gray[(y + 1) * m_width + x + 1]);

                // clamp it to 0-255
                grad[y * m_width + x].x = grad_x / 8.f;
                grad[y * m_width + x].y = grad_y / 8.f;
            }
        }
    }
}

void PatchMatchStereo::Propagation() {
    PMSPropagation left_propagation(
        m_left_img, m_right_img, m_left_grad.data(), m_right_grad.data(),
        m_width, m_height, m_option, m_left_plane.data(), m_right_plane.data(),
        m_left_cost.data(), m_right_cost.data());
    PMSPropagation right_propagation(
        m_right_img, m_left_img, m_right_grad.data(), m_left_grad.data(),
        m_width, m_height, m_option, m_right_plane.data(), m_left_plane.data(),
        m_right_cost.data(), m_left_cost.data());
}

void PatchMatchStereo::PlaneToDisparity() {
    for (int k = 0; k < 2; ++k) {
        auto plane_ptr = k == 0 ? m_left_plane.data() : m_right_plane.data();
        auto disp_ptr =
            k == 0 ? m_left_disparity.data() : m_right_disparity.data();
        for (int32_t y = 0; y < m_height; ++y) {
            for (int32_t x = 0; x < m_width; ++x) {
                const size_t p = x + y * m_width;
                disp_ptr[p] = plane_ptr[p].GetDisparity(x, y);
            }
        }
    }
}
