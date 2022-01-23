#include <cstring>
#include <iostream>
#include <random>
#include <filesystem>

#include "PMSPropagation.hpp"
#include "Utils.hpp"

#include "stb_image_write.h"

const std::filesystem::path DEBUG_PATH = "img/debug";

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
                           float *data, const std::string &name) {
    std::vector<uint8_t> integer_img(width * height * channels);
    for (size_t i = 0; i < integer_img.size(); ++i) {
        integer_img[i] = std::round(data[i]) / 64 * 255;
    }
    OutputDebugImg(width, height, channels, integer_img.data(), name);
}

static void OutputDebugImg(int32_t width, int32_t height,
                           PatchMatchStereo::Gradient *data,
                           const std::string &name) {
    std::vector<float> combine_grad(height * width);
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const auto grad = data[y * width + x];
            combine_grad[y * width + x] = 0.5 * grad.x + 0.5 * grad.y;
        }
    }
    OutputDebugImg(width, height, 1, combine_grad.data(), name);
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
        OutputDebugImg(m_width, m_height, 1, m_left_cost.data(), "left_cost");
        OutputDebugImg(m_width, m_height, 1, m_right_cost.data(), "right_cost");
        OutputDebugImg(m_width, m_height, 1, m_left_disparity.data(),
                       "left_disparity");
        OutputDebugImg(m_width, m_height, 1, m_right_disparity.data(),
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

    OutputDisparity(left_disparity);

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
    ThreadPool pool(2);
    Timer timer;

    std::cout << "initializing cost propagation data" << std::endl;
    timer.Restart();
    PMSPropagation left_propagation(
        m_left_img, m_right_img, m_left_grad.data(), m_right_grad.data(),
        m_width, m_height, m_option, m_left_plane.data(), m_right_plane.data(),
        m_left_cost.data(), m_right_cost.data());
    PMSPropagation right_propagation(
        m_right_img, m_left_img, m_right_grad.data(), m_left_grad.data(),
        m_width, m_height, m_option, m_right_plane.data(), m_left_plane.data(),
        m_right_cost.data(), m_left_cost.data());
    std::cout << "cost data initialization took " << timer.GetElapsedMS()
              << " ms." << std::endl;

    pool.Queue(&PatchMatchStereo::DoPropagation, this, left_propagation,
               "left");
    pool.Queue(&PatchMatchStereo::DoPropagation, this, right_propagation,
               "right");
}

void PatchMatchStereo::OutputDisparity(float *disparity) {
    ThreadPool pool(2);

    pool.Queue(PatchMatchStereo::PlaneToDisparity, m_left_plane.data(),
               m_left_disparity.data(), m_width, m_height);
    pool.Queue(PatchMatchStereo::PlaneToDisparity, m_right_plane.data(),
               m_right_disparity.data(), m_width, m_height);

    if (disparity) {
        memcpy(disparity, m_left_disparity.data(),
               m_width * m_height * sizeof(float));
    }
}

std::mutex cout_mutex;

void PatchMatchStereo::DoPropagation(PMSPropagation &propagation,
                                     const std::string &view_name) {
    Timer timer;
    for (int32_t i = 0; i < m_option.num_iters; ++i) {
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "computing propagation " << i + 1 << " of "
                      << m_option.num_iters << " for " << view_name << "..."
                      << std::endl;
        }
        timer.Restart();
        propagation.DoPropagation();
        {
            std::lock_guard<std::mutex> lock(cout_mutex);
            std::cout << "propagation for " << view_name << " took "
                      << timer.GetElapsedMS() << " ms." << std::endl;
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
