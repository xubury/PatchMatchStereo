#include <cstring>
#include <iostream>

#include "PMSPropagation.hpp"
#include "Utils.hpp"

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

        if (m_option.debug_level > 1) {
            OutputDebugImg(m_width, m_height, 1, m_left_gray.data(),
                           "left_gray");
            OutputDebugImg(m_width, m_height, 1, m_right_gray.data(),
                           "right_gray");
            OutputDebugImg(m_width, m_height, m_left_grad.data(),
                           "left_gradient");
            OutputDebugImg(m_width, m_height, m_right_grad.data(),
                           "right_gradient");
            OutputDebugImg(m_width, m_height, 1, m_left_cost.data(), 0, 255,
                           "left_cost");
            OutputDebugImg(m_width, m_height, 1, m_right_cost.data(), 0, 255,
                           "right_cost");
        }
        OutputDebugImg(m_width, m_height, 1, m_left_disparity.data(),
                       m_option.min_disparity, m_option.max_disparity,
                       "pms_left_disp");
        OutputDebugImg(m_width, m_height, 1, m_right_disparity.data(),
                       -m_option.max_disparity, -m_option.min_disparity,
                       "pms_right_disp");
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

    //?????? ??????????????????
    const int32_t img_size = width * height;
    // const int32_t disp_range = option.max_disparity -
    // option.min_disparity; ????????????
    m_left_gray.resize(img_size);
    m_right_gray.resize(img_size);
    // ????????????
    m_left_grad.resize(img_size);
    m_right_grad.resize(img_size);
    // ????????????
    m_left_cost.resize(img_size);
    m_right_cost.resize(img_size);
    // ?????????
    m_left_disparity.resize(img_size);
    m_right_disparity.resize(img_size);
    // ?????????
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
                                  int width, int height, int32_t min_disparity,
                                  int32_t max_disparity, bool is_integer_disp,
                                  bool is_force_fpw) {
    using FloatRndFunc = float (*)(float, float);
    auto rand_d =
        std::bind<FloatRndFunc>(Random::Uniform, min_disparity, max_disparity);
    auto rand_n = std::bind<FloatRndFunc>(Random::Uniform, -1.f, 1.f);

    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            const int32_t p = y * width + x;
            float disp = rand_d();
            if (is_integer_disp) {
                disp = std::round(disp);
            }
            disparity[p] = disp;

            Vector3f norm;
            if (!is_force_fpw) {
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

void PatchMatchStereo::ComputeGradient(const uint8_t *gray, Gradient *grad,
                                       int32_t width, int32_t height) {
    // Sobel????????????
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
               m_left_disparity.data(), m_width, m_height,
               m_option.min_disparity, m_option.max_disparity,
               m_option.is_integer_disp, m_option.is_force_fpw);
    pool.Queue(PatchMatchStereo::RandomInit, m_right_plane.data(),
               m_right_disparity.data(), m_width, m_height,
               -m_option.max_disparity, m_option.min_disparity,
               m_option.is_integer_disp, m_option.is_force_fpw);

    pool.Queue(ComputeGray, m_left_img, m_left_gray.data(), m_width, m_height);
    pool.Queue(ComputeGray, m_right_img, m_right_gray.data(), m_width,
               m_height);

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
                // ??????????????????
                auto &disp = disp_left[y * m_width + x];

                if (disp == INVALID_FLOAT) {
                    mismatches.emplace_back(x, y);
                    continue;
                }

                // ??????????????????????????????????????????????????????
                const auto x_right = lround(x - disp);

                if (x_right >= 0 && x_right < m_width) {
                    // ????????????????????????????????????
                    auto &disp_r = disp_right[y * m_width + x_right];

                    // ??????????????????????????????????????????????????????????????????
                    // ??????????????????????????????????????????????????????
                    if (std::abs(disp + disp_r) > m_option.lrcheck_thresh) {
                        // ??????????????????
                        disp = INVALID_FLOAT;
                        mismatches.emplace_back(x, y);
                    }
                } else {
                    // ???????????????????????????????????????????????????????????????????????????
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
        // ????????????????????????????????????
        std::vector<float> fill_disps(mismatches.size());
        for (size_t n = 0; n < mismatches.size(); ++n) {
            const auto x = mismatches[n].x;
            const auto y = mismatches[n].y;
            std::vector<Vector2i> candidates;

            // ?????????????????????????????????????????????????????????
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
                // ?????????????????????
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
    pool.Queue(&WeightMedianFilter, m_left_img, m_width, m_height,
               m_left_mismatches, m_left_disparity.data(), m_option.patch_size,
               m_option.gamma);
    right_ret.get();
    pool.Queue(&WeightMedianFilter, m_right_img, m_width, m_height,
               m_right_mismatches, m_right_disparity.data(),
               m_option.patch_size, m_option.gamma);
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
