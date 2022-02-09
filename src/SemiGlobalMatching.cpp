#include "SemiGlobalMatching.hpp"
#include "Utils.hpp"

#include <cstring>
#include <iostream>

SemiGlobalMatching::SemiGlobalMatching()
    : m_width(0), m_height(0), m_is_initialized(false) {}

SemiGlobalMatching::~SemiGlobalMatching() {
    OutputDebugImg(m_width, m_height, 1, m_left_disparity.data(),
                   m_option.min_disparity, m_option.max_disparity, "sgm_disp");
}

bool SemiGlobalMatching::Init(int32_t width, int32_t height,
                              const Option &option) {
    m_width = width;
    m_height = height;
    m_option = option;

    std::cout << "initializing semi-global matching..." << std::endl;

    if (m_width == 0 || m_height == 0) {
        std::cout << "invalid image size!" << std::endl;
        return false;
    }
    const int32_t disp_range = option.max_disparity - option.min_disparity;
    if (disp_range <= 0) {
        std::cout << "invalid disparity range!" << std::endl;
        return false;
    }

    const std::size_t img_size = m_width * m_height;
    m_left_gray.resize(img_size);
    m_right_gray.resize(img_size);

    m_left_census.resize(img_size);
    m_right_census.resize(img_size);

    m_cost.resize(img_size * disp_range);
    m_cost_aggr.resize(img_size * disp_range);

    m_left_disparity.resize(img_size);

    m_is_initialized = true;
    return true;
}

bool SemiGlobalMatching::Match(const uint8_t *left_img,
                               const uint8_t *right_img,
                               float *left_disparity) {
    if (!m_is_initialized) {
        return false;
    }
    if (!left_img || !right_img) {
        return false;
    }
    m_left_img = left_img;
    m_right_img = right_img;

    ComputeGray(m_left_img, m_left_gray.data(), m_width, m_height);
    ComputeGray(m_right_img, m_right_gray.data(), m_width, m_height);
    CensusTransform();
    ComputeCost();
    CostAggregation();
    ComputeDisparity();
    LRCheck();

    if (left_disparity) {
        memcpy(left_disparity, m_left_disparity.data(),
               m_width * m_height *
                   sizeof(decltype(m_left_disparity)::value_type));
    }
    return true;
}

void SemiGlobalMatching::CensusTransform() {
    CensusTransform5x5(m_left_gray.data(), m_left_census.data(), m_width,
                       m_height);
    CensusTransform5x5(m_right_gray.data(), m_right_census.data(), m_width,
                       m_height);
}

void SemiGlobalMatching::CensusTransform5x5(const uint8_t *src,
                                            uint32_t *census, int32_t width,
                                            int32_t height) {
    if (src == nullptr || census == nullptr || width <= 5 || height <= 5) {
        return;
    }
    // 逐像素计算census值
    for (int32_t y = 2; y < height - 2; ++y) {
        for (int32_t x = 2; x < width - 2; ++x) {
            // 中心像素值
            const uint8_t gray_center = src[y * width + x];

            // 遍历大小为5x5的窗口内邻域像素，逐一比较像素值与中心像素值的的大小，计算census值
            uint32_t census_val = 0u;
            for (int32_t r = -2; r <= 2; ++r) {
                for (int32_t c = -2; c <= 2; ++c) {
                    census_val <<= 1;
                    const uint8_t gray = src[(y + r) * width + x + c];
                    if (gray < gray_center) {
                        census_val += 1;
                    }
                }
            }

            // 中心像素的census值
            census[y * width + x] = census_val;
        }
    }
}

void SemiGlobalMatching::ComputeCost() {
    const int32_t disp_range = m_option.max_disparity - m_option.min_disparity;
    for (int32_t y = 0; y < m_height; ++y) {
        for (int32_t x = 0; x < m_width; ++x) {
            // 左影像census值
            const int32_t census_val_l = m_left_census[y * m_width + x];

            // 逐视差计算代价值
            for (int32_t d = m_option.min_disparity; d < m_option.max_disparity;
                 d++) {
                const uint32_t d_idx = d - m_option.min_disparity;
                auto &cost =
                    m_cost[y * m_width * disp_range + x * disp_range + d_idx];
                if (x - d < 0 || x - d >= m_width) {
                    cost = UINT8_MAX / 2;
                    continue;
                }
                // 右影像对应像点的census值
                const int32_t census_val_r =
                    m_right_census[y * m_width + x - d];

                // 计算匹配代价
                cost = Hamming(census_val_l, census_val_r);
            }
        }
    }
}

void SemiGlobalMatching::CostAggregation() {}

void SemiGlobalMatching::ComputeDisparity() {
    const int32_t disp_range = m_option.max_disparity - m_option.min_disparity;
    // 未实现聚合步骤，暂用初始代价值来代替
    auto cost_ptr = m_cost.data();

    // 逐像素计算最优视差
    for (int32_t y = 0; y < m_height; ++y) {
        for (int32_t x = 0; x < m_width; ++x) {
            uint16_t min_cost = UINT16_MAX;
            uint16_t max_cost = 0;
            int32_t best_disparity = 0;

            // 遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
            for (int32_t d = m_option.min_disparity; d < m_option.max_disparity;
                 ++d) {
                const int32_t d_idx = d - m_option.min_disparity;
                const auto &cost =
                    cost_ptr[y * m_width * disp_range + x * disp_range + d_idx];
                if (min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
                max_cost = std::max(max_cost, static_cast<uint16_t>(cost));
            }

            // 最小代价值对应的视差值即为像素的最优视差
            if (max_cost != min_cost) {
                m_left_disparity[y * m_width + x] = best_disparity;
            } else {
                // 如果所有视差下的代价值都一样，则该像素无效
                m_left_disparity[y * m_width + x] = INVALID_FLOAT;
            }
        }
    }
}

void SemiGlobalMatching::LRCheck() {}
