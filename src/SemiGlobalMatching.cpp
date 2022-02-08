#include "SemiGlobalMatching.hpp"
#include <cstring>

SemiGlobalMatching::SemiGlobalMatching()
    : m_width(0), m_height(0), m_is_initialized(false) {}

SemiGlobalMatching::~SemiGlobalMatching() {}

bool SemiGlobalMatching::Init(int32_t width, int32_t height,
                              const Option &option) {
    m_width = width;
    m_height = height;
    m_option = option;

    if (m_width == 0 || m_height == 0) {
        return false;
    }
    int32_t disp_range = option.max_disparity - option.min_disparity;
    if (disp_range <= 0) {
        return false;
    }

    std::size_t img_size = m_width * m_height;
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

    CensusTransform();
    ComputeCost();
    CostAggregation();
    ComputeDisparity();
    LRCheck();

    memcpy(left_disparity, m_left_disparity.data(),
           m_width * m_height * sizeof(decltype(m_left_disparity)::value_type));
    return true;
}

void SemiGlobalMatching::CensusTransform() {}

void SemiGlobalMatching::ComputeCost() {}

void SemiGlobalMatching::CostAggregation() {}

void SemiGlobalMatching::ComputeDisparity() {}

void SemiGlobalMatching::LRCheck() {}
