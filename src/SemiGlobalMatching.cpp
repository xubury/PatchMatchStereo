#include "SemiGlobalMatching.hpp"
#include "Utils.hpp"

#include <cstring>
#include <iostream>

SemiGlobalMatching::SemiGlobalMatching()
    : m_width(0), m_height(0), m_is_initialized(false) {}

SemiGlobalMatching::~SemiGlobalMatching() {
    OutputDebugImg(m_width, m_height, 1, m_left_disparity.data(),
                   m_option.min_disparity, m_option.max_disparity,
                   "sgm_left_disp");
    OutputDebugImg(m_width, m_height, 1, m_right_disparity.data(),
                   -m_option.max_disparity, -m_option.min_disparity,
                   "sgm_right_disp");
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
    m_right_disparity.resize(img_size);

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

    ComputeCost(m_cost.data(), m_left_census.data(), m_right_census.data(),
                m_width, m_height, m_option.min_disparity,
                m_option.max_disparity);
    CostAggregation(m_left_gray.data(), m_width, m_height, m_cost.data(),
                    m_option.min_disparity, m_option.max_disparity, m_option.p1,
                    m_option.p2, m_cost_aggr.data());
    ComputeDisparity(m_left_disparity.data(), m_width, m_height,
                     m_cost_aggr.data(), m_option.min_disparity,
                     m_option.max_disparity);

    ComputeCost(m_cost.data(), m_right_census.data(), m_left_census.data(),
                m_width, m_height, -m_option.max_disparity,
                -m_option.min_disparity);
    CostAggregation(m_right_gray.data(), m_width, m_height, m_cost.data(),
                    -m_option.max_disparity, -m_option.min_disparity,
                    m_option.p1, m_option.p2, m_cost_aggr.data());
    ComputeDisparity(m_right_disparity.data(), m_width, m_height,
                     m_cost_aggr.data(), -m_option.max_disparity,
                     -m_option.min_disparity);

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
    // ???????????????census???
    for (int32_t y = 2; y < height - 2; ++y) {
        for (int32_t x = 2; x < width - 2; ++x) {
            // ???????????????
            const uint8_t gray_center = src[y * width + x];

            // ???????????????5x5???????????????????????????????????????????????????????????????????????????????????????census???
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

            // ???????????????census???
            census[y * width + x] = census_val;
        }
    }
}

void SemiGlobalMatching::ComputeCost(uint8_t *cost_ptr,
                                     const uint32_t *left_census,
                                     const uint32_t *right_census,
                                     int32_t width, int32_t height,
                                     int32_t min_disparity,
                                     int32_t max_disparity) {
    const int32_t disp_range = max_disparity - min_disparity;
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            // ?????????census???
            {
                const int32_t census_val_l = left_census[y * width + x];

                // ????????????????????????
                for (int32_t d = min_disparity; d < max_disparity; d++) {
                    const uint32_t d_idx = d - min_disparity;
                    auto &cost = cost_ptr[y * width * disp_range +
                                          x * disp_range + d_idx];
                    if (x - d < 0 || x - d >= width) {
                        cost = UINT8_MAX / 2;
                        continue;
                    }
                    // ????????????????????????census???
                    const int32_t census_val_r =
                        right_census[y * width + x - d];

                    // ??????????????????
                    cost = Hamming(census_val_l, census_val_r);
                }
            }
        }
    }
}

void SemiGlobalMatching::CostAggregation(const uint8_t *img, int32_t width,
                                         int32_t height, const uint8_t *cost,
                                         int32_t min_disparity,
                                         int32_t max_disparity, uint32_t p1,
                                         uint32_t p2, uint8_t *cost_aggr) {
    const size_t size = width * height * (max_disparity - min_disparity);
    std::vector<uint8_t> cost_left(size);
    std::vector<uint8_t> cost_right(size);
    std::vector<uint8_t> cost_top(size);
    std::vector<uint8_t> cost_bottom(size);
    std::vector<uint8_t> cost_tl(size);
    std::vector<uint8_t> cost_br(size);
    std::vector<uint8_t> cost_tr(size);
    std::vector<uint8_t> cost_bl(size);
    {
        ThreadPool pool(std::thread::hardware_concurrency());
        pool.Queue(CostAggregationLeft, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_left.data(), true);
        pool.Queue(CostAggregationLeft, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_right.data(), false);

        pool.Queue(CostAggregationTop, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_top.data(), true);
        pool.Queue(CostAggregationTop, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_bottom.data(), false);

        pool.Queue(CostAggregationTopLeft, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_tl.data(), true);
        pool.Queue(CostAggregationTopLeft, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_br.data(), false);

        pool.Queue(CostAggregationTopRight, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_tr.data(), true);
        pool.Queue(CostAggregationTopRight, img, width, height, min_disparity,
                   max_disparity, p1, p2, cost, cost_bl.data(), false);
    }
    for (size_t i = 0; i < size; ++i) {
        cost_aggr[i] =
            (cost_left[i] + cost_right[i] + cost_top[i] + cost_bottom[i] +
             cost_tl[i] + cost_br[i] + cost_tr[i] + cost_bl[i]) /
            8;
    }
}

void SemiGlobalMatching::CostAggregationLeft(
    const uint8_t *img, int32_t width, int32_t height, int32_t min_disparity,
    int32_t max_disparity, uint32_t p1, uint32_t p2, const uint8_t *cost,
    uint8_t *cost_aggr, bool is_forward) {
    const int32_t disp_range = max_disparity - min_disparity;
    const int32_t direction = is_forward ? 1 : -1;
    for (int32_t y = 0; y < height; ++y) {
        auto cost_row = is_forward
                            ? cost + y * width * disp_range
                            : cost + (y * width + width - 1) * disp_range;
        auto cost_aggr_row =
            is_forward ? cost_aggr + y * width * disp_range
                       : cost_aggr + (y * width + width - 1) * disp_range;

        auto img_row =
            is_forward ? img + y * width : img + y * width + width - 1;

        uint8_t gray = *img_row;
        uint8_t gray_last = *img_row;

        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        memcpy(cost_aggr_row, cost_row,
               static_cast<uint32_t>(disp_range * sizeof(uint8_t)));
        memcpy(&cost_last_path[1], cost_aggr_row,
               static_cast<uint16_t>(disp_range * sizeof(uint8_t)));
        cost_row += direction * disp_range;
        cost_aggr_row += direction * disp_range;
        img_row += direction;

        uint8_t min_cost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            min_cost_last_path = std::min(cost, min_cost_last_path);
        }

        for (int32_t x = 0; x < width - 1; ++x) {
            gray = *img_row;
            uint8_t min_cost = UINT8_MAX;
            for (int32_t d = 0; d < disp_range; ++d) {
                const uint8_t cost = cost_row[d];
                const uint32_t l1 = cost_last_path[d + 1];
                const uint32_t l2 = cost_last_path[d] + p1;
                const uint32_t l3 = cost_last_path[d + 2] + p1;
                const uint32_t l4 =
                    min_cost_last_path +
                    std::max(p1, p2 / (abs(gray - gray_last) + 1));
                const uint8_t cost_s =
                    cost + std::min(std::min(l1, l2), std::min(l3, l4)) -
                    min_cost_last_path;
                cost_aggr_row[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }

            min_cost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_row,
                   disp_range * sizeof(uint8_t));
            cost_row += direction * disp_range;
            cost_aggr_row += direction * disp_range;
            img_row += direction;

            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::CostAggregationTop(
    const uint8_t *img, int32_t width, int32_t height, int32_t min_disparity,
    int32_t max_disparity, int32_t p1, int32_t p2, const uint8_t *cost,
    uint8_t *cost_aggr, bool is_forward) {
    const int32_t disp_range = max_disparity - min_disparity;
    const int32_t direction = is_forward ? 1 : -1;
    for (int32_t x = 0; x < width; ++x) {
        auto cost_col = is_forward
                            ? cost + x * disp_range
                            : cost + ((height - 1) * width + x) * disp_range;
        auto cost_aggr_col =
            is_forward ? cost_aggr + x * disp_range
                       : cost_aggr + ((height - 1) * width + x) * disp_range;

        auto img_col = is_forward ? img + x : img + x + (height - 1) * width;

        uint8_t gray = *img_col;
        uint8_t gray_last = *img_col;

        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        memcpy(cost_aggr_col, cost_col,
               static_cast<uint32_t>(disp_range * sizeof(uint8_t)));
        memcpy(&cost_last_path[1], cost_aggr_col,
               static_cast<uint16_t>(disp_range * sizeof(uint8_t)));
        cost_col += direction * width * disp_range;
        cost_aggr_col += direction * width * disp_range;
        img_col += direction * width;

        uint8_t min_cost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            min_cost_last_path = std::min(cost, min_cost_last_path);
        }

        for (int32_t y = 0; y < height - 1; ++y) {
            gray = *img_col;
            uint8_t min_cost = UINT8_MAX;
            for (int32_t d = 0; d < disp_range; ++d) {
                const uint8_t cost = cost_col[d];
                const uint32_t l1 = cost_last_path[d + 1];
                const uint32_t l2 = cost_last_path[d] + p1;
                const uint32_t l3 = cost_last_path[d + 2] + p1;
                const uint32_t l4 =
                    min_cost_last_path +
                    std::max(p1, p2 / (abs(gray - gray_last) + 1));
                const uint8_t cost_s =
                    cost + std::min(std::min(l1, l2), std::min(l3, l4)) -
                    min_cost_last_path;
                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }

            min_cost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col,
                   disp_range * sizeof(uint8_t));
            cost_col += direction * width * disp_range;
            cost_aggr_col += direction * width * disp_range;
            img_col += direction * width;

            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::CostAggregationTopLeft(
    const uint8_t *img_data, int32_t width, int32_t height,
    int32_t min_disparity, int32_t max_disparity, int32_t p1, int32_t p2,
    const uint8_t *cost, uint8_t *cost_aggr, bool is_forward) {
    // ????????????
    const int32_t disp_range = max_disparity - min_disparity;

    // ??????(??????->??????) ???is_forward = true ; direction = 1
    // ??????(??????->??????) ???is_forward = false; direction = -1;
    const int32_t direction = is_forward ? 1 : -1;

    // ??????

    // ?????????????????????????????????????????????????????????
    int32_t current_row = 0;
    int32_t current_col = 0;

    for (int32_t j = 0; j < width; j++) {
        // ???????????????????????????(???,dir=-1)?????????
        auto cost_init_col =
            (is_forward)
                ? (cost + j * disp_range)
                : (cost + (height - 1) * width * disp_range + j * disp_range);
        auto cost_aggr_col =
            (is_forward) ? (cost_aggr + j * disp_range)
                         : (cost_aggr + (height - 1) * width * disp_range +
                            j * disp_range);
        auto img_col = (is_forward) ? (img_data + j)
                                    : (img_data + (height - 1) * width + j);

        // ?????????????????????????????????????????????????????????????????????????????????????????????????????????
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        // ??????????????????????????????????????????????????????????????????
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

        // ?????????????????????????????????????????????
        uint8_t gray = *img_col;
        uint8_t gray_last = *img_col;

        // ???????????????????????????????????????????????????width+1?????????
        // ??????????????????????????????
        // ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        current_row = is_forward ? 0 : height - 1;
        current_col = j;
        if (is_forward && current_col == width - 1 &&
            current_row < height - 1) {
            // ??????->?????????????????????
            cost_init_col =
                cost + (current_row + direction) * width * disp_range;
            cost_aggr_col =
                cost_aggr + (current_row + direction) * width * disp_range;
            img_col = img_data + (current_row + direction) * width;
            current_col = 0;
        } else if (!is_forward && current_col == 0 && current_row > 0) {
            // ??????->?????????????????????
            cost_init_col = cost +
                            (current_row + direction) * width * disp_range +
                            (width - 1) * disp_range;
            cost_aggr_col = cost_aggr +
                            (current_row + direction) * width * disp_range +
                            (width - 1) * disp_range;
            img_col =
                img_data + (current_row + direction) * width + (width - 1);
            current_col = width - 1;
        } else {
            cost_init_col += direction * (width + 1) * disp_range;
            cost_aggr_col += direction * (width + 1) * disp_range;
            img_col += direction * (width + 1);
        }

        // ???????????????????????????????????????
        uint8_t mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // ???????????????2??????????????????????????????
        for (int32_t i = 0; i < height - 1; i++) {
            gray = *img_col;
            uint8_t min_cost = UINT8_MAX;
            for (int32_t d = 0; d < disp_range; d++) {
                // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1,
                // Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
                const uint8_t cost = cost_init_col[d];
                const uint16_t l1 = cost_last_path[d + 1];
                const uint16_t l2 = cost_last_path[d] + p1;
                const uint16_t l3 = cost_last_path[d + 2] + p1;
                const uint16_t l4 =
                    mincost_last_path +
                    std::max(p1, p2 / (abs(gray - gray_last) + 1));

                const uint8_t cost_s =
                    cost + static_cast<uint8_t>(
                               std::min(std::min(l1, l2), std::min(l3, l4)) -
                               mincost_last_path);

                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }

            // ???????????????????????????????????????????????????
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col,
                   disp_range * sizeof(uint8_t));

            // ????????????????????????
            current_row += direction;
            current_col += direction;

            // ???????????????,??????????????????????????????
            // ??????????????????????????????
            // ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            if (is_forward && current_col == width - 1 &&
                current_row < height - 1) {
                // ??????->?????????????????????
                cost_init_col =
                    cost + (current_row + direction) * width * disp_range;
                cost_aggr_col =
                    cost_aggr + (current_row + direction) * width * disp_range;
                img_col = img_data + (current_row + direction) * width;
                current_col = 0;
            } else if (!is_forward && current_col == 0 && current_row > 0) {
                // ??????->?????????????????????
                cost_init_col = cost +
                                (current_row + direction) * width * disp_range +
                                (width - 1) * disp_range;
                cost_aggr_col = cost_aggr +
                                (current_row + direction) * width * disp_range +
                                (width - 1) * disp_range;
                img_col =
                    img_data + (current_row + direction) * width + (width - 1);
                current_col = width - 1;
            } else {
                cost_init_col += direction * (width + 1) * disp_range;
                cost_aggr_col += direction * (width + 1) * disp_range;
                img_col += direction * (width + 1);
            }

            // ?????????????????????
            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::CostAggregationTopRight(
    const uint8_t *img_data, int32_t width, int32_t height,
    int32_t min_disparity, int32_t max_disparity, int32_t p1, int32_t p2,
    const uint8_t *cost, uint8_t *cost_aggr, bool is_forward) {
    // ????????????
    const int32_t disp_range = max_disparity - min_disparity;

    // ??????(??????->??????) ???is_forward = true ; direction = 1
    // ??????(??????->??????) ???is_forward = false; direction = -1;
    const int32_t direction = is_forward ? 1 : -1;

    // ??????

    // ?????????????????????????????????????????????????????????
    int32_t current_row = 0;
    int32_t current_col = 0;

    for (int32_t j = 0; j < width; j++) {
        // ???????????????????????????(???,dir=-1)?????????
        auto cost_init_col =
            (is_forward)
                ? (cost + j * disp_range)
                : (cost + (height - 1) * width * disp_range + j * disp_range);
        auto cost_aggr_col =
            (is_forward) ? (cost_aggr + j * disp_range)
                         : (cost_aggr + (height - 1) * width * disp_range +
                            j * disp_range);
        auto img_col = (is_forward) ? (img_data + j)
                                    : (img_data + (height - 1) * width + j);

        // ?????????????????????????????????????????????????????????????????????????????????????????????????????????
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        // ??????????????????????????????????????????????????????????????????
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

        // ?????????????????????????????????????????????
        uint8_t gray = *img_col;
        uint8_t gray_last = *img_col;

        // ???????????????????????????????????????????????????width-1?????????
        // ??????????????????????????????
        // ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        current_row = is_forward ? 0 : height - 1;
        current_col = j;
        if (is_forward && current_col == 0 && current_row < height - 1) {
            // ??????->?????????????????????
            cost_init_col = cost +
                            (current_row + direction) * width * disp_range +
                            (width - 1) * disp_range;
            cost_aggr_col = cost_aggr +
                            (current_row + direction) * width * disp_range +
                            (width - 1) * disp_range;
            img_col =
                img_data + (current_row + direction) * width + (width - 1);
            current_col = width - 1;
        } else if (!is_forward && current_col == width - 1 && current_row > 0) {
            // ??????->?????????????????????
            cost_init_col =
                cost + (current_row + direction) * width * disp_range;
            cost_aggr_col =
                cost_aggr + (current_row + direction) * width * disp_range;
            img_col = img_data + (current_row + direction) * width;
            current_col = 0;
        } else {
            cost_init_col += direction * (width - 1) * disp_range;
            cost_aggr_col += direction * (width - 1) * disp_range;
            img_col += direction * (width - 1);
        }

        // ???????????????????????????????????????
        uint8_t mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // ???????????????2??????????????????????????????
        for (int32_t i = 0; i < height - 1; i++) {
            gray = *img_col;
            uint8_t min_cost = UINT8_MAX;
            for (int32_t d = 0; d < disp_range; d++) {
                // Lr(p,d) = C(p,d) + min( Lr(p-r,d), Lr(p-r,d-1) + P1,
                // Lr(p-r,d+1) + P1, min(Lr(p-r))+P2 ) - min(Lr(p-r))
                const uint8_t cost = cost_init_col[d];
                const uint16_t l1 = cost_last_path[d + 1];
                const uint16_t l2 = cost_last_path[d] + p1;
                const uint16_t l3 = cost_last_path[d + 2] + p1;
                const uint16_t l4 =
                    mincost_last_path +
                    std::max(p1, p2 / (abs(gray - gray_last) + 1));

                const uint8_t cost_s =
                    cost + static_cast<uint8_t>(
                               std::min(std::min(l1, l2), std::min(l3, l4)) -
                               mincost_last_path);

                cost_aggr_col[d] = cost_s;
                min_cost = std::min(min_cost, cost_s);
            }

            // ???????????????????????????????????????????????????
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col,
                   disp_range * sizeof(uint8_t));

            // ????????????????????????
            current_row += direction;
            current_col -= direction;

            // ???????????????,??????????????????????????????
            // ??????????????????????????????
            // ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
            if (is_forward && current_col == 0 && current_row < height - 1) {
                // ??????->?????????????????????
                cost_init_col = cost +
                                (current_row + direction) * width * disp_range +
                                (width - 1) * disp_range;
                cost_aggr_col = cost_aggr +
                                (current_row + direction) * width * disp_range +
                                (width - 1) * disp_range;
                img_col =
                    img_data + (current_row + direction) * width + (width - 1);
                current_col = width - 1;
            } else if (!is_forward && current_col == width - 1 &&
                       current_row > 0) {
                // ??????->?????????????????????
                cost_init_col =
                    cost + (current_row + direction) * width * disp_range;
                cost_aggr_col =
                    cost_aggr + (current_row + direction) * width * disp_range;
                img_col = img_data + (current_row + direction) * width;
                current_col = 0;
            } else {
                cost_init_col += direction * (width - 1) * disp_range;
                cost_aggr_col += direction * (width - 1) * disp_range;
                img_col += direction * (width - 1);
            }

            // ?????????????????????
            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::ComputeDisparity(float *disparity, int32_t width,
                                          int32_t height,
                                          const uint8_t *cost_ptr,
                                          float min_disparity,
                                          float max_disparity) {
    const int32_t disp_range = max_disparity - min_disparity;
    std::vector<uint16_t> cost_local(disp_range);

    // ???????????????????????????
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            uint16_t min_cost = UINT16_MAX;
            int32_t best_disparity = 0;

            // ????????????????????????????????????????????????????????????????????????????????????
            for (int32_t d = min_disparity; d < max_disparity; ++d) {
                const int32_t d_idx = d - min_disparity;
                const auto &cost = cost_local[d_idx] =
                    cost_ptr[y * width * disp_range + x * disp_range + d_idx];
                if (min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
            }
            // ---???????????????
            if (best_disparity == min_disparity ||
                best_disparity == max_disparity - 1) {
                disparity[y * width + x] = INVALID_FLOAT;
                continue;
            }
            // ???????????????????????????????????????cost_1??????????????????????????????cost_2
            const int32_t idx_1 = best_disparity - 1 - min_disparity;
            const int32_t idx_2 = best_disparity + 1 - min_disparity;
            const uint16_t cost_1 = cost_local[idx_1];
            const uint16_t cost_2 = cost_local[idx_2];
            // ???????????????????????????
            const uint16_t denom = std::max(1, cost_1 + cost_2 - 2 * min_cost);
            disparity[y * width + x] =
                static_cast<float>(best_disparity) +
                static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
        }
    }
}

void SemiGlobalMatching::LRCheck() {
    for (int k = 0; k < 2; ++k) {
        auto *disp_left =
            k == 0 ? m_left_disparity.data() : m_right_disparity.data();
        auto *disp_right =
            k == 0 ? m_right_disparity.data() : m_left_disparity.data();
        auto &mismatches = k == 0 ? m_left_mismatches : m_right_mismatches;
        auto &occlusions = k == 0 ? m_left_occlusion : m_right_occlusion;

        mismatches.clear();
        occlusions.clear();

        // ---?????????????????????
        for (int32_t y = 0; y < m_height; y++) {
            for (int32_t x = 0; x < m_width; x++) {
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
                    const auto &disp_r = disp_right[y * m_width + x_right];

                    // ?????????????????????????????????????????????????????????
                    if (std::abs(disp + disp_r) > m_option.lrcheck_thresh) {
                        // ??????????????????????????????
                        // ????????????????????????????????????????????????????????????????????????disp_rl
                        // if(disp_rl > disp)
                        //		pixel in occlusions
                        // else
                        //		pixel in mismatches
                        const int32_t x_rl = lround(x_right + disp_r);
                        if (x_rl >= 0 && x_rl < m_width) {
                            const auto &disp_l = disp_left[y * m_width + x_rl];
                            if (disp_l > disp) {
                                occlusions.emplace_back(x, y);
                            } else {
                                mismatches.emplace_back(x, y);
                            }
                        } else {
                            mismatches.emplace_back(x, y);
                        }
                        // ??????????????????
                        disp = INVALID_FLOAT;
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

void SemiGlobalMatching::FillHole() {
    const auto task = [](float *disparity, int32_t width, int32_t height,
                         std::vector<Vector2i> &mismatches,
                         std::vector<Vector2i> &occlusions, float min_disparity,
                         float max_disparity) {
        std::vector<float> disp_collects;
        // ??????8?????????
        const float pi = 3.1415926f;
        float angle1[8] = {pi, 3 * pi / 4, pi / 2,     pi / 4,
                           0,  7 * pi / 4, 3 * pi / 2, 5 * pi / 4};
        float angle2[8] = {pi, 5 * pi / 4, 3 * pi / 2, 7 * pi / 4,
                           0,  pi / 4,     pi / 2,     3 * pi / 4};
        float *angle = angle1;
        // ??????????????????????????????????????????????????????
        const int32_t max_search_length =
            1.0 * std::max(abs(max_disparity), abs(min_disparity));

        for (int32_t k = 0; k < 2; k++) {
            // ??????????????????????????????????????????????????????????????????
            auto &trg_pixels = (k == 0) ? occlusions : mismatches;
            if (trg_pixels.empty()) {
                continue;
            }
            std::vector<float> fill_disps(trg_pixels.size());
            std::vector<Vector2i> inv_pixels;
            if (k == 2) {
                //  ?????????????????????????????????????????????????????????
                for (int32_t i = 0; i < height; i++) {
                    for (int32_t j = 0; j < width; j++) {
                        if (disparity[i * width + j] == INVALID_FLOAT) {
                            inv_pixels.emplace_back(i, j);
                        }
                    }
                }
                trg_pixels = inv_pixels;
            }

            // ?????????????????????
            for (auto n = 0u; n < trg_pixels.size(); n++) {
                auto &pix = trg_pixels[n];
                const int32_t y = pix.y;
                const int32_t x = pix.x;

                if (y == height / 2) {
                    angle = angle2;
                }

                // ??????8??????????????????????????????????????????
                disp_collects.clear();
                for (int32_t s = 0; s < 8; s++) {
                    const float ang = angle[s];
                    const float sina = float(sin(ang));
                    const float cosa = float(cos(ang));
                    for (int32_t m = 1; m < max_search_length; m++) {
                        const int32_t yy = lround(y + m * sina);
                        const int32_t xx = lround(x + m * cosa);
                        if (yy < 0 || yy >= height || xx < 0 || xx >= width) {
                            break;
                        }
                        const auto &disp = *(disparity + yy * width + xx);
                        if (disp != INVALID_FLOAT) {
                            disp_collects.push_back(disp);
                            break;
                        }
                    }
                }
                if (disp_collects.empty()) {
                    continue;
                }

                std::sort(disp_collects.begin(), disp_collects.end());

                // ???????????????????????????????????????????????????
                // ???????????????????????????????????????
                if (k == 0) {
                    if (disp_collects.size() > 1) {
                        fill_disps[n] = disp_collects[1];
                    } else {
                        fill_disps[n] = disp_collects[0];
                    }
                } else {
                    fill_disps[n] = disp_collects[disp_collects.size() / 2];
                }
            }
            for (auto n = 0u; n < trg_pixels.size(); n++) {
                auto &pix = trg_pixels[n];
                const int32_t y = pix.y;
                const int32_t x = pix.x;
                disparity[y * width + x] = fill_disps[n];
            }
        }
    };
    ThreadPool pool(std::thread::hardware_concurrency());
    auto left_ret = pool.Queue(task, m_left_disparity.data(), m_width, m_height,
                               m_left_mismatches, m_left_occlusion,
                               m_option.min_disparity, m_option.max_disparity);
    auto right_ret = pool.Queue(
        task, m_right_disparity.data(), m_width, m_height, m_right_mismatches,
        m_right_occlusion, -m_option.max_disparity, -m_option.min_disparity);

    left_ret.get();
    pool.Queue(&WeightMedianFilter, m_left_img, m_width, m_height,
               m_left_mismatches, m_left_disparity.data(), 35.f, 10.f);
    right_ret.get();
    pool.Queue(&WeightMedianFilter, m_right_img, m_width, m_height,
               m_right_mismatches, m_right_disparity.data(), 35.f, 10.f);
}
