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
    ComputeDisparity(m_left_disparity.data());

    ComputeCost(m_cost.data(), m_right_census.data(), m_left_census.data(),
                m_width, m_height, -m_option.max_disparity,
                -m_option.min_disparity);
    CostAggregation(m_right_gray.data(), m_width, m_height, m_cost.data(),
                    -m_option.max_disparity, -m_option.min_disparity,
                    m_option.p1, m_option.p2, m_cost_aggr.data());
    ComputeDisparity(m_right_disparity.data());

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

void SemiGlobalMatching::ComputeCost(uint8_t *cost_ptr,
                                     const uint32_t *left_census,
                                     const uint32_t *right_census,
                                     int32_t width, int32_t height,
                                     int32_t min_disparity,
                                     int32_t max_disparity) {
    const int32_t disp_range = max_disparity - min_disparity;
    for (int32_t y = 0; y < height; ++y) {
        for (int32_t x = 0; x < width; ++x) {
            // 左影像census值
            {
                const int32_t census_val_l = left_census[y * width + x];

                // 逐视差计算代价值
                for (int32_t d = min_disparity; d < max_disparity; d++) {
                    const uint32_t d_idx = d - min_disparity;
                    auto &cost = cost_ptr[y * width * disp_range +
                                          x * disp_range + d_idx];
                    if (x - d < 0 || x - d >= width) {
                        cost = UINT8_MAX / 2;
                        continue;
                    }
                    // 右影像对应像点的census值
                    const int32_t census_val_r =
                        right_census[y * width + x - d];

                    // 计算匹配代价
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
    // 视差范围
    const int32_t disp_range = max_disparity - min_disparity;

    // 正向(左上->右下) ：is_forward = true ; direction = 1
    // 反向(右下->左上) ：is_forward = false; direction = -1;
    const int32_t direction = is_forward ? 1 : -1;

    // 聚合

    // 存储当前的行列号，判断是否到达影像边界
    int32_t current_row = 0;
    int32_t current_col = 0;

    for (int32_t j = 0; j < width; j++) {
        // 路径头为每一列的首(尾,dir=-1)行像素
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

        // 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        // 初始化：第一个像素的聚合代价值等于初始代价值
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

        // 路径上当前灰度值和上一个灰度值
        uint8_t gray = *img_col;
        uint8_t gray_last = *img_col;

        // 对角线路径上的下一个像素，中间间隔width+1个像素
        // 这里要多一个边界处理
        // 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
        current_row = is_forward ? 0 : height - 1;
        current_col = j;
        if (is_forward && current_col == width - 1 &&
            current_row < height - 1) {
            // 左上->右下，碰右边界
            cost_init_col =
                cost + (current_row + direction) * width * disp_range;
            cost_aggr_col =
                cost_aggr + (current_row + direction) * width * disp_range;
            img_col = img_data + (current_row + direction) * width;
            current_col = 0;
        } else if (!is_forward && current_col == 0 && current_row > 0) {
            // 右下->左上，碰左边界
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

        // 路径上上个像素的最小代价值
        uint8_t mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // 自方向上第2个像素开始按顺序聚合
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

            // 重置上个像素的最小代价值和代价数组
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col,
                   disp_range * sizeof(uint8_t));

            // 当前像素的行列号
            current_row += direction;
            current_col += direction;

            // 下一个像素,这里要多一个边界处理
            // 这里要多一个边界处理
            // 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
            if (is_forward && current_col == width - 1 &&
                current_row < height - 1) {
                // 左上->右下，碰右边界
                cost_init_col =
                    cost + (current_row + direction) * width * disp_range;
                cost_aggr_col =
                    cost_aggr + (current_row + direction) * width * disp_range;
                img_col = img_data + (current_row + direction) * width;
                current_col = 0;
            } else if (!is_forward && current_col == 0 && current_row > 0) {
                // 右下->左上，碰左边界
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

            // 像素值重新赋值
            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::CostAggregationTopRight(
    const uint8_t *img_data, int32_t width, int32_t height,
    int32_t min_disparity, int32_t max_disparity, int32_t p1, int32_t p2,
    const uint8_t *cost, uint8_t *cost_aggr, bool is_forward) {
    // 视差范围
    const int32_t disp_range = max_disparity - min_disparity;

    // 正向(右上->左下) ：is_forward = true ; direction = 1
    // 反向(左下->右上) ：is_forward = false; direction = -1;
    const int32_t direction = is_forward ? 1 : -1;

    // 聚合

    // 存储当前的行列号，判断是否到达影像边界
    int32_t current_row = 0;
    int32_t current_col = 0;

    for (int32_t j = 0; j < width; j++) {
        // 路径头为每一列的首(尾,dir=-1)行像素
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

        // 路径上上个像素的代价数组，多两个元素是为了避免边界溢出（首尾各多一个）
        std::vector<uint8_t> cost_last_path(disp_range + 2, UINT8_MAX);

        // 初始化：第一个像素的聚合代价值等于初始代价值
        memcpy(cost_aggr_col, cost_init_col, disp_range * sizeof(uint8_t));
        memcpy(&cost_last_path[1], cost_aggr_col, disp_range * sizeof(uint8_t));

        // 路径上当前灰度值和上一个灰度值
        uint8_t gray = *img_col;
        uint8_t gray_last = *img_col;

        // 对角线路径上的下一个像素，中间间隔width-1个像素
        // 这里要多一个边界处理
        // 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
        current_row = is_forward ? 0 : height - 1;
        current_col = j;
        if (is_forward && current_col == 0 && current_row < height - 1) {
            // 右上->左下，碰左边界
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
            // 左下->右上，碰右边界
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

        // 路径上上个像素的最小代价值
        uint8_t mincost_last_path = UINT8_MAX;
        for (auto cost : cost_last_path) {
            mincost_last_path = std::min(mincost_last_path, cost);
        }

        // 自路径上第2个像素开始按顺序聚合
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

            // 重置上个像素的最小代价值和代价数组
            mincost_last_path = min_cost;
            memcpy(&cost_last_path[1], cost_aggr_col,
                   disp_range * sizeof(uint8_t));

            // 当前像素的行列号
            current_row += direction;
            current_col -= direction;

            // 下一个像素,这里要多一个边界处理
            // 这里要多一个边界处理
            // 沿对角线前进的时候会碰到影像列边界，策略是行号继续按原方向前进，列号到跳到另一边界
            if (is_forward && current_col == 0 && current_row < height - 1) {
                // 右上->左下，碰左边界
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
                // 左下->右上，碰右边界
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

            // 像素值重新赋值
            gray_last = gray;
        }
    }
}

void SemiGlobalMatching::ComputeDisparity(float *disparity) {
    const int32_t disp_range = m_option.max_disparity - m_option.min_disparity;
    auto cost_ptr = m_cost_aggr.data();

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
                disparity[y * m_width + x] = best_disparity;
            } else {
                // 如果所有视差下的代价值都一样，则该像素无效
                disparity[y * m_width + x] = INVALID_FLOAT;
            }
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

void SemiGlobalMatching::FillHole() {
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

    // left_ret.get();
    // pool.Queue(&PatchMatchStereo::WeightMedianFilter, this, m_left_img,
    //            m_left_mismatches, m_left_disparity.data());
    // right_ret.get();
    // pool.Queue(&PatchMatchStereo::WeightMedianFilter, this, m_right_img,
    //            m_right_mismatches, m_right_disparity.data());
}
