#include "PMSPropagation.hpp"

#include "Utils.hpp"

const float PUNISH_FACTOR = 12.f;

CostComputerPMS::CostComputerPMS(const uint8_t *left_img,
                                 const uint8_t *right_img,
                                 const PatchMatchStereo::Gradient *left_grad,
                                 const PatchMatchStereo::Gradient *right_grad,
                                 int32_t width, int32_t height,
                                 const PatchMatchStereo::Option &option)
    : m_left_img(left_img),
      m_right_img(right_img),
      m_left_grad(left_grad),
      m_right_grad(right_grad),
      m_width(width),
      m_height(height),
      m_patch_size(option.patch_size),
      m_min_disp(option.min_disparity),
      m_max_disp(option.max_disparity),
      m_alpha(option.alpha),
      m_gamma(option.gamma),
      m_tau_col(option.tau_col),
      m_tau_grad(option.tau_grad) {}

float CostComputerPMS::Compute(int32_t x, int32_t y, float d) const {
    // 计算代价值，(1-a)*颜色空间距离+a*梯度空间距离
    float xr = x - d;
    if (xr < 0.0f || xr >= m_width) {
        return (1 - m_alpha) * m_tau_col + m_alpha * m_tau_grad;
    }
    const auto color_left = GetColor(m_left_img, x, y);
    const auto color_right = GetColor(m_right_img, xr, y);

    const auto dist_color =
        std::min<float>(std::abs(color_left.r - color_right.r) +
                            std::abs(color_left.g - color_right.g) +
                            std::abs(color_left.b - color_right.b),
                        m_tau_col);

    const auto grad_left = GetGradient(m_left_grad, x, y);
    const auto grad_right = GetGradient(m_right_grad, x, y);
    const auto dist_grad =
        std::min<float>(std::abs(grad_left.x - grad_right.x) +
                            std::abs(grad_left.y - grad_right.y),
                        m_tau_grad);
    return (1 - m_alpha) * dist_color + m_alpha * dist_grad;
}

float CostComputerPMS::Compute(const Color &color,
                               const PatchMatchStereo::Gradient &gradient,
                               int32_t x, int32_t y, float d) const {
    // 计算代价值，(1-a)*颜色空间距离+a*梯度空间距离
    float xr = x - d;
    if (xr < 0.0f || xr >= m_width) {
        return (1 - m_alpha) * m_tau_col + m_alpha * m_tau_grad;
    }
    const auto color_right = GetColor(m_right_img, xr, y);

    const auto dist_color = std::min<float>(
        std::abs(color.r - color_right.r) + std::abs(color.g - color_right.g) +
            std::abs(color.b - color_right.b),
        m_tau_col);

    const auto grad_right = GetGradient(m_right_grad, x, y);
    const auto dist_grad =
        std::min<float>(std::abs(gradient.x - grad_right.x) +
                            std::abs(gradient.y - grad_right.y),
                        m_tau_grad);
    return (1 - m_alpha) * dist_color + m_alpha * dist_grad;
}

float CostComputerPMS::ComputeAggregation(int32_t x, int32_t y,
                                          const DisparityPlane &p) const {
    const auto pat = m_patch_size / 2;
    const auto &col_p = GetColor(m_left_img, x, y);
    float cost = 0.0f;
    for (int32_t r = -pat; r <= pat; r++) {
        const int32_t yr = y + r;
        for (int32_t c = -pat; c <= pat; c++) {
            const int32_t xc = x + c;
            if (yr < 0 || yr > m_height - 1 || xc < 0 || xc > m_width - 1) {
                continue;
            }
            // 计算视差值
            const float d = p.GetDisparity(xc, yr);
            if (d < m_min_disp || d > m_max_disp) {
                cost += PUNISH_FACTOR;
                continue;
            }

            // 计算权值
            const auto &col_q = GetColor(m_left_img, xc, yr);
            const auto dc = abs(col_p.r - col_q.r) + abs(col_p.g - col_q.g) +
                            abs(col_p.b - col_q.b);
            const auto w = exp(-dc / m_gamma);

            // 聚合代价
            const auto left_grad = GetGradient(m_left_grad, xc, yr);
            cost += w * Compute(col_q, left_grad, xc, yr, d);
        }
    }
    return cost;
}

PMSPropagation::PMSPropagation(
    const uint8_t *left_img, const uint8_t *right_img,
    const PatchMatchStereo::Gradient *left_grad,
    const PatchMatchStereo::Gradient *right_grad, int32_t width, int32_t height,
    const PatchMatchStereo::Option &option, DisparityPlane *left_plane,
    DisparityPlane *right_plane, float *left_cost, float *right_cost)
    : m_left_cost_computer(left_img, right_img, left_grad, right_grad, width,
                           height, option),
      m_right_cost_computer(right_img, left_img, right_grad, left_grad, width,
                            height, option),
      m_left_img(left_img),
      m_right_img(right_img),
      m_left_grad(left_grad),
      m_right_grad(right_grad),
      m_width(width),
      m_height(height),
      m_option(option),
      m_left_plane(left_plane),
      m_right_plane(right_plane),
      m_left_cost(left_cost),
      m_right_cost(right_cost),
      m_num_iter(0) {
    ComputeCostData();
}

void PMSPropagation::DoPropagation() {
    // 偶数次迭代从左上到右下传播
    // 奇数次迭代从右下到左上传播
    const int32_t dir = m_num_iter % 2 == 0 ? 1 : -1;
    int32_t y = (dir == 1) ? 0 : m_height - 1;
    for (int32_t i = 0; i < m_height; ++i, y += dir) {
        int32_t x = (dir == 1) ? 0 : m_width - 1;
        for (int32_t j = 0; j < m_width; ++j, x += dir) {
            SpatialPropagation(x, y, dir);

            if (!m_option.is_force_fpw) {
                PlaneRefine(x, y);
            }
            ViewPropagation(x, y);
        }
    }
    ++m_num_iter;
}

void PMSPropagation::ComputeCostData() {
    for (int32_t y = 0; y < m_height; y++) {
        for (int32_t x = 0; x < m_width; x++) {
            const int32_t p = y * m_width + x;
            const auto &plane_p = m_left_plane[p];
            m_left_cost[p] =
                m_left_cost_computer.ComputeAggregation(x, y, plane_p);
        }
    }
}

void PMSPropagation::SpatialPropagation(int32_t x, int32_t y,
                                        int32_t direction) {
    // ---
    // 空间传播

    // 获取p当前的视差平面并计算代价
    const int32_t p = y * m_width + x;
    auto &plane_p = m_left_plane[p];
    auto &cost_p = m_left_cost[p];

    // 获取p左(右)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
    const int32_t xd = x - direction;
    if (xd >= 0 && xd < m_width) {
        auto &plane = m_left_plane[y * m_width + xd];
        if (plane != plane_p) {
            const auto cost =
                m_left_cost_computer.ComputeAggregation(x, y, plane);
            if (cost < cost_p) {
                plane_p = plane;
                cost_p = cost;
            }
        }
    }

    // 获取p上(下)侧像素的视差平面，计算将平面分配给p时的代价，取较小值
    const int32_t yd = y - direction;
    if (yd >= 0 && yd < m_height) {
        auto &plane = m_left_plane[yd * m_width + x];
        if (plane != plane_p) {
            const auto cost =
                m_left_cost_computer.ComputeAggregation(x, y, plane);
            if (cost < cost_p) {
                plane_p = plane;
                cost_p = cost;
            }
        }
    }
}

void PMSPropagation::PlaneRefine(int32_t x, int32_t y) {
    // --
    // 平面优化

    // 像素p的平面、代价、视差、法线
    const int32_t p = y * m_width + x;
    auto &plane_p = m_left_plane[p];
    auto &cost_p = m_left_cost[p];

    float d_p = plane_p.GetDisparity(x, y);
    Vector3f norm_p = plane_p.GetNormal();

    float disp_update =
        (m_option.max_disparity - m_option.min_disparity) / 2.0f;
    float norm_update = 1.0f;
    const float stop_thres = 0.1f;

    // 迭代优化
    while (disp_update > stop_thres) {
        // 在 -disp_update ~ disp_update 范围内随机一个视差增量
        float disp_rd = Random::Uniform(-disp_update, disp_update);
        if (m_option.is_integer_disp) {
            disp_rd = std::round(disp_rd);
        }

        // 计算像素p新的视差
        const float d_p_new = d_p + disp_rd;
        if (d_p_new < m_option.min_disparity ||
            d_p_new > m_option.max_disparity) {
            disp_update /= 2;
            norm_update /= 2;
            continue;
        }

        // 在 -norm_update ~ norm_update 范围内随机三个值作为法线增量的三个分量
        Vector3f norm_rd;
        if (!m_option.is_force_fpw) {
            norm_rd.x = Random::Uniform(-norm_update, norm_update);
            norm_rd.y = Random::Uniform(-norm_update, norm_update);
            float z = Random::Uniform(-norm_update, norm_update);
            while (z == 0.0f) {
                z = Random::Uniform(-norm_update, norm_update);
            }
            norm_rd.z = z;
        } else {
            norm_rd.x = 0.0f;
            norm_rd.y = 0.0f;
            norm_rd.z = 0.0f;
        }

        // 计算像素p新的法线
        auto norm_p_new = Normalize(norm_p + norm_rd);

        // 计算新的视差平面
        auto plane_new = DisparityPlane(x, y, norm_p_new, d_p_new);

        // 比较Cost
        if (plane_new != plane_p) {
            const float cost =
                m_left_cost_computer.ComputeAggregation(x, y, plane_new);

            if (cost < cost_p) {
                plane_p = plane_new;
                cost_p = cost;
                d_p = d_p_new;
                norm_p = norm_p_new;
            }
        }

        disp_update /= 2.0f;
        norm_update /= 2.0f;
    }
}

void PMSPropagation::ViewPropagation(int32_t x, int32_t y) {
    // 视图传播
    // 搜索p在右视图的同名点q，更新q的平面

    // 左视图匹配点p的位置及其视差平面
    const int32_t p = y * m_width + x;
    const auto &plane_p = m_left_plane[p];

    const float d_p = plane_p.GetDisparity(x, y);

    // 计算右视图列号
    const int32_t xr = std::round(x - d_p);
    if (xr < 0 || xr >= m_width) {
        return;
    }

    const int32_t q = y * m_width + xr;
    auto &plane_q = m_right_plane[q];
    auto &cost_q = m_right_cost[q];

    // 将左视图的视差平面转换到右视图
    const auto plane_p2q = plane_p.GetPairPlane(x, y);
    const auto cost =
        m_right_cost_computer.ComputeAggregation(xr, y, plane_p2q);
    if (cost < cost_q) {
        plane_q = plane_p2q;
        cost_q = cost;
    }
}
