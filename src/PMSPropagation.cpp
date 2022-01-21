#include "PMSPropagation.hpp"

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
      m_right_cost(right_cost) {
    ComputeCostData();
}

void PMSPropagation::ComputeCostData() {
    for (int32_t y = 0; y < m_height; y++) {
        for (int32_t x = 0; x < m_width; x++) {
            const auto &plane_p = m_left_plane[y * m_width + x];
            m_left_cost[y * m_width + x] =
                m_left_cost_computer.ComputeAggregation(x, y, plane_p);
        }
    }
}

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
                cost += 12.f;  // PUNISH
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
