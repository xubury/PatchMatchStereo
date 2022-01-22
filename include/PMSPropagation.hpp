#ifndef PMS_PROPAGATION_HPP
#define PMS_PROPAGATION_HPP

#include "PatchMatchStereo.hpp"

class CostComputerPMS {
   public:
    CostComputerPMS(const uint8_t* left_img, const uint8_t* right_img,
                    const PatchMatchStereo::Gradient* left_grad,
                    const PatchMatchStereo::Gradient* right_grad, int32_t width,
                    int32_t height, const PatchMatchStereo::Option& option);

    float Compute(int32_t x, int32_t y, float d) const;

    float Compute(const Color& color,
                  const PatchMatchStereo::Gradient& gradient, int32_t x,
                  int32_t y, float d) const;

    float ComputeAggregation(int32_t x, int32_t y,
                             const DisparityPlane& p) const;

    PatchMatchStereo::Gradient GetGradient(
        const PatchMatchStereo::Gradient* data, int32_t x, int32_t y) const {
        return data[y * m_width + x];
    }
    Color GetColor(const uint8_t* data, float x, int32_t y) const {
        Color color;
        int32_t x1 = std::floor(x);
        int32_t x2 = std::ceil(x);
        float offset = x - x1;
        for (int32_t i = 0; i < 3; ++i) {
            const auto g1 = data[y * m_width * 3 + 3 * x1 + i];
            const auto g2 =
                (x2 < m_width) ? data[y * m_width * 3 + 3 * x2 + i] : g1;
            color[i] = std::round((1 - offset) * g1 + offset * g2);
        }
        return color;
    }

   private:
    const uint8_t* m_left_img;
    const uint8_t* m_right_img;

    const PatchMatchStereo::Gradient* m_left_grad;
    const PatchMatchStereo::Gradient* m_right_grad;

    int32_t m_width;
    int32_t m_height;

    int32_t m_patch_size;

    int32_t m_min_disp;
    int32_t m_max_disp;

    float m_alpha;
    float m_gamma;
    float m_tau_col;
    float m_tau_grad;
};

class PMSPropagation {
   public:
    PMSPropagation(const uint8_t* left_img, const uint8_t* right_img,
                   const PatchMatchStereo::Gradient* left_grad,
                   const PatchMatchStereo::Gradient* right_grad, int32_t width,
                   int32_t height, const PatchMatchStereo::Option& option,
                   DisparityPlane* left_plane, DisparityPlane* right_plane,
                   float* left_cost, float* right_cost);
    ~PMSPropagation() = default;

    void DoPropagation();

   private:
    void ComputeCostData();

    /**
     * \brief 空间传播
     * \param x 像素x坐标
     * \param y 像素y坐标
     * \param direction 传播方向
     */
    void SpatialPropagation(int32_t x, int32_t y, int32_t direction);

    /**
     * \brief 视图传播
     * \param x 像素x坐标
     * \param y 像素y坐标
     */
    void ViewPropagation(int32_t x, int32_t y);

    /**
     * \brief 平面优化
     * \param x 像素x坐标
     * \param y 像素y坐标
     */
    void PlaneRefine(int32_t x, int32_t y);

    CostComputerPMS m_left_cost_computer;
    CostComputerPMS m_right_cost_computer;

    const uint8_t* m_left_img;
    const uint8_t* m_right_img;

    const PatchMatchStereo::Gradient* m_left_grad;
    const PatchMatchStereo::Gradient* m_right_grad;

    int32_t m_width;
    int32_t m_height;

    const PatchMatchStereo::Option& m_option;

    DisparityPlane* m_left_plane;
    DisparityPlane* m_right_plane;

    float* m_left_cost;
    float* m_right_cost;

    int32_t m_num_iter;
};

#endif
