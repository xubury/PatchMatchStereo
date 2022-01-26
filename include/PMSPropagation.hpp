#ifndef PMS_PROPAGATION_HPP
#define PMS_PROPAGATION_HPP

#include "PatchMatchStereo.hpp"

class CostComputerPMS {
   public:
    CostComputerPMS(const uint8_t* left_img, const uint8_t* right_img,
                    const PatchMatchStereo::Gradient* left_grad,
                    const PatchMatchStereo::Gradient* right_grad, int32_t width,
                    int32_t height, int32_t patch_size, int32_t min_disp,
                    int32_t max_disp, float alpha, float gamma, float tau_col,
                    float tau_grad);

    float Compute(int32_t x, int32_t y, float d) const;

    float Compute(const Color& color,
                  const PatchMatchStereo::Gradient& gradient, int32_t x,
                  int32_t y, float d) const;

    float ComputeAggregation(int32_t x, int32_t y,
                             const DisparityPlane& p) const;

    Vector2i GetGradient(const PatchMatchStereo::Gradient* data, int32_t x,
                         int32_t y) const {
        return data[y * m_width + x];
    }

    Vector2f GetGradient(const PatchMatchStereo::Gradient* grad_data, float x,
                         int32_t y) const {
        const int32_t x1 = std::floor(x);
        const int32_t x2 = std::ceil(x);
        const float ofs = x - x1;

        const auto& g1 = grad_data[y * m_width + x1];
        const auto& g2 = (x2 < m_width) ? grad_data[y * m_width + x2] : g1;

        return {(1 - ofs) * g1.x + ofs * g2.x, (1 - ofs) * g1.y + ofs * g2.y};
    }

    Color GetColor(const uint8_t* data, int32_t x, int32_t y) const {
        const int channels = 3;
        auto* pixel = data + y * m_width * channels + channels * x;
        return {pixel[0], pixel[1], pixel[2]};
    }

    Vector3f GetColor(const uint8_t* data, float x, int32_t y) const {
        Vector3f color;
        const int channels = 3;
        const int32_t x1 = std::floor(x);
        const int32_t x2 = std::ceil(x);
        const float offset = x - x1;
        for (int32_t i = 0; i < 3; ++i) {
            const float g1 = data[y * m_width * channels + channels * x1 + i];
            const float g2 =
                x2 < m_width ? data[y * m_width * channels + channels * x2 + i]
                             : g1;
            color[i] = (1 - offset) * g1 + offset * g2;
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
