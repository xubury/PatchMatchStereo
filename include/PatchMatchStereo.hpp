#ifndef PATCH_MATCH_STEREO_HPP
#define PATCH_MATCH_STEREO_HPP

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>
#include <vector>

#include "CostComputer.hpp"

struct DisparityPlane {
    glm::vec3 p;
    DisparityPlane() = default;
    DisparityPlane(const glm::vec3& v) { p = v; }
    DisparityPlane(const int32_t x, const int32_t y, const glm::vec3& n,
                   const float d) {
        p.x = -n.x / n.z;
        p.y = -n.y / n.z;
        p.z = (n.x * x + n.y * y + n.z * d) / n.z;
    }

    /**
     * \brief 获取该平面下像素(x,y)的视差
     * \param x		像素x坐标
     * \param y		像素y坐标
     * \return 像素(x,y)的视差
     */
    float GetDisparity(const int32_t x, const int32_t y) const {
        return glm::dot(p, glm::vec3(x, y, 1.0f));
    }

    /** \brief 获取平面的法线 */
    glm::vec3 GetNormal() const {
        glm::vec3 n(p.x, p.y, -1.0f);
        return glm::normalize(n);
    }

    /**
     * \brief 将视差平面转换到另一视图
     * 假设左视图平面方程为 d = a_p*xl + b_p*yl + c_p
     * 左右视图满足：(1) xr = xl - d_p; (2) yr = yl; (3)
     * 视差符号相反(本代码左视差为正值，右视差为负值)
     * 代入左视图视差平面方程就可得到右视图坐标系下的平面方程:
     * d = -a_p*xr - b_p*yr - (c_p+a_p*d_p) 右至左同理
     * \param x	像素x坐标
     * \param y 像素y坐标
     * \return 转换后的平面
     */
    DisparityPlane GetPairPlane(const int32_t x, const int32_t y) const {
        const float d = GetDisparity(x, y);
        return glm::vec3(-p.x, -p.y, -p.z - p.x * d);
    }

    // operator ==
    bool operator==(const DisparityPlane& v) const { return p == v.p; }
    // operator !=
    bool operator!=(const DisparityPlane& v) const { return p != v.p; }
};

class PatchMatchStereo {
   public:
    struct Option {
        int32_t patch_size;
        int32_t min_disparity;
        int32_t max_disparity;

        float gamma;
        float alpha;

        float tau_col;
        float tau_grad;

        int32_t num_iters;

        bool is_check_lr;
        float lrcheck_thresh;

        bool is_fill_hole;

        bool is_force_fpw;
        bool is_integer_disp;

        bool is_debug;
        Option()
            : patch_size(25),
              min_disparity(0),
              max_disparity(64),
              gamma(10.f),
              alpha(0.9f),
              tau_col(10.f),
              tau_grad(2.0f),
              num_iters(3),
              is_check_lr(false),
              lrcheck_thresh(0),
              is_fill_hole(false),
              is_force_fpw(false),
              is_integer_disp(false),
              is_debug(true) {}
    };

    struct Gradient {
        int16_t x, y;
    };
    PatchMatchStereo();
    ~PatchMatchStereo();

    bool Init(int32_t width, int32_t height, const Option& option);

    bool Match(const uint8_t* left_img, const uint8_t* right_img,
               float* left_disparity);

   private:
    void RandomInit();

    void ComputeGray();
    void ComputeGradient();

    void PlaneToDisparity();

    Option m_option;

    bool m_is_initialized;

    int32_t m_width;
    int32_t m_height;

    const uint8_t* m_left_img;
    const uint8_t* m_right_img;

    std::vector<uint8_t> m_left_gray;
    std::vector<uint8_t> m_right_gray;

    std::vector<Gradient> m_left_grad;
    std::vector<Gradient> m_right_grad;

    std::vector<float> m_left_cost;
    std::vector<float> m_right_cost;

    std::vector<DisparityPlane> m_left_plane;
    std::vector<DisparityPlane> m_right_plane;

    std::vector<float> m_left_disparity;
    std::vector<float> m_right_disparity;
};

class CostComputerPMS : public CostComputer {
   public:
    CostComputerPMS() = default;
    CostComputerPMS(const uint8_t* left_img, const uint8_t* right_img,
                    const PatchMatchStereo::Gradient* left_grad,
                    const PatchMatchStereo::Gradient* right_grad, int32_t width,
                    int32_t height, const PatchMatchStereo::Option& option);

    float Compute(int32_t x, int32_t y, float d) const override;

    PatchMatchStereo::Gradient GetGradient(
        const PatchMatchStereo::Gradient* data, int32_t x, int32_t y) const {
        return data[y * m_width + x];
    }
    glm::i8vec3 GetColor(const uint8_t* data, float x, int32_t y) const {
        glm::i8vec3 color;
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

#endif  // !PATCH_MATCH_STEREO_HPP
