#ifndef PATCH_MATCH_STEREO_HPP
#define PATCH_MATCH_STEREO_HPP

#include <cstdint>
#include <glm/glm.hpp>
#include <memory>

template <typename T>
using Scope = std::unique_ptr<T>;
template <typename T, typename... Args>
constexpr Scope<T> CreateScope(Args&&... args) {
    return std::make_unique<T>(std::forward<Args>(args)...);
}

template <typename T>
using Ref = std::shared_ptr<T>;
template <typename T, typename... Args>
constexpr Ref<T> CreateRef(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

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
              is_integer_disp(false) {}
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

    void PlaneToDisparity();

    Option m_option;

    bool m_is_initialized;

    int32_t m_width;
    int32_t m_height;

    const uint8_t* m_left_img;
    const uint8_t* m_right_img;

    Scope<uint8_t[]> m_left_gray;
    Scope<uint8_t[]> m_right_gray;

    Scope<Gradient[]> m_left_grad;
    Scope<Gradient[]> m_right_grad;

    Scope<float[]> m_left_cost;
    Scope<float[]> m_right_cost;

    Scope<DisparityPlane[]> m_left_plane;
    Scope<DisparityPlane[]> m_right_plane;

    Scope<float[]> m_left_disparity;
    Scope<float[]> m_right_disparity;
};

#endif  // !PATCH_MATCH_STEREO_HPP
