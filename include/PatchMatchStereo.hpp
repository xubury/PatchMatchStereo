#ifndef PATCH_MATCH_STEREO_HPP
#define PATCH_MATCH_STEREO_HPP

#include <cstdint>
#include <memory>
#include <vector>

// TODO: Put this to another separate module and
//  make an option for no glm dependency in the future
#include <glm/glm.hpp>
using Color = glm::i8vec3;
using Vector2f = glm::vec2;
using Vector3f = glm::vec3;
using Vector2i = glm::i32vec2;

inline const auto& Dot =
    static_cast<float (*)(const Vector3f&, const Vector3f&)>(glm::dot);
inline const auto& Normalize =
    static_cast<Vector3f (*)(const Vector3f&)>(glm::normalize);
/////////////////////////////////////////////////////////////////////

struct DisparityPlane {
    Vector3f p;
    DisparityPlane() = default;
    DisparityPlane(const Vector3f& v) { p = v; }
    DisparityPlane(const int32_t x, const int32_t y, const Vector3f& n,
                   const float d) {
        p.x = -n.x / n.z;
        p.y = -n.y / n.z;
        p.z = (n.x * x + n.y * y + n.z * d) / n.z;
    }

    /**
     * \param x		像素x坐标
     * \param y		像素y坐标
     * \return 像素(x,y)的视差
     */
    float GetDisparity(const int32_t x, const int32_t y) const {
        return Dot(p, Vector3f(x, y, 1.0f));
    }

    /** \brief 获取平面的法线 */
    Vector3f GetNormal() const {
        Vector3f n(p.x, p.y, -1.0f);
        return Normalize(n);
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
        return Vector3f(-p.x, -p.y, -p.z - p.x * d);
    }

    // operator ==
    bool operator==(const DisparityPlane& v) const { return p == v.p; }
    // operator !=
    bool operator!=(const DisparityPlane& v) const { return p != v.p; }
};

class PMSPropagation;

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
            : patch_size(35),
              min_disparity(0),
              max_disparity(64),
              gamma(10.f),
              alpha(0.9f),
              tau_col(10.f),
              tau_grad(2.0f),
              num_iters(3),
              is_check_lr(true),
              lrcheck_thresh(1.0f),
              is_fill_hole(true),
              is_force_fpw(false),
              is_integer_disp(false),
              is_debug(true) {}
    };

    using Gradient = Vector2i;

    PatchMatchStereo();
    ~PatchMatchStereo();

    bool Init(int32_t width, int32_t height, const Option& option);

    bool Match(const uint8_t* left_img, const uint8_t* right_img,
               float* left_disparity);

   private:
    void PreCompute();

    void Propagation();

    void LRCheck();

    void FillHole();

    void WeightMedianFilter(const uint8_t* img_ptr,
                            const std::vector<Vector2i>& mismatches,
                            float* disparity) const;

    static void RandomInit(DisparityPlane* plane, float* disparity, int width,
                           int height, const Option& option, int8_t sign);

    static void ComputeGray(const uint8_t* img, uint8_t* gray, int32_t width,
                            int32_t height);
    static void ComputeGradient(const uint8_t* gray, Gradient* grad,
                                int32_t width, int32_t height);

    static void PlaneToDisparity(const DisparityPlane* plane, float* disparity,
                                 int32_t width, int32_t height);

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

    std::vector<Vector2i> m_left_mismatches;
    std::vector<Vector2i> m_right_mismatches;
};

#endif  // !PATCH_MATCH_STEREO_HPP
