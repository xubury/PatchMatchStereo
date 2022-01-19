#include "PatchMatchStereo.hpp"
#include <cstring>
#include <iostream>
#include <random>

PatchMatchStereo::PatchMatchStereo()
    : m_is_initialized(false), m_width(0), m_height(0) {}

PatchMatchStereo::~PatchMatchStereo() {}

bool PatchMatchStereo::Init(int32_t width, int32_t height,
                            const Option &option) {
  if (width <= 0 || height <= 0) {
    return false;
  }
  m_width = width;
  m_height = height;
  m_option = option;
  std::cout << "initializing patch match stereo" << std::endl;

  //··· 开辟内存空间
  const int32_t img_size = width * height;
  // const int32_t disp_range = option.max_disparity - option.min_disparity;
  // 灰度数据
  m_left_gray.resize(img_size);
  ;
  m_right_gray.resize(img_size);
  // 梯度数据
  m_left_grad.resize(img_size);
  m_right_grad.resize(img_size);
  // 代价数据
  m_left_cost.resize(img_size);
  m_right_cost.resize(img_size);
  // 视差图
  m_left_disparity.resize(img_size);
  m_right_disparity.resize(img_size);
  // 平面集
  m_left_plane.resize(img_size);
  m_right_plane.resize(img_size);

  m_is_initialized = true;

  return m_is_initialized;
}

bool PatchMatchStereo::Match(const uint8_t *left_img, const uint8_t *right_img,
                             float *left_disparity) {
  if (!m_is_initialized) {
    std::cout << "patch match stereo not init!\n" << std::endl;
    return false;
  }

  if (left_img == nullptr || right_img == nullptr) {
    return false;
  }
  m_left_img = left_img;
  m_right_img = right_img;

  RandomInit();

  ComputeGray();

  PlaneToDisparity();

  if (left_disparity) {
    memcpy(left_disparity, m_left_disparity.data(),
           m_width * m_height * sizeof(float));
  }

  return true;
}

void PatchMatchStereo::RandomInit() {
  std::cout << "initializing random plane" << std::endl;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> rand_d(m_option.min_disparity,
                                               m_option.max_disparity);
  std::uniform_real_distribution<float> rand_n(-1.f, 1.f);

  for (int k = 0; k < 2; ++k) {
    auto disp_ptr = k == 0 ? m_left_disparity.data() : m_right_disparity.data();
    auto plane_ptr = k == 0 ? m_left_plane.data() : m_right_plane.data();
    int32_t sign = k == 0 ? 1 : -1;
    for (int32_t y = 0; y < m_height; ++y) {
      for (int32_t x = 0; x < m_width; ++x) {
        const int32_t p = y * m_width + x;
        float disp = sign * rand_d(gen);
        if (m_option.is_integer_disp) {
          disp = std::round(disp);
        }
        disp_ptr[p] = disp;

        glm::vec3 norm;
        if (!m_option.is_force_fpw) {
          norm.x = rand_n(gen);
          norm.y = rand_n(gen);
          float z = rand_n(gen);
          while (z == 0) {
            z = rand_n(gen);
          }
          norm.z = z;
          norm = glm::normalize(norm);
        } else {
          norm = glm::vec3(0, 0, 1);
        }
        plane_ptr[p] = DisparityPlane(x, y, norm, disp);
      }
    }
  }
}

void PatchMatchStereo::ComputeGray() {}

void PatchMatchStereo::PlaneToDisparity() {
  std::cout << "computing plane to disparity" << std::endl;
  for (int k = 0; k < 2; ++k) {
    auto plane_ptr = k == 0 ? m_left_plane.data() : m_right_plane.data();
    auto disp_ptr = k == 0 ? m_left_disparity.data() : m_right_disparity.data();
    for (int32_t y = 0; y < m_height; ++y) {
      for (int32_t x = 0; x < m_width; ++x) {
        const size_t p = x + y * m_width;
        disp_ptr[p] = plane_ptr[p].GetDisparity(x, y);
      }
    }
  }
}
