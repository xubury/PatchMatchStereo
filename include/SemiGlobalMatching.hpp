#ifndef SEMI_GLOBAL_MATCHING_HPP
#define SEMI_GLOBAL_MATCHING_HPP

#include <cstdint>

class SemiGlobalMatching {
   public:
    struct Option {
        uint8_t num_paths;
        int32_t min_disparity;
        int32_t max_disparity;

        int32_t p1;
        int32_t p2;

        Option()
            : num_paths(8),
              min_disparity(0),
              max_disparity(64),
              p1(10),
              p2(150) {}
    };
    SemiGlobalMatching();
    ~SemiGlobalMatching();
    bool Init(int32_t width, int32_t height, const Option& option);
    bool Match(const uint8_t* left_img, const uint8_t* right_img,
               float* left_disparity);

   private:
    void CensusTransform();
    void ComputeCost();
    void CostAggregation();
    void ComputeDispairty();
    void LRCheck();

    int32_t m_width;
    int32_t m_height;
    bool m_is_initialized;
};

#endif
