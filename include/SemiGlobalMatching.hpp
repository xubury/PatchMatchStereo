#ifndef SEMI_GLOBAL_MATCHING_HPP
#define SEMI_GLOBAL_MATCHING_HPP

#include <cstdint>
#include <vector>

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
    void ComputeDisparity();
    void LRCheck();

    Option m_option;
    int32_t m_width;
    int32_t m_height;
    const uint8_t* m_left_img;
    const uint8_t* m_right_img;

    std::vector<uint32_t> m_left_census;
    std::vector<uint32_t> m_right_census;
    std::vector<uint8_t> m_cost;
    std::vector<uint8_t> m_cost_aggr;
    std::vector<float> m_left_disparity;
    bool m_is_initialized;
};

#endif
