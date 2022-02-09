#ifndef SEMI_GLOBAL_MATCHING_HPP
#define SEMI_GLOBAL_MATCHING_HPP

#include "Utils.hpp"

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

        float lrcheck_thresh;
        bool is_check_lr;
        bool is_fill_hole;
        Option()
            : num_paths(8),
              min_disparity(0),
              max_disparity(64),
              p1(10),
              p2(150),
              lrcheck_thresh(1),
              is_check_lr(true),
              is_fill_hole(true) {}
    };
    SemiGlobalMatching();
    ~SemiGlobalMatching();
    bool Init(int32_t width, int32_t height, const Option& option);
    bool Match(const uint8_t* left_img, const uint8_t* right_img,
               float* left_disparity);

    static void CensusTransform5x5(const uint8_t* src, uint32_t* census,
                                   int32_t width, int32_t height);

   private:
    void CensusTransform();
    static void ComputeCost(uint8_t* cost_ptr, const uint32_t* left_census,
                            const uint32_t* right_census, int32_t width,
                            int32_t height, int32_t min_disparity,
                            int32_t max_disparity);
    void CostAggregation(const uint8_t* img, const uint8_t* cost);
    void ComputeDisparity(float* disparity);
    void LRCheck();
    void FillHole();

    static void CostAggregationLeft(const uint8_t* img, int32_t width,
                                    int32_t height, int32_t min_disparity,
                                    int32_t max_disparity, uint32_t p1,
                                    uint32_t p2, const uint8_t* cost,
                                    uint8_t* cost_aggr, bool is_forward);
    static void CostAggregationTop(const uint8_t* img, int32_t width,
                                   int32_t height, int32_t min_disparity,
                                   int32_t max_disparity, int32_t p1,
                                   int32_t p2, const uint8_t* cost,
                                   uint8_t* cost_aggr, bool is_forward);
    static void CostAggregationTopLeft(const uint8_t* img, int32_t width,
                                       int32_t height, int32_t min_disparity,
                                       int32_t max_disparity, int32_t p1,
                                       int32_t p2, const uint8_t* cost,
                                       uint8_t* cost_aggr, bool is_forward);
    static void CostAggregationTopRight(const uint8_t* img, int32_t width,
                                        int32_t height, int32_t min_disparity,
                                        int32_t max_disparity, int32_t p1,
                                        int32_t p2, const uint8_t* cost,
                                        uint8_t* cost_aggr, bool is_forward);

    Option m_option;

    int32_t m_width;
    int32_t m_height;

    bool m_is_initialized;

    const uint8_t* m_left_img;
    const uint8_t* m_right_img;

    std::vector<uint8_t> m_left_gray;
    std::vector<uint8_t> m_right_gray;

    std::vector<uint32_t> m_left_census;
    std::vector<uint32_t> m_right_census;

    std::vector<uint8_t> m_cost;
    std::vector<uint8_t> m_cost_aggr;
    std::vector<uint8_t> m_cost_aggr_left;
    std::vector<uint8_t> m_cost_aggr_right;
    std::vector<uint8_t> m_cost_aggr_top;
    std::vector<uint8_t> m_cost_aggr_bottom;
    std::vector<uint8_t> m_cost_aggr_tl;
    std::vector<uint8_t> m_cost_aggr_br;
    std::vector<uint8_t> m_cost_aggr_tr;
    std::vector<uint8_t> m_cost_aggr_bl;

    std::vector<float> m_left_disparity;
    std::vector<float> m_right_disparity;

    std::vector<Vector2i> m_left_mismatches;
    std::vector<Vector2i> m_right_mismatches;
};

#endif
