#include <iostream>

#include "PatchMatchStereo.hpp"
#include "SemiGlobalMatching.hpp"
#include "Utils.hpp"
#include "stb_image.h"

int main(int argc, char *args[]) {
    if (argc < 3) {
        std::cout << "Not enough arguments!"
                  << "\n";
        std::cout << "Usage: " << args[0]
                  << " [path-to-left-image] [path-to-right-image]" << std::endl;
        return 1;
    }

    Timer timer;

    int width, height, channels;
    std::cout << "Loading images..." << std::endl;
    timer.Restart();
    uint8_t *left_img = stbi_load(args[1], &width, &height, &channels, 0);
    uint8_t *right_img = stbi_load(args[2], &width, &height, &channels, 0);
    assert(left_img && right_img);

    std::cout << "image loading took " << timer.GetElapsedMS() << " ms.\n";
    std::cout << "img size width: " << width << " height: " << height
              << std::endl;

    std::vector<float> pms_disp(width * height);
    {
        std::cout << "starting PMS..." << std::endl;
        timer.Restart();
        PatchMatchStereo pms;
        PatchMatchStereo::Option option;
        // option.is_fill_hole = false;
        // option.is_check_lr = false;
        option.num_iters = 3;
        pms.Init(width, height, option);
        pms.Match(left_img, right_img, pms_disp.data());
        std::cout << "PMS took " << timer.GetElapsedMS() << " ms. "
                  << std::endl;
    }

    std::vector<float> sgm_disp(width * height);
    {
        std::cout << "starting SGM..." << std::endl;
        timer.Restart();
        SemiGlobalMatching sgm;
        SemiGlobalMatching::Option option;
        // option.is_fill_hole = false;
        // option.is_check_lr = false;

        sgm.Init(width, height, option);
        sgm.Match(left_img, right_img, sgm_disp.data());
        std::cout << "SGM took " << timer.GetElapsedMS() << " ms. "
                  << std::endl;
    }

    stbi_image_free(left_img);
    stbi_image_free(right_img);
    return 0;
}
