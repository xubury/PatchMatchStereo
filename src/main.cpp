#include <iostream>

#include "PatchMatchStereo.hpp"
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

    std::vector<float> disparity_map(width * height);
    PatchMatchStereo stereo;
    std::cout << "starting PMS..." << std::endl;
    timer.Restart();
    stereo.Init(width, height, PatchMatchStereo::Option());
    stereo.Match(left_img, right_img, disparity_map.data());
    std::cout << "PMS took " << timer.GetElapsedMS() << " ms. " << std::endl;

    stbi_image_free(left_img);
    stbi_image_free(right_img);
    return 0;
}
