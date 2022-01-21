#include <iostream>
#include <opencv2/opencv.hpp>

#include "PatchMatchStereo.hpp"
#include "Utils.hpp"

int main() {
    cv::Mat left_img = cv::imread("img/im2.png");
    cv::Mat right_img = cv::imread("img/im6.png");
    const int width = left_img.cols;
    const int height = left_img.rows;
    std::cout << "img size width: " << width << " height: " << height
              << std::endl;
    cv::Mat img(height, width, CV_32F, cv::Scalar(0));

    PatchMatchStereo stereo;
    std::cout << "starting PMS..." << std::endl;
    Timer timer;
    stereo.Init(width, height, PatchMatchStereo::Option());
    stereo.Match(left_img.ptr(), right_img.ptr(), img.ptr<float>());
    std::cout << "PMS took " << timer.GetElapsedMS() << " ms. " << std::endl;

    return 0;
}
