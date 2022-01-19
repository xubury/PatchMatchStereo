#include <iostream>
#include <opencv2/opencv.hpp>

#include "PatchMatchStereo.hpp"

int main() {
    cv::Mat left_img = cv::imread("img/im2.png");
    cv::Mat right_img = cv::imread("img/im6.png");
    const int width = left_img.cols;
    const int height = left_img.rows;
    std::cout << "img size width: " << width << " height: " << height
              << std::endl;
    cv::Mat img(height, width, CV_32F, cv::Scalar(0));

    PatchMatchStereo stereo;
    stereo.Init(width, height, PatchMatchStereo::Option());

    stereo.Match(left_img.ptr(), right_img.ptr(), img.ptr<float>());

    while (true) {
        cv::imshow("Test Window", img);
        if (cv::waitKey() == 27) {
            break;
        }
    }
    return 0;
}
