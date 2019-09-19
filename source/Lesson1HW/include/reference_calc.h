#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "utils.h"

class Color2Grey
{
public:
    void preProcess(uchar4 **inputImage, unsigned char **greyImage,
        uchar4 **d_rgbaImage, unsigned char **d_greyImage,
        const std::string &filename);

    void postProcess(const std::string& output_file, unsigned char* data_ptr);

    void refColor2Grey();

    void cleanup();

    size_t numRows() { return imageRGBA_.rows; }
    size_t numCols() { return imageRGBA_.cols; }
private:
    cv::Mat imageGrey_;
    cv::Mat imageRGBA_;
    cv::Mat tmpGrey_;

    uchar4* d_rgbaImage_;
    unsigned char* d_greyImage_;
};


