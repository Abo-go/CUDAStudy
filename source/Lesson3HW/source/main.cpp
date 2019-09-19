#include <stdio.h>
#include "utils.h"
#include <chrono>
#include <string>
#include <opencv2/opencv.hpp>

void your_gaussian_blur(const uchar4 * const h_inputImageRGBA,
    uchar4 * const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA,
    const size_t numRows, const size_t numCols,
    unsigned char *d_redBlurred,
    unsigned char *d_greenBlurred,
    unsigned char *d_blueBlurred,
    const int filterWidth);

void your_gaussian_blur2(const uchar4 * const h_inputImageRGBA,
    uchar4 * const d_inputImageRGBA,
    uchar4* const d_outputImageRGBA,
    const size_t numRows, const size_t numCols,
    unsigned char *d_redBlurred,
    unsigned char *d_greenBlurred,
    unsigned char *d_blueBlurred,
    const int filterWidth);

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth);

void allocateMemoryAndCopyToGPU2(const size_t numRowsImage, const size_t numColsImage,
    const float* const h_filter, const size_t filterWidth);

void serial_blur(const cv::Mat& inputImg, cv::Mat& outputImg, cv::Size kSize)
{
    //cv::blur(inputImg, outputImg, cv::Size(5, 5));
    cv::GaussianBlur(inputImg, outputImg, kSize, 1.5);
}

float gaussian_kernel(float sigma, float x, float y)
{
    float a = 0.25 * 0.5 / (sigma * sigma);
    return std::exp(-(x*x + y*y) * a);
}

int main()
{
    std::string inputFile;
    std::string outputFile;
    std::string outputFile2;

    inputFile = "4kImg.jpg";
    //inputFile = "cinque_terre_small.jpg";
    outputFile = "outputImg.jpg";
    outputFile2 = "outputImg2.jpg";

    cv::Mat inputImg = cv::imread(inputFile, CV_LOAD_IMAGE_COLOR);
    size_t numRows = inputImg.rows;
    size_t numCols = inputImg.cols;

    cv::Mat outputImg = cv::Mat(numRows, numCols, CV_8UC4);
    cv::Mat outputImg2 = cv::Mat(numRows, numCols, CV_8UC4);
    cvtColor(inputImg, inputImg, CV_BGR2RGBA);

    uchar4* h_inputImageRGBA = inputImg.ptr<uchar4>(0), *h_outputImageRGBA = outputImg.ptr<uchar4>(0);
    uchar4* d_inputImageRGBA, *d_outputImageRGBA;
    unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;

    size_t filterWidth = 7;
#if 0
    float* h_filter = new float[filterWidth * filterWidth];
    float* centerPtr = &h_filter[filterWidth * filterWidth / 2];

    size_t filterCenter = filterWidth / 2;
    float filterSum = 0;
    for (int i = 0; i < filterCenter + 1; i++)
    {
        for (int j = 0; j < filterCenter + 1; j++)
        {
            float tmpKernel = gaussian_kernel(1.5, j, i);
            //float tmpKernel = 1;
            centerPtr[i * filterWidth + j] = centerPtr[i * filterWidth - j]
                = centerPtr[-i * filterWidth + j] = centerPtr[-i * filterWidth - j]
                = tmpKernel;

            if (i == 0 && j == 0)
            {
                filterSum += tmpKernel;
            }
            else if (i == 0 && j != 0)
            {
                filterSum += tmpKernel * 2;
            }
            else if (i != 0 && j == 0)
            {
                filterSum += tmpKernel * 2;
            }
            else
            {
                filterSum += tmpKernel * 4;
            }
        }
    }

    filterSum = filterWidth * filterWidth / filterSum;

    for (int i = 0; i < filterWidth * filterWidth; i++)
    {
        h_filter[i] *= filterSum;
    }
#else
    float* h_filter = new float[filterWidth];
    float* centerPtr = &h_filter[filterWidth / 2];

    size_t filterCenter = filterWidth / 2;
    float filterSum = 0;
    float sigma = 1.5;
    float sigma2 = -0.5 / (sigma * sigma);
    for (int i = 0; i < filterCenter + 1; i++)
    {
        centerPtr[i] = centerPtr[-i] = std::exp(i * i * sigma2);
        filterSum += centerPtr[i] * 2;
        if (i == 0)
        {
            filterSum -= centerPtr[i];
        }
    }

    filterSum = filterWidth / filterSum;

    for (int i = 0; i < filterWidth; i++)
    {
        h_filter[i] *= filterSum;
    }
#endif

    

    checkCudaErrors(cudaMalloc(&d_redBlurred, sizeof(unsigned char) * numRows * numCols));
    checkCudaErrors(cudaMalloc(&d_greenBlurred, sizeof(unsigned char) * numRows * numCols));
    checkCudaErrors(cudaMalloc(&d_blueBlurred, sizeof(unsigned char) * numRows * numCols));

    checkCudaErrors(cudaMalloc(&d_inputImageRGBA, sizeof(uchar4) * numRows * numCols));
    checkCudaErrors(cudaMalloc(&d_outputImageRGBA, sizeof(uchar4) * numRows * numCols));

#if 0
    allocateMemoryAndCopyToGPU(numRows, numCols, h_filter, filterWidth);

    checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice));

    std::chrono::steady_clock::time_point  now = std::chrono::steady_clock::now();

    your_gaussian_blur(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,
        numRows, numCols, d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);

    printf("Your code ran in: %f msecs.\n", time_span);
#else
    allocateMemoryAndCopyToGPU2(numRows, numCols, h_filter, filterWidth);

    checkCudaErrors(cudaMemcpy(d_inputImageRGBA, h_inputImageRGBA, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice));

    std::chrono::steady_clock::time_point  now = std::chrono::steady_clock::now();

    your_gaussian_blur2(h_inputImageRGBA, d_inputImageRGBA, d_outputImageRGBA,
        numRows, numCols, d_redBlurred, d_greenBlurred, d_blueBlurred, filterWidth);

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);

    printf("Your code ran in: %f msecs.\n", time_span);
#endif

    checkCudaErrors(cudaMemcpy(h_outputImageRGBA, d_outputImageRGBA, sizeof(uchar4) * numRows * numCols, cudaMemcpyDeviceToHost));

    cv::imwrite(outputFile, outputImg);

    //serial_blur(inputImg, outputImg);

    now = std::chrono::steady_clock::now();

    serial_blur(inputImg, outputImg2, cv::Size(filterWidth, filterWidth));

    t2 = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);

    printf("Your code ran in: %f msecs.\n", time_span);

    cv::imwrite(outputFile2, outputImg2);

    delete h_filter;
    return 0;
}

