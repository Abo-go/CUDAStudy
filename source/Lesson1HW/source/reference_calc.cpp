#include "reference_calc.h"

void Color2Grey::preProcess(uchar4 **inputImage, unsigned char **greyImage, uchar4 **d_rgbaImage, unsigned char **d_greyImage, const std::string &filename)
{
    checkCudaErrors(cudaFree(0));
    cv::Mat image;
    image = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
        std::cerr << "Couldn't open file: " << filename << std::endl;
        exit(1);
    }

    cv::cvtColor(image, imageRGBA_, CV_BGR2RGBA);

    imageGrey_.create(image.rows, image.cols, CV_8UC1);
    tmpGrey_.create(image.rows, image.cols, CV_8UC1);

    *inputImage = (uchar4 *)imageRGBA_.ptr<unsigned char>(0);
    *greyImage = imageGrey_.ptr<unsigned char>(0);

    const size_t numPixels = numRows() * numCols();

    //allocate memory on the device for both input and output
    checkCudaErrors(cudaMalloc(d_rgbaImage, sizeof(uchar4) * numPixels));
    checkCudaErrors(cudaMalloc(d_greyImage, sizeof(unsigned char) * numPixels));
    checkCudaErrors(cudaMemset(*d_greyImage, 0, numPixels * sizeof(unsigned char))); //make sure no memory is left laying around
                                                                                //copy input array to the GPU
    checkCudaErrors(cudaMemcpy(*d_rgbaImage, *inputImage, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice));

    d_rgbaImage_ = *d_rgbaImage;
    d_greyImage_ = *d_greyImage;
}

void Color2Grey::postProcess(const std::string& output_file, unsigned char* data_ptr)
{
    cv::Mat output(numRows(), numCols(), CV_8UC1, data_ptr);

    cv::imwrite(output_file, output);
}

void Color2Grey::refColor2Grey()
{
    //cv::Mat tmpGrey;
    cv::cvtColor(imageRGBA_, tmpGrey_, CV_RGBA2GRAY);
    //uchar4* rgbaPtr = imageRGBA_.ptr<uchar4>(0);
    //unsigned char* greyPtr = tmpGrey_.ptr<unsigned char>(0);
    //for (int i = 0; i < numRows() * numCols(); i++)
    //{
    //    uchar4 pix = rgbaPtr[i];
    //    greyPtr[i] = 0.299 * pix.x + 0.589 * pix.y + 0.114 * pix.z;
    //}
}

void Color2Grey::cleanup()
{
    cudaFree(d_rgbaImage_);
    cudaFree(d_greyImage_);
}
