#include <stdio.h>
#include "utils.h"
#include "reference_calc.h"
#include "timer.h"
#include <chrono>
//#include "cudaAdd.cu"

int addWithCuda(int *c, const int *a, const int *b, unsigned int size);
void your_rgba_to_greyscale(const uchar4* const h_rgbaImage,
    uchar4* const d_rgbaImage, unsigned char* const d_greyImage,
    size_t numRows, size_t numCols);

int main()
{
    //const int arraySize = 5;
    //const int a[arraySize] = { 1, 2, 3, 4, 5 };
    //const int b[arraySize] = { 10, 20, 30, 40, 50 };
    //int c[arraySize] = { 0 };

    //// Add vectors in parallel.
    //int cudaStatus = addWithCuda(c, a, b, arraySize);

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    uchar4        *h_rgbaImage, *d_rgbaImage;
    unsigned char *h_greyImage, *d_greyImage;

    std::string input_file;
    std::string output_file;

    input_file = "test.jpg";
    //input_file = "cinque_terre_small.jpg";
    output_file = "outputImg.jpg";

    Color2Grey color2Grey;
    color2Grey.preProcess(&h_rgbaImage, &h_greyImage, &d_rgbaImage, &d_greyImage, input_file);

    std::chrono::steady_clock::time_point  now = std::chrono::steady_clock::now();

    your_rgba_to_greyscale(h_rgbaImage, d_rgbaImage, d_greyImage,
        color2Grey.numRows(), color2Grey.numCols());

    cudaDeviceSynchronize();

    auto t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);

    checkCudaErrors(cudaGetLastError());
    printf("Your code ran in: %f msecs.\n", time_span);

    now = std::chrono::steady_clock::now();

    color2Grey.refColor2Grey();
    t2 = std::chrono::steady_clock::now();
    time_span = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - now);

    int err = printf("Your code ran in: %f msecs.\n", time_span);

    size_t numPixels = color2Grey.numCols() * color2Grey.numRows();
    checkCudaErrors(cudaMemcpy(h_greyImage, d_greyImage, sizeof(unsigned char) * numPixels, cudaMemcpyDeviceToHost));

    color2Grey.postProcess(output_file, h_greyImage);
    
    color2Grey.cleanup();
    return 0;
}

