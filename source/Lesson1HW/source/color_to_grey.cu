#include "utils.h"
#include "device_launch_parameters.h"


__global__
void rgba_to_greyscale(const uchar4* const rgbaImage,
    unsigned char* const greyImage, int numRows, int numCols)
{


#if 1
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;

    printf("idx:%d, idy:%d\n", idx, idy);

    if (idx < numCols && idy < numRows)
    {
        uchar4 rgb = rgbaImage[idy * numCols + idx];
        greyImage[idy * numCols + idx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
    }
#else
    const unsigned int pixIdx = blockDim.x * blockIdx.x + threadIdx.x;
    uchar4 rgb = rgbaImage[pixIdx];
    greyImage[pixIdx] = 0.299f * rgb.x + 0.587f * rgb.y + 0.114f * rgb.z;
#endif // 0

}


void your_rgba_to_greyscale(const uchar4* const h_rgbaImage,
    uchar4* const d_rgbaImage, unsigned char* const d_greyImage,
    size_t numRows, size_t numCols)
{
#if 1
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);
#else

    int threads = 256;
    size_t pixels = numRows * numCols;
    dim3 threadsPerBlock(threads);
    dim3 blocksPerGrid((pixels + threads - 1) / threads);

#endif // 0

    rgba_to_greyscale << <blocksPerGrid, threadsPerBlock >> >(d_rgbaImage, d_greyImage, numRows, numCols);

    int a = 1;
}
