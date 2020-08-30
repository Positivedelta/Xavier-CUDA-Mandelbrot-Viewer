//
// (c) Max van Daalen, August 2020
//

#include <iostream>
#include "cuda_renderer.hpp"

//
// nvcc -std=c++14 -O3 -c cuda_renderer.cu -o cuda_renderer.o
//

__global__ void doMandelbrot(uint8_t *pixelBuffer, const int32_t pixelBufferSpan, const cuDoubleComplex start, const cuDoubleComplex step, const int32_t maxIterations)
{
    const int32_t x = threadIdx.x + blockIdx.x * blockDim.x;
    const int32_t y = threadIdx.y + blockIdx.y * blockDim.y;
    const cuDoubleComplex c = make_cuDoubleComplex(cuCreal(start) + (x * cuCreal(step)), cuCimag(start) + (y * cuCimag(step)));

    // notes 1. using fma() as it rounds coping with NaN and +/-  infinities, (zReal^2 + zImag^2) doesn't and eventually fails
    //       2. the GPU's fma(x, y, z) method efficiently computes (x * y) + z
    //       3. for speed, use (fabs(zReal) + fabs(zImag)) as an appromimation to norm(), however this does generate some minor artefacts
    //
    cuDoubleComplex z = make_cuDoubleComplex(0.0, 0.0);
    int32_t iterations = maxIterations;
    while ((iterations > 0) && (fma(cuCreal(z), cuCreal(z), (cuCimag(z) * cuCimag(z))) < 4.0))
    {
        z = cuCadd(cuCmul(z, z), c);
        iterations--;
    }

    // using a smooth bernstein polynomial to generate the RGB components
    //
    const double t = (double)iterations / (double)maxIterations;
    const uint8_t r = (uint8_t)(9.0 * (1.0 - t) * t * t * t * 255.0);
    const uint8_t g = (uint8_t)(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0);
    const uint8_t b = (uint8_t)(8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0);

    // note, no need to update the alpha channel, it's expected to be pre-initialised outside of the kernel
    //
    const int32_t pixelAddr = (x << 2) + (y * pixelBufferSpan);
    pixelBuffer[pixelAddr] = r;
    pixelBuffer[pixelAddr + 1] = g;
    pixelBuffer[pixelAddr + 2] = b;
}

// note, the Xavier has 512 cores, hence using a block dimension of (32, 16)
//
CudaRenderer::CudaRenderer(const int32_t width, const int32_t height, const int32_t pixelBufferSpan):
    width(width), height(height), pixelBufferSpan(pixelBufferSpan), dimGrid(dim3(width / 32, height / 16)), dimBlock(dim3(32, 16)) {
        std::cout << "CUDA Kernel grid dimensions: (" << dimGrid.x << ", " << dimGrid.y << "), block: (" << dimBlock.x << ", " << dimBlock.y << ")\n";

        // used to time the doMandelbrot() kernel
        //
        cudaEventCreate(&timerStart);
        cudaEventCreate(&timerStop);

        const int32_t pixelBufferSize = pixelBufferSpan * height;
        std::cout << "Attempting to malloc " << pixelBufferSize << " bytes\n";

        cudaError_t status = cudaMallocManaged(&pixelBuffer, pixelBufferSize);
        switch (status)
        {
            case cudaSuccess:
                std::cout << "Managed malloc successful\n";
                break;

            case cudaErrorMemoryAllocation:
                std::cout << "Unable to allocate enough memory\nProgram exiting!\n";
                doExit(status);
                break;

            case cudaErrorNotSupported:
                std::cout << "Operation is not supported on the current device\nProgram exiting!\n";
                doExit(status);
                break;

            case cudaErrorInvalidValue:
                std::cout << "One or more of the parameters passed to the API call is not within an acceptable range\nProgram exiting!\n";
                doExit(status);
                break;

            default:
                std::cout << "Managed malloc error: " << status << "\nProgram exiting!\n";
                doExit(status);
                break;
        }
}

CudaRenderer::~CudaRenderer()
{
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
    cudaFree(pixelBuffer);
}

uint8_t* CudaRenderer::getPixelBuffer()
{
    return pixelBuffer;
}

void CudaRenderer::paintMandelbrot(const std::complex<double> start, const std::complex<double> end, const int32_t maxIterations)
{
    std::cout << "CUDA render starting...";
    const std::complex<double> range = end - start;
    const std::complex<double> step = std::complex<double>(range.real() / (double)width, range.imag() / (double)height);

    const cuDoubleComplex cuStart = make_cuDoubleComplex(start.real(), start.imag());
    const cuDoubleComplex cuStep = make_cuDoubleComplex(step.real(), step.imag());
    cudaEventRecord(timerStart, 0);
    doMandelbrot<<<dimGrid, dimBlock>>>(pixelBuffer, pixelBufferSpan, cuStart, cuStep, maxIterations);
    cudaEventRecord(timerStop, 0);
    const int32_t status = cudaDeviceSynchronize();
    if (status == cudaSuccess)
    {
        cudaEventSynchronize(timerStop);

        float elapsedTime;
        cudaEventElapsedTime(&elapsedTime, timerStart, timerStop);
        std::cout << " completed in "<< elapsedTime << " ms\n\n";
    }
    else
    {
        std::cout << "\nCUDA device synchronisation failure: " << status << "\nProgram exiting!\n";
        doExit(status);
    }
}

void CudaRenderer::doExit(int32_t status)
{
    cudaEventDestroy(timerStart);
    cudaEventDestroy(timerStop);
    cudaFree(pixelBuffer);

    // FIXME! not sure about the reset, it would probably mess up any other running applications that were using the GPU
    //
    cudaDeviceReset();
    exit(status);
}
