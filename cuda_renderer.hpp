//
// (c) Max van Daalen, August 2020
//

#ifndef CUDA_RENDERER_HPP
#define CUDA_RENDERER_HPP

#include <cstdint>
#include <complex>
#include "cuComplex.h"
#include "cuda_runtime.h"

class CudaRenderer
{
    private:
        const int32_t width, height, pixelBufferSpan;
        const dim3 dimBlock;
        const dim3 dimGrid;
        cudaEvent_t timerStart, timerStop;
        uint8_t *pixelBuffer;

    public:
        CudaRenderer(const int32_t width, const int32_t height, const int32_t pixelBufferSpan);
        ~CudaRenderer();

        uint8_t* getPixelBuffer();
        void paintMandelbrot(const std::complex<double> start, const std::complex<double> end, const int32_t maxIterations);

    private:
        void doExit(int32_t status);
};

#endif
