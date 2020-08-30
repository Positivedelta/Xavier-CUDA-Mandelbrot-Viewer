CC=g++
CXX_COMPILE_FLAGS=-std=c++17 -O3 -I /usr/local/cuda/include `pkg-config --cflags gtkmm-3.0`
NVCC=nvcc
NVCC_COMPILE_FLAGS=-std=c++14 -O3 -gencode arch=compute_72,code=sm_72 -Xptxas=-v
NVCC_LINK_FLAGS=-gencode arch=compute_72,code=sm_72 `pkg-config --libs gtkmm-3.0`

all: cuda_renderer.o mandelbrot.o
	$(NVCC) $(NVCC_LINK_FLAGS) cuda_renderer.o mandelbrot.o -o mandelbrot

cuda_renderer.o: cuda_renderer.cu
	$(NVCC) $(NVCC_COMPILE_FLAGS) -c cuda_renderer.cu

mandelbrot.o: mandelbrot.cpp
	$(CC) $(CXX_COMPILE_FLAGS) -c mandelbrot.cpp

clean:
	rm -f *.o mandelbrot
