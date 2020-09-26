# Xavier CUDA Mandelbrot Viewer
#### A CUDA based Mandelbrot viewer for the NVIDIA Xavier written in C++17 and rendered using GTK

#### Prerequisites
```
sudo apt install libgtkmm-3.0-dev
sudo apt libgstreamermm-1.0-dev
```

#### Raw build
```
nvcc -std=c++14 -O3 -gencode arch=compute_72,code=sm_72 -c cuda_renderer.cu -o cuda_renderer.o
g++ -std=c++17 -O3 -I /usr/local/cuda/include -c mandelbrot.cpp -o mandelbrot.o `pkg-config --cflags gtkmm-3.0`
nvcc cuda_renderer.o mandelbrot.o -o mandelbrot `pkg-config --libs gtkmm-3.0` -gencode arch=compute_72,code=sm_72
```

#### Notes
```
- An appropraite Makefile has been included in this repository, use: make clean all
- Add -Xptxas=-v to the NVCC compiler options to obtain shared memory and register stats for a given kernel
```

#### Example Render
```
An example render, zoomed in by a factor 2^46
At this level of zoom, the pixelation is caused by the loss of precision in the 64 bit floating point calculations
```
<img src='https://github.com/Positivedelta/Xavier-CUDA-Mandelbrot-Viewer/blob/master/double_limit_x70368744177664.png' width='912' height='652'>
