#include <iostream>

#include "Window.h"
#include "Cuda.h"
#include "Engine.h"

__global__ void addKernel(int *c, const int *a, const int *b)
{
  printf("thread id: %i", threadIdx.x);
  int i = threadIdx.x;
  c[i] = a[i] + b[i];
}


void computeExample() {
  int* gpuA;
  int* gpuB;
  int* gpuC;

  cudaMalloc((void**)&gpuA, 5 * sizeof(int));
  cudaMalloc((void**)&gpuB, 5 * sizeof(int));
  cudaMalloc((void**)&gpuC, 5 * sizeof(int));

  int a[5] = { 1, 2, 3, 4, 5 };
  int b[5] = { 5, 4, 3, 2, 6 };
  int c[5] = { 0, 0, 0, 0, 0 };

  cudaMemcpy(gpuA, a, 5 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuB, b, 5 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(gpuC, c, 5 * sizeof(int), cudaMemcpyHostToDevice);

  for (unsigned int i = 0; i < 5; i++) {
    std::cout << "c[" << i << "] = " << c[i] << std::endl;
  }

  dim3 blocks{ 1, 1, 1 };
  int numberOfThreadsPerBlock = 5;

  addKernel << <1, numberOfThreadsPerBlock >> >(gpuC, gpuA, gpuB);
  cudaGetLastError();

  cudaDeviceSynchronize();

  cudaMemcpy(a, gpuA, 5 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(b, gpuB, 5 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(c, gpuC, 5 * sizeof(int), cudaMemcpyDeviceToHost);


  std::cout << "---------" << std::endl;

  for (unsigned int i = 0; i < 5; i++) {
    std::cout << "c[" << i << "] = " << c[i] << std::endl;
  }

  cudaFree(gpuA);
  cudaFree(gpuB);
  cudaFree(gpuC);

}

int main(int argc, char* argv[]) {

  Window window{"PBD Simulation CUDA", 1024, 768};

  Cuda cuda{2, 0};

  Engine engine{window.getGLFWWindow()};

  computeExample();

  engine.run();

  //std::cin.get();

  return EXIT_SUCCESS;

}