#include "AddKernel.h"


__global__ void addKernel() {

  /*
  if (threadIdx.x < 4) {
    positions[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
  */

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;


  //float value = tex1Dfetch(tex, offset);

	printf("Spam");

}

/*


__global__ void cudaArrayPrintoutTexture(int width, int height)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  printf("Thread index: (%i, %i); cudaArray = %f\n", x, y, tex2D(texRef, x / (float)width + 0.5f, y / (float)height + 0.5f));
}


__global__ void cudaArrayPrintoutSurface(int width, int height)
{
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  float temp;

  surf2Dread(&temp, surf2D, x * 4, y);

  printf("Thread index: (%i, %i); cudaArray = %f\n", x, y, temp);
}
*/