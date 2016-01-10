#include "Kernels.h"

surface<void, cudaSurfaceType2D> positions4;
surface<void, cudaSurfaceType2D> colors4;



// --------------------------------------------------------------------------

__global__ void applyForces(const unsigned int numberOfParticles,
                            const unsigned int textureWidth) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = idx % textureWidth;
  const unsigned int y = idx / textureWidth;

  float4 elementRead = make_float4(5.0f, 5.0f, 5.0f, 5.0f);
  surf2Dread(&elementRead, positions4, x * sizeof(float4), y);

  float4 elementWrite = elementRead;
  elementWrite.y = elementWrite.y - 0.01f;
  surf2Dwrite(elementWrite, positions4, x * sizeof(float4), y);

  surf2Dwrite(make_float4(1.0f, 1.0f, 1.0f, 1.0f), colors4, x * sizeof(float4), y);
}

void cudaCallApplyForces() {
  auto glShared = GL_Shared::getInstance();

  auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
  unsigned int textureWidth = glShared.get_texture("positions4")->width_;

  dim3 blocks((*numberOfParticles)/128, 1, 1);
  dim3 threads(128, 1, 1);

  applyForces<<<blocks, threads>>>(*numberOfParticles, textureWidth);
}

// --------------------------------------------------------------------------





















void initializeTexture(surface<void, cudaSurfaceType2D>& surf, const std::string name) {
  auto glShared = GL_Shared::getInstance();
  GLuint gluint = glShared.get_texture(name)->texture_;

  cudaStream_t cudaStream;
  CUDA(cudaStreamCreate(&cudaStream));

  cudaGraphicsResource* resource;
  CUDA(cudaGraphicsGLRegisterImage(&resource,
                                   gluint,
                                   GL_TEXTURE_2D,
                                   cudaGraphicsRegisterFlagsSurfaceLoadStore));

  CUDA(cudaGraphicsMapResources(1, &resource, cudaStream));

  cudaArray* array;
  CUDA(cudaGraphicsSubResourceGetMappedArray(&array, resource, 0, 0));

  CUDA(cudaBindSurfaceToArray(surf, array));

  CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));
}
#define CUDA_INITIALIZE_SHARED_TEXTURE(name) initializeTexture(name, #name)


void cudaInitializeKernels() {
  CUDA_INITIALIZE_SHARED_TEXTURE(positions4);
  CUDA_INITIALIZE_SHARED_TEXTURE(colors4);
}
