#include "CudaGlobalVariables.h"


/*
surface<void, cudaSurfaceType2D> surfD;

void initializeTexture(surface<void, cudaSurfaceType2D>& surf, const std::string name) {
  auto glShared = GL_Shared::getInstance();

  // Positions4
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



void cudaSharedInitialize() {
  CUDA_INITIALIZE_SHARED_TEXTURE(surfD);
}
*/