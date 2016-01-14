#include "Util.h"


size_t initializeSharedBuffer(void* buffer, const std::string name) {
  auto glShared = GL_Shared::getInstance();
  GLuint gluint = glShared.get_buffer(name)->buffer_;

  cudaStream_t cudaStream;
  CUDA(cudaStreamCreate(&cudaStream));

  cudaGraphicsResource* resource;
  CUDA(cudaGraphicsGLRegisterBuffer(&resource, gluint, cudaGraphicsMapFlagsNone));

  CUDA(cudaGraphicsMapResources(1, &resource, cudaStream));

  size_t size;
  CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffer, &size, resource));

  CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));

  return size;
}

/*
void initializeSharedTexture(void* surf, const std::string name) {
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

  CUDA(cudaBindSurfaceToArray((surface<void, cudaSurfaceType2D>&)surf, array));

  CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));
}
*/