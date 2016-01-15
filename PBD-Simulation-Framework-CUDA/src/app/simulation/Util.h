#ifndef UTIL_H
#define UTIL_H

#include <iostream>
#include <string>
#include <limits>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"


template<typename T>
void initializeSharedBuffer(T& buffer, const std::string name) {
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
}


template<typename T>
void initializeSharedTexture(T& surface, const std::string name) {
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

  CUDA(cudaBindSurfaceToArray(surface, array));

  CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));
}


#endif // UTIL_H