#include "Collision.h"

Collision::Collision() {

  GLuint pos = glShared_.get_texture("positions4")->texture_;
  GLuint ppos = glShared_.get_texture("predictedPositions4")->texture_;
  
  std::cout << "pos = " << pos << std::endl;
  std::cout << "ppos = " << ppos << std::endl;

  cudaGraphicsResource* positions4_;
  /*
  CUDA(cudaGraphicsGLRegisterImage(&positions4_, 
                                   pos, 
                                   GL_TEXTURE_2D, 
                                   cudaGraphicsRegisterFlagsSurfaceLoadStore));
  */

  CUDA(cudaGraphicsGLRegisterImage(&positions4_,
                                   pos,
                                   GL_TEXTURE_2D,
                                   cudaGraphicsMapFlagsNone));

  // cudaGraphicsMapFlagsNone
  // cudaGraphicsGLRegisterBuffer 

  cudaStream_t cudaStream;
  CUDA(cudaStreamCreate(&cudaStream));
  CUDA(cudaGraphicsMapResources(1, &positions4_, cudaStream));

  float4 *positions = nullptr;
  size_t size;
  //CUDA(cudaGraphicsResourceGetMappedPointer((void **)(&positions), &size, positions4_));
  //std::cout << "Size: " << size << std::endl;

  

  cudaGraphicsMapResources(1, &positions4_, 0);

  cudaArray* array;
  cudaGraphicsSubResourceGetMappedArray(&array, positions4_, 0, 0);

  cudaGraphicsUnmapResources(1, &positions4_, 0);

  dim3 blocks(1, 1, 1);
  dim3 threads(2, 2, 2);

  //addKernel<<<blocks, threads>>>();

  CUDA(cudaGraphicsUnmapResources(1, &positions4_, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));

}


void Collision::compute() {
  //std::cout << "Collision compute" << std::endl;
}