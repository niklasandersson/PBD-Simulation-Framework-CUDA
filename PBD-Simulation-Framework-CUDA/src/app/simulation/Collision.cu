#include "Collision.h"

#include <thrust\device_vector.h>

surface<void, cudaSurfaceType2D> surfD;
texture<float4, 2, cudaReadModeElementType> texRef;

__global__ void testKernel() {

  float4 elementRead = make_float4(5.0f, 5.0f, 5.0f, 5.0f);
  surf2Dread(&elementRead, surfD, 0, 0);
  //printf("elementRead: %f, %f, %f\n", elementRead.x, elementRead.y, elementRead.z);

  float4 elementWrite = make_float4(1337.0f, 1337.0f, 1337.0f, 1337.0f);

  elementWrite = elementRead;
  elementWrite.y = elementWrite.y - 0.01f;
  surf2Dwrite(elementWrite, surfD, 0, 0);

  //float4 elementRead2 = make_float4(3.0f, 3.0f, 3.0f, 3.0f);
  //surf2Dread(&elementRead2, surfD, 0, 0);
  //printf("elementRead2: %f, %f, %f\n", elementRead2.x, elementRead2.y, elementRead2.z);




  /*
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  int offset = x + y * blockDim.x * gridDim.x;

  float4 value = tex2D(texRef, 0, 0);

  printf("value.x: %f\n", value.x);
  printf("value.y: %f\n", value.y);
  printf("value.z: %f\n", value.z);
  printf("value.w: %f\n", value.w);

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  */

}

Collision::Collision() {

  GLuint pos = glShared_.get_texture("positions4")->texture_;
  GLuint ppos = glShared_.get_texture("predictedPositions4")->texture_;
  GLuint cols = glShared_.get_texture("colors4")->texture_;

  std::cout << "pos = " << pos << std::endl;
  std::cout << "ppos = " << ppos << std::endl;

  
  cudaStream_t cudaStream;
  CUDA(cudaStreamCreate(&cudaStream));

  cudaGraphicsResource* positions4_;
  CUDA(cudaGraphicsGLRegisterImage(&positions4_,
                                   pos,
                                   GL_TEXTURE_2D,
                                   cudaGraphicsRegisterFlagsSurfaceLoadStore));

  CUDA(cudaGraphicsMapResources(1, &positions4_, cudaStream));

  cudaArray* array;
  CUDA(cudaGraphicsSubResourceGetMappedArray(&array, positions4_, 0, 0));

  CUDA(cudaBindSurfaceToArray(surfD, array));
  //testKernel << <1, 1 >> >();
  

  
  //CUDA(cudaBindTextureToArray(texRef, array));
  //texRef.filterMode = cudaFilterModePoint;

  


  CUDA(cudaGraphicsUnmapResources(1, &positions4_, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));








  /*
  cudaGraphicsResource* positions4_;
  
  CUDA(cudaGraphicsGLRegisterImage(&positions4_, 
                                   pos, 
                                   GL_TEXTURE_2D, 
                                   cudaGraphicsRegisterFlagsSurfaceLoadStore));
  

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

  // cudaGraphicsResourceGetMappedPointer <- for buffers
  // cudaGraphicsSubResourceGetMappedArray <- for textures
  //CUDA(cudaGraphicsResourceGetMappedPointer((void **)(&positions), &size, positions4_));
  //std::cout << "Size: " << size << std::endl5;



  cudaArray* array;
  CUDA(cudaGraphicsSubResourceGetMappedArray(&array, positions4_, 0, 0));


  // create the CUDA texture reference
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
  texture<float4, 2, cudaReadModeElementType> tex;
  tex.addressMode[0] = cudaAddressModeClamp;
  tex.addressMode[1] = cudaAddressModeClamp;
  tex.filterMode = cudaFilterModePoint;


  // bind the CUDA array to a texture object (THIS is where the error happens)
  // CUDA(cudaBindTextureToArray(tex, array, channelDesc));

  // Create the surface object
  cudaSurfaceObject_t surfaceWrite = 0;

  CUDA(cudaBindSurfaceToArray(surfaceWrite, array));

  //cudaDestroyTextureObject(tex);
 
  dim3 blocks(1, 1, 1);
  dim3 threads(2, 2, 2);


  addKernel << <blocks, threads >> >();

  
  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = cudaReadModeElementType;


  // create texture object: we only have to do this once!
  //cudaTextureObject_t tex = 0;
  //cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL);

  //texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

  //texture<float, cudaTextureType2D, cudaReadModeElementType > tex;

 // cudaBindTextureToArray(&tex, array, &texDesc);





  CUDA(cudaGraphicsUnmapResources(1, &positions4_, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));
  */
}


void Collision::compute() {
  //std::cout << "Collision compute" << std::endl;
  testKernel << <1, 1 >> >();
}