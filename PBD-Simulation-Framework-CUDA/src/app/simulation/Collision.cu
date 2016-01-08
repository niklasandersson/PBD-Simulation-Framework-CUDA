#include "Collision.h"

surface<void, cudaSurfaceType2D> surfD;
texture<float4, 2, cudaReadModeElementType> texRef;

__global__ void testKernel(const unsigned int numberOfParticles,
                           const unsigned int textureWidth) {

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = idx % textureWidth;
  const unsigned int y = idx / textureWidth;

  //printf("xy: %i, %i\n", x, y);

  //float2 tvo = make_float2(1.0f, 2.0f);
  //printf("two: %f, %f\n", tvo.x, tvo.y);
  //printf("length of two: %f\n", length(tvo));

  float4 elementRead = make_float4(5.0f, 5.0f, 5.0f, 5.0f);
  surf2Dread(&elementRead, surfD, x * sizeof(float4), y);

  //__syncthreads();
  //printf("elementRead: %f, %f, %f\n", elementRead.x, elementRead.y, elementRead.z);

  float4 elementWrite = make_float4(1337.0f, 1337.0f, 1337.0f, 1337.0f);

  elementWrite = elementRead;
  elementWrite.y = elementWrite.y - 0.01f;
  surf2Dwrite(elementWrite, surfD, x * sizeof(float4), y);

  //float4 elementRead2 = make_float4(3.0f, 3.0f, 3.0f, 3.0f);
  //surf2Dread(&elementRead2, surfD, 0, 0);
  //printf("elementRead2: %f, %f, %f\n", elementRead2.x, elementRead2.y, elementRead2.z);




  /*
  

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

  auto numberOfParticles = glShared_.get_unsigned_int_value("numberOfParticles");
  unsigned int textureWidth = glShared_.get_texture("positions4")->width_;

  //std::cout << "numberOfParticles: " << *numberOfParticles << std::endl;

  dim3 blocks((*numberOfParticles)/128, 1, 1);
  dim3 threads(128, 1, 1);

  testKernel<<<blocks, threads>>>(*numberOfParticles, 
                                  textureWidth);
}