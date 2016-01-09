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

void thrustSort(double V[], int P[], int N)
{
  thrust::device_vector<int> d_P(N);
  thrust::sequence(d_P.begin(), d_P.end());

  thrust::device_vector<double> d_V(V, V + N);
  

  thrust::sort_by_key(d_V.begin(), d_V.end(), d_P.begin());

  thrust::copy(d_P.begin(), d_P.end(), P);
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

  /*
  const int N = 4;
  thrust::device_vector<int> indices(N);
  thrust::sequence(indices.begin(), indices.end());

  int keysArr[N] = { 4, 3, 2, 1 };
  thrust::device_vector<int> keys(keysArr, keysArr + N);


  thrust::sort_by_key(keys.begin(), keys.end(), indices.begin());

  int vals0Arr[N] = { 1, 2, 3, 4 };
  thrust::device_vector<int> vals0(vals0Arr, vals0Arr + N);
  int vals1Arr[N] = { 5, 6, 7, 8 };
  thrust::device_vector<int> vals1(vals1Arr, vals1Arr + N);

  thrust::device_vector<int> temp(N);
  thrust::device_vector<int>* sorted = &temp;

  thrust::device_vector<int>* pVals0 = &vals0;
  thrust::device_vector<int>* pVals1 = &vals1;

  thrust::gather(indices.begin(), indices.end(), vals0.begin(), sorted->begin());
  pVals0 = sorted; sorted = &vals0;

  thrust::gather(indices.begin(), indices.end(), vals1.begin(), sorted->begin());
  pVals1 = sorted; sorted = &vals1;
  */

/*
  
  const unsigned int N = 4;

  unsigned int* h_ind = new unsigned int[N]{ 0, 1, 2, 3 };
  unsigned int* h_keys = new unsigned int[N]{ 4, 2, 3, 1 }; 
  unsigned int* h_vals0 = new unsigned int[N]{ 1, 2, 3, 4 };
  unsigned int* h_vals1 = new unsigned int[N]{ 5, 2, 3, 6 };

  unsigned int* d_ind;
  unsigned int* d_keys;
  unsigned int* d_vals0;
  unsigned int* d_vals1;

  cudaMalloc((void**)&d_ind, N * sizeof(unsigned int));
  cudaMalloc((void**)&d_keys, N * sizeof(unsigned int));
  cudaMalloc((void**)&d_vals0, N * sizeof(unsigned int));
  cudaMalloc((void**)&d_vals1, N * sizeof(unsigned int));

  CUDA(cudaMemcpy(d_ind, h_ind, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(d_keys, h_keys, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(d_vals0, h_vals0, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(d_vals1, h_vals1, N * sizeof(unsigned int), cudaMemcpyHostToDevice));
 
  cudaDeviceSynchronize();

  cudaMemcpy(h_keys, d_keys, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ind, d_ind, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
 
  std::cout << "h_keys before: ";
  for(unsigned int i=0; i<N; i++) {
    std::cout << h_keys[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "h_ind before: ";
  for(unsigned int i=0; i<N; i++) {
    std::cout << h_ind[i] << " ";
  }
  std::cout << std::endl;

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  
  thrust::device_ptr<unsigned int> indices(d_ind);
  thrust::device_ptr<unsigned int> keys(d_keys);
  thrust::device_ptr<unsigned int> vals0(d_vals0);
  thrust::device_ptr<unsigned int> vals1(d_vals1);

  // allocate space for the output
  thrust::device_vector<unsigned int> sortedVals0(N);
  thrust::device_vector<unsigned int> sortedVals1(N);

  cudaDeviceSynchronize();

  //thrust::sort_by_key(keys, keys, indices);
  //cudaThreadSynchronize();
  thrust::sort_by_key(keys, keys + N, indices);
  //std::cout << "SORTED!" << std::endl;
  */

  /*
  thrust::host_vector<int> h_keys2(N);
  h_keys2[0] = 4;
  h_keys2[1] = 1;
  h_keys2[2] = 2;
  h_keys2[3] = 0;
  
  thrust::host_vector<int> h_indices2(N);
  h_indices2[0] = 0;
  h_indices2[1] = 1;
  h_indices2[2] = 2;
  h_indices2[3] = 3;




  thrust::device_vector<int> keys2(N);
  keys2[0] = 4;
  keys2[1] = 1;
  keys2[2] = 2;
  keys2[3] = 0;
  thrust::device_vector<int> indices2(N);
  indices2[0] = 0;
  indices2[1] = 1;
  indices2[2] = 2;
  indices2[3] = 3;

  thrust::device_ptr<int> keys3 = &keys2[0];
  thrust::device_ptr<int> indices3 = &indices2[0];

	thrust::sort_by_key(keys3, keys3 + N, indices3);

  */


  //thrust::sort_by_key(keys, keys + N , indices);

  /*
  const int N2 = 6;
  int    keys22[N2] = {  1,   4,   2,   8,   5,   7};
  char values22[N2] = {'a', 'b', 'c', 'd', 'e', 'f'};
  thrust::sort_by_key(keys22, keys22 + N2, values22);
  */

  /*
  // first sort the keys and indices by the keys
  thrust::host_vector<int> h_keys2(N);
  h_keys2[0] = 4;
  h_keys2[1] = 1;
  h_keys2[2] = 2;
  h_keys2[3] = 0;
  
  thrust::host_vector<int> h_indices2(N);
  h_indices2[0] = 0;
  h_indices2[1] = 1;
  h_indices2[2] = 2;
  h_indices2[3] = 3;

  thrust::device_vector<int> keys2 = h_keys2;
  thrust::device_vector<int> indices2 = h_indices2;

	thrust::sort_by_key(keys2.begin(), keys2.end(), 
                      indices2.begin(), thrust::greater<int>());
  */

   /*
  // Now reorder the ID arrays using the sorted indices
  thrust::gather(indices.begin(), indices.end(), vals0, sortedVals0.begin());
  thrust::gather(indices.begin(), indices.end(), vals1, sortedVals1.begin());

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Took %f milliseconds for %d elements\n", milliseconds, N);
     
  cudaDeviceSynchronize();

  cudaMemcpy(h_keys, d_keys, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy(h_ind, d_ind, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
 
  std::cout << "h_keys sorted: ";
  for(unsigned int i=0; i<N; i++) {
    std::cout << h_keys[i] << " ";
  }
  std::cout << std::endl;
  
  std::cout << "h_ind sorted: ";
  for(unsigned int i=0; i<N; i++) {
    std::cout << h_ind[i] << " ";
  }
  std::cout << std::endl;
  */

  //unsigned int * raw_ptr = thrust::raw_pointer_cast(dev_data_ptr);

  
  /*
  thrust::device_vector<int>  indices(N);
  thrust::sequence(indices.begin(), indices.end());
  thrust::sort_by_key(keys.begin(), keys.end(), indices.begin());

  thrust::device_vector<int> temp(N);
  thrust::device_vector<int> *sorted = &temp;
  thrust::device_vector<int> *pa_01 = &a_01;
  thrust::device_vector<int> *pa_02 = &a_02;
  
   thrust::device_vector<int> *pa_20 = &a_20;

  thrust::gather(indices.begin(), indices.end(), *pa_01, *sorted);
  pa_01 = sorted; sorted = &a_01;
  thrust::gather(indices.begin(), indices.end(), *pa_02, *sorted);
  pa_02 = sorted; sorted = &a_02;
  
  thrust::gather(indices.begin(), indices.end(), *pa_20, *sorted);
  pa_20 = sorted; sorted = &a_20;
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