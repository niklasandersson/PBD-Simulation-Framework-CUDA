#include "Kernels.h"
#include "cub/cub.cuh"


surface<void, cudaSurfaceType2D> positions4;
surface<void, cudaSurfaceType2D> predictedPositions4;
surface<void, cudaSurfaceType2D> velocities4;
surface<void, cudaSurfaceType2D> colors4;

surface<void, cudaSurfaceType2D> positions4Copy;
surface<void, cudaSurfaceType2D> predictedPositions4Copy;
surface<void, cudaSurfaceType2D> velocities4Copy;
surface<void, cudaSurfaceType2D> colors4Copy;

unsigned int* d_cellIds_in;
unsigned int* d_cellIds_out;

unsigned int* d_particleIds_in;
unsigned int* d_particleIds_out;

void* d_sortTempStorage = nullptr;
size_t sortTempStorageBytes = 0;

unsigned int* d_cellStarts;
unsigned int* d_cellEndings;

unsigned int* d_contacts;
unsigned int* d_contactCounters;
int* d_contactConstraintSucces;
int* d_contactConstraintParticleUsed;

float* densities;

const float deltaT = 0.01f;

// --------------------------------------------------------------------------

__global__ void applyForces(const unsigned int numberOfParticles,
                            const unsigned int textureWidth,
                            const float deltaT) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  if( idx < numberOfParticles ) {
    const float inverseMass = 1.0f;
    const float gravity = -9.82;

    float4 velocity;
    surf2Dread(&velocity, velocities4, x, y);
    velocity.y += inverseMass * gravity * deltaT; 

    float4 position;
    surf2Dread(&position, positions4, x, y);

    float4 predictedPosition = position + velocity * deltaT;
    surf2Dwrite(predictedPosition, predictedPositions4, x, y);
  }
}

void cudaCallApplyForces() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  applyForces<<<blocks, threads>>>(numberOfParticles, textureWidth, deltaT);
}

// --------------------------------------------------------------------------

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ __forceinline__ unsigned int expandBits(unsigned int v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

// Calculates a 30-bit Morton code for the
// given 3D point located within the cube [0.0, 1023.0].
__device__ __forceinline__ unsigned int mortonCode(float4 pos) {
    pos.x = min(max(pos.x, 0.0f), 1023.0f);
    pos.y = min(max(pos.y, 0.0f), 1023.0f);
    pos.z = min(max(pos.z, 0.0f), 1023.0f);
    // x = min(max(x * 1024.0f, 0.0f), 1023.0f);
    // y = min(max(y * 1024.0f, 0.0f), 1023.0f);
    // z = min(max(z * 1024.0f, 0.0f), 1023.0f);
    const unsigned int xx = expandBits((unsigned int)pos.x) << 2;
    const unsigned int yy = expandBits((unsigned int)pos.y) << 1;
    const unsigned int zz = expandBits((unsigned int)pos.z);
    //return xx * 4 + yy * 2 + zz;
    return xx + yy + zz;
}

__global__ void initializeCellIds(const unsigned int numberOfParticles,
                                  const unsigned int textureWidth,
                                  unsigned int* cellIdsIn) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < numberOfParticles ) {
    float4 predictedPosition;
    surf2Dread(&predictedPosition, predictedPositions4, x, y);
    cellIdsIn[idx] = mortonCode(predictedPosition);
  } else {
    cellIdsIn[idx] = UINT_MAX;
  }
}

void cudaCallInitializeCellIds() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  initializeCellIds<<<blocks, threads>>>(numberOfParticles, textureWidth, d_cellIds_in);
}

// --------------------------------------------------------------------------

void sortIds() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  
  cub::DeviceRadixSort::SortPairs(d_sortTempStorage, 
                                  sortTempStorageBytes, 
                                  d_cellIds_in, 
                                  d_cellIds_out, 
                                  d_particleIds_in, 
                                  d_particleIds_out, 
                                  numberOfParticles);
}

// --------------------------------------------------------------------------

__global__ void copy(const unsigned int numberOfParticles,
                     const unsigned int textureWidth) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
        
  if( idx < numberOfParticles ) {
    float4 data;
    surf2Dread(&data, positions4, x, y);
    surf2Dwrite(data, positions4Copy, x, y);

    surf2Dread(&data, predictedPositions4, x, y);
    surf2Dwrite(data, predictedPositions4Copy, x, y);

    surf2Dread(&data, velocities4, x, y);
    surf2Dwrite(data, velocities4Copy, x, y);

    surf2Dread(&data, colors4, x, y);
    surf2Dwrite(data, colors4Copy, x, y);
  } 
}

void cudaCallCopy() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  
  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  copy<<<blocks, threads>>>(numberOfParticles, textureWidth);
}

// --------------------------------------------------------------------------

__global__ void reorder(const unsigned int numberOfParticles,
                        const unsigned int textureWidth,
                        unsigned int* cellIdsOut,
                        unsigned int* particleIdsOut) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  if( idx < numberOfParticles ) {
    const unsigned int cellId = cellIdsOut[idx];
    const unsigned int particleId = particleIdsOut[idx];

    //const unsigned int cellIdReadX = (cellId % textureWidth) * sizeof(float4);
    //const unsigned int cellIdReadY = cellId / textureWidth;

    const unsigned int particleIdX = (particleId % textureWidth) * sizeof(float4);
    const unsigned int particleIdY = particleId / textureWidth;

    
    float4 data;
    surf2Dread(&data, positions4Copy, particleIdX, particleIdY);
    surf2Dwrite(data, positions4, x, y);

    surf2Dread(&data, predictedPositions4Copy, particleIdX, particleIdY);
    surf2Dwrite(data, predictedPositions4, x, y);

    surf2Dread(&data, velocities4Copy, particleIdX, particleIdY);
    surf2Dwrite(data, velocities4, x, y);

    surf2Dread(&data, colors4Copy, particleIdX, particleIdY);
    surf2Dwrite(data, colors4, x, y);
  } 
}

void cudaCallReorder() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  
  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  reorder<<<blocks, threads>>>(numberOfParticles, textureWidth, d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

void reorderStorage() {
  cudaCallCopy();
  cudaCallReorder();
}

// --------------------------------------------------------------------------

__global__ void resetCellInfo(const unsigned int numberOfParticles,
                                const unsigned int textureWidth,
                                unsigned int* cellStarts,
                                unsigned int* cellEndings) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  cellStarts[idx] = UINT_MAX;
  cellEndings[idx] = numberOfParticles;
}

void cudaCallResetCellInfo() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int maxGrid = *GL_Shared::getInstance().get_unsigned_int_value("maxGrid");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  
  const dim3 blocks((maxGrid)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  resetCellInfo<<<blocks, threads>>>(numberOfParticles, textureWidth, d_cellStarts, d_cellEndings);
}

// --------------------------------------------------------------------------

__global__ void computeCellInfo(const unsigned int numberOfParticles,
                                const unsigned int textureWidth,
                                unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* cellIdsOut,
                                unsigned int* particleIdsOut)  {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  const unsigned int cellId = cellIdsOut[idx];
  const unsigned int particleId = particleIdsOut[idx];

  if( idx < numberOfParticles ) {
    if( idx == 0 ) {
      cellStarts[cellId] = 0; 
    } else {
      const unsigned int previousCellId = cellIdsOut[idx-1];
      if( previousCellId != cellId ) {
        cellStarts[cellId] = idx;
        cellEndings[previousCellId] = idx;
      }
    }
  }
}

void cudaCallComputeCellInfo() {
  auto glShared = GL_Shared::getInstance();
  auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  
  const dim3 blocks((*numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  computeCellInfo<<<blocks, threads>>>(*numberOfParticles, textureWidth, d_cellStarts, d_cellEndings, d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

__global__ void updatePositions(const unsigned int numberOfParticles,
                                const unsigned int textureWidth,
                                const float deltaT) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < numberOfParticles ) {
    float4 position;
    surf2Dread(&position, positions4, x, y);

    float4 predictedPosition;
    surf2Dread(&predictedPosition, predictedPositions4, x, y);

    float4 velocity = (predictedPosition - position) / deltaT;

    surf2Dwrite(predictedPosition, positions4, x, y);
    surf2Dwrite(velocity, velocities4, x, y);
  }
}

void cudaCallUpdatePositions() {
  auto glShared = GL_Shared::getInstance();
  const auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

  const dim3 blocks((*numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  updatePositions<<<blocks, threads>>>(*numberOfParticles, textureWidth, deltaT);
}

// --------------------------------------------------------------------------

__global__ void initializeParticleIds(const unsigned int numberOfParticles,
                                      const unsigned int textureWidth,
                                      unsigned int* particleIdsIn) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < numberOfParticles ) {
    particleIdsIn[idx] = idx;
  }
}

void cudaCallInitializeParticleIds() {
  auto glShared = GL_Shared::getInstance();
  auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;

  const dim3 blocks((*numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  initializeParticleIds<<<blocks, threads>>>(*numberOfParticles, textureWidth, d_particleIds_in);
}

// --------------------------------------------------------------------------

void initializeSort() {
  const unsigned int maxParticles = *GL_Shared::getInstance().get_unsigned_int_value("maxParticles");
  CUDA(cudaMalloc((void**)&d_cellIds_in, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_cellIds_out, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_particleIds_in, maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_particleIds_out, maxParticles * sizeof(unsigned int)));

  cudaCallInitializeParticleIds();
  
  cub::DeviceRadixSort::SortPairs(d_sortTempStorage, 
                                  sortTempStorageBytes,
		                              d_cellIds_in, 
                                  d_cellIds_out, 
                                  d_particleIds_in, 
                                  d_particleIds_out,
                                  maxParticles);

  CUDA(cudaMalloc(&d_sortTempStorage, sortTempStorageBytes));
}

// --------------------------------------------------------------------------

void initializeCellInfo() {
  const unsigned int maxGrid = *GL_Shared::getInstance().get_unsigned_int_value("maxGrid");
  CUDA(cudaMalloc((void**)&d_cellStarts, maxGrid * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_cellEndings, maxGrid * sizeof(unsigned int)));
}

// --------------------------------------------------------------------------

void initializeCollision() {
  const unsigned int maxParticles = *GL_Shared::getInstance().get_unsigned_int_value("maxParticles");
  const unsigned int maxContactConstraints = 12 * maxParticles;
  CUDA(cudaMalloc((void**)&d_contacts, maxContactConstraints * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactCounters, maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintSucces, maxContactConstraints * sizeof(int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintParticleUsed, maxParticles * sizeof(int)));
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

// --------------------------------------------------------------------------
/*
void initializeBuffer(float* buffer, const std::string name) {
  auto glShared = GL_Shared::getInstance();
  GLuint gluint = glShared.get_buffer(name)->buffer_;

  cudaStream_t cudaStream;
  CUDA(cudaStreamCreate(&cudaStream));

  cudaGraphicsResource* resource;
  CUDA(cudaGraphicsGLRegisterBuffer(&resource, gluint, cudaGraphicsMapFlagsNone));

  CUDA(cudaGraphicsMapResources(1, &resource, cudaStream));
 
  size_t size;
  CUDA(cudaGraphicsResourceGetMappedPointer((void**)&densities, &size, resource));

  CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream));
  CUDA(cudaStreamDestroy(cudaStream));
} */
//#define CUDA_INITIALIZE_SHARED_BUFFER(name) initializeBuffer(name, #name)

#define CUDA_INITIALIZE_SHARED_BUFFER(name) \
  [&]{ \
  auto glShared = GL_Shared::getInstance(); \
  GLuint gluint = glShared.get_buffer(#name)->buffer_; \
  \
  cudaStream_t cudaStream; \
  CUDA(cudaStreamCreate(&cudaStream)); \
  \
  cudaGraphicsResource* resource; \
  CUDA(cudaGraphicsGLRegisterBuffer(&resource, gluint, cudaGraphicsMapFlagsNone)); \
  \
  CUDA(cudaGraphicsMapResources(1, &resource, cudaStream)); \
  \
  size_t size; \
  CUDA(cudaGraphicsResourceGetMappedPointer((void**)&name, &size, resource)); \
  \
  CUDA(cudaGraphicsUnmapResources(1, &resource, cudaStream)); \
  CUDA(cudaStreamDestroy(cudaStream)); \
  }()

// --------------------------------------------------------------------------

void cudaInitializeKernels() {
  CUDA_INITIALIZE_SHARED_TEXTURE(positions4);
  CUDA_INITIALIZE_SHARED_TEXTURE(predictedPositions4);
  CUDA_INITIALIZE_SHARED_TEXTURE(velocities4);
  CUDA_INITIALIZE_SHARED_TEXTURE(colors4);

  CUDA_INITIALIZE_SHARED_TEXTURE(positions4Copy);
  CUDA_INITIALIZE_SHARED_TEXTURE(predictedPositions4Copy);
  CUDA_INITIALIZE_SHARED_TEXTURE(velocities4Copy);
  CUDA_INITIALIZE_SHARED_TEXTURE(colors4Copy);

  CUDA_INITIALIZE_SHARED_BUFFER(densities);

  initializeSort();
  initializeCellInfo();
  initializeCollision();
}
