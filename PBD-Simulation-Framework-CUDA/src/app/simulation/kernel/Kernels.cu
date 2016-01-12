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
const float particleRadius = 0.5f;
const float particleDiameter = 2.0f * particleRadius;
const unsigned int maxContactsPerCell = 12;

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
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  
  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  computeCellInfo<<<blocks, threads>>>(numberOfParticles, textureWidth, d_cellStarts, d_cellEndings, d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

__global__ void resetContacts(unsigned int* contacts, 
                              const unsigned int numberOfContacts) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfContacts ) {
    contacts[idx] = UINT_MAX;
  }
}

void cudaCallResetContacts() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int numberOfContacts = maxContactsPerCell * numberOfParticles;

  const dim3 blocks((numberOfContacts)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  resetContacts<<<blocks, threads>>>(d_contacts, numberOfContacts);
}

// --------------------------------------------------------------------------

__global__ void findContacts(const unsigned int numberOfParticles,
                             const unsigned int textureWidth,
                             const unsigned int maxGrid,
                             const unsigned int maxContactsPerCell,
                             const float particleDiameter,
                             unsigned int* cellStarts,
                             unsigned int* contacts,
                             unsigned int* cellIds,
                             unsigned int* contactCounters,
                             float* densities) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  contacts += maxContactsPerCell * idx;
  unsigned int counter = 0;
  float numberOfCloseNeighours = 0.0f;

  if( idx < numberOfParticles ) {
    float4 predictedPosition;
    surf2Dread(&predictedPosition, predictedPositions4, x, y);
    predictedPosition.w = 0.0f;

    const int searchWidth = 1;
    float4 tempPos;
    unsigned int morton;
    unsigned int start;
    unsigned int index;
    unsigned int x2;
    unsigned int y2;
    float distance;
    float4 otherPredictedPosition; 
    for(int i=-searchWidth; i<=searchWidth; i++) {
      for(int j=-searchWidth; j<=searchWidth; j++) {
        for(int k=-searchWidth; k<=searchWidth; k++) {
          tempPos = predictedPosition;
          tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
          morton = mortonCode(tempPos);
          if( morton < maxGrid ) {
            start = cellStarts[morton]; 
            if( start != UINT_MAX ) {
              index = start;
              do {
                numberOfCloseNeighours += 1.0f;
                if( idx != index && index < numberOfParticles ) {
                  x2 = (index % textureWidth) * sizeof(float4);
                  y2 = index / textureWidth;
                  surf2Dread(&otherPredictedPosition, predictedPositions4, x2, y2);
                  otherPredictedPosition.w = 0.0f;
                  distance = length(predictedPosition - otherPredictedPosition);
                  if( distance < particleDiameter ) {
                    contacts[counter++] = index;
                  } 
                }
              } while( cellIds[++index] == morton && counter < maxContactsPerCell );
            }
          }

        }
      }
    }

  }
  contactCounters[idx] = counter;
  densities[idx] = numberOfCloseNeighours / 26.0f;

}

void cudaCallFindContacts() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int maxGrid = *glShared.get_unsigned_int_value("maxGrid");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  
  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  findContacts<<<blocks, threads>>>(numberOfParticles, textureWidth, maxGrid, maxContactsPerCell, particleDiameter, d_cellStarts, d_contacts , d_cellIds_out, d_contactCounters, densities);

}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintSuccess(const unsigned int maxContactConstraints,
                                              int* contactConstraintSucces) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contactConstraintSucces[idx] = -1;
  } 
}

void cudaCallResetContactConstraintSuccess() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int maxContactConstraints = maxContactsPerCell * numberOfParticles;

  const dim3 blocks((maxContactConstraints)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  resetContactConstraintSuccess<<<blocks, threads>>>(maxContactConstraints, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintParticleUsed(const unsigned int numberOfParticles,
                                                   int* contactConstraintParticleUsed) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfParticles  ) {
    contactConstraintParticleUsed[idx] = -1;
  }
}

void cudaCallResetContactConstraintParticleUsed() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");

  const dim3 blocks((numberOfParticles)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  resetContactConstraintParticleUsed<<<blocks, threads>>>(numberOfParticles, d_contactConstraintParticleUsed);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatches(const unsigned int numberOfContactConstraints,
                                                const unsigned int maxContactsPerCell,
                                                unsigned int* contacts,
                                                int* particleUsed,
                                                int* constraintSucces) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfContactConstraints ) {
    const int success = constraintSucces[idx];
    if( success < 0 ) { // If not success
      const unsigned int particle1 = idx / maxContactsPerCell;
      const unsigned int particle2 = contacts[idx];
      if( particle2 != UINT_MAX ) {
        const unsigned int localId = threadIdx.x;
        const unsigned int localWorkSize = blockDim.x;
        for(unsigned int i=0; i<localWorkSize; i++) {
          if( (i == localId) && (particleUsed[particle1] < 0) && (particleUsed[particle2] < 0) ) {
            if( particleUsed[particle1] == -1 ) {
              particleUsed[particle1] = idx;
            }
            if( particleUsed[particle2] == -1 ) {
              particleUsed[particle2] = idx;
            }
          }
          __syncthreads();
        }
      }

    }
  }
}

void cudaCallSetupCollisionConstraintBatches() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int maxContactConstraints = maxContactsPerCell * numberOfParticles;

  const dim3 blocks((maxContactConstraints)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  setupCollisionConstraintBatches<<<blocks, threads>>>(maxContactConstraints, maxContactsPerCell, d_contacts, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatchesCheck(const unsigned int numberOfContactConstraints,
                                                     const unsigned int maxContactsPerCell,
                                                     const unsigned int textureWidth,
                                                     const float particleDiameter,
                                                     unsigned int* contacts,
                                                     int* particleUsed,
                                                     int* constraintSucces) {
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfContactConstraints ) {
    const unsigned int particle1 = idx / maxContactsPerCell;
    const unsigned int particle2 = contacts[idx];
    if( particle2 != UINT_MAX ) {
       if( (particleUsed[particle1] == idx) && (particleUsed[particle2] == idx) ) {
        constraintSucces[idx] = 1;

        // Solve constraint for particle1 and particle2
        const unsigned int x1 = (particle1 % textureWidth) * sizeof(float4);
        const unsigned int y1 = particle1 / textureWidth;
        const unsigned int x2 = (particle2 % textureWidth) * sizeof(float4);
        const unsigned int y2 = particle2 / textureWidth;

        float4 predictedPosition1;
        surf2Dread(&predictedPosition1, predictedPositions4, x1, y1);
        predictedPosition1.w = 0.0f;  

        float4 predictedPosition2;
        surf2Dread(&predictedPosition2, predictedPositions4, x2, y2);
        predictedPosition2.w = 0.0f;

        const float distance = length(predictedPosition2 - predictedPosition1);
        const float overlap = particleDiameter - distance;

        if( overlap > 0 ) {
          const float4 pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 
          const float halfOverlap = overlap / 2.0f;

          const float4 addTo1 = -1.0 * pos1ToPos2 * halfOverlap;
          const float4 addTo2 = pos1ToPos2 * halfOverlap;

          predictedPosition1 += addTo1;
          predictedPosition2 += addTo2;

          surf2Dwrite(predictedPosition1, predictedPositions4, x1, y1);
          surf2Dwrite(predictedPosition2, predictedPositions4, x2, y2);

          float4 position1;
          surf2Dread(&position1, positions4, x1, y1);
       
          float4 position2;
          surf2Dread(&position2, positions4, x2, y2);

          position1 += addTo1;
          position2 += addTo2;

          surf2Dwrite(position1, positions4, x1, y1);
          surf2Dwrite(position2, positions4, x2, y2);
        }
        
      } 
    }
  }
}

void cudaCallSetupCollisionConstraintBatchesCheck() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  const unsigned int maxContactConstraints = maxContactsPerCell * numberOfParticles;

  const dim3 blocks((maxContactConstraints)/128, 1, 1);
  const dim3 threads(128, 1, 1);

  setupCollisionConstraintBatchesCheck<<<blocks, threads>>>(maxContactConstraints, maxContactsPerCell, textureWidth, particleDiameter, d_contacts, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

void solveCollisions() {
  cudaCallResetContacts();
  cudaCallFindContacts();
  cudaCallResetContactConstraintSuccess();
  const unsigned int maxBatches = maxContactsPerCell;
  for(unsigned int b=0; b<maxBatches; b++) {
    cudaCallResetContactConstraintParticleUsed();
    cudaCallSetupCollisionConstraintBatches();
    cudaCallSetupCollisionConstraintBatchesCheck();
  }
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

    if( predictedPosition.y < 1.5f ) {
      predictedPosition.y = 1.5f;
    }

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
  const unsigned int maxContactConstraints = maxContactsPerCell * maxParticles;
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

#define CUDA_INITIALIZE_SHARED_BUFFER(name) \
  [&]{ \
  auto glShared = GL_Shared::getInstance(); \
  GLuint gluint = glShared.get_buffer(#name)->buffer_; \
  cudaStream_t cudaStream; \
  CUDA(cudaStreamCreate(&cudaStream)); \
  cudaGraphicsResource* resource; \
  CUDA(cudaGraphicsGLRegisterBuffer(&resource, gluint, cudaGraphicsMapFlagsNone)); \
  CUDA(cudaGraphicsMapResources(1, &resource, cudaStream)); \
  size_t size; \
  CUDA(cudaGraphicsResourceGetMappedPointer((void**)&name, &size, resource)); \
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
