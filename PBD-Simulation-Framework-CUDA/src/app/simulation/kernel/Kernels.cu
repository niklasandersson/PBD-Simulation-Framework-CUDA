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

  float* d_densities;
  /*
struct Buffers {
  float* d_densities;
  float4* d_positions;
  float4* d_predictedPositions;
  float4* d_velocities;
  float4* d_colors;

  float* d_densities;
  float4* d_positions;
  float4* d_predictedPositions;
  float4* d_velocities;
  float4* d_colors;
};

Buffers buffers;
*/

struct SimulationParameters {
  unsigned int numberOfParticles;
  unsigned int textureWidth;
  unsigned int maxContactsPerParticle;
  unsigned int maxContactConstraints;
  unsigned int maxGrid;
  unsigned int maxParticles;
  float particleRadius;
  float particleDiameter;
  float deltaT;
};

__constant__ SimulationParameters d_simulationParameters;

SimulationParameters simulationParameters;

struct CudaCallParameters {
  dim3 blocksForParticleBased;
  dim3 threadsForParticleBased;

  dim3 blocksForContactBased;
  dim3 threadsForContactBased;

  dim3 blocksForGridBased;
  dim3 threadsForGridBased;
};

CudaCallParameters cudaCallParameters;

// --------------------------------------------------------------------------

void initializeFrame() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  const unsigned int maxGrid = *glShared.get_unsigned_int_value("maxGrid");
  const unsigned int maxParticles = *glShared.get_unsigned_int_value("maxParticles");

  simulationParameters.numberOfParticles = numberOfParticles;
  simulationParameters.textureWidth = textureWidth;
  simulationParameters.maxContactsPerParticle = 12;
  simulationParameters.maxContactConstraints = simulationParameters.maxContactsPerParticle * simulationParameters.numberOfParticles;
  simulationParameters.maxGrid = maxGrid;
  simulationParameters.maxParticles = maxParticles;
  simulationParameters.deltaT = 0.01f;
  simulationParameters.particleRadius = 0.5f;
  simulationParameters.particleDiameter = 2.0f * simulationParameters.particleRadius;

  CUDA(cudaMemcpyToSymbol(d_simulationParameters, &simulationParameters, sizeof(SimulationParameters)));

  unsigned int threadsPerBlock = 128;
  cudaCallParameters.blocksForParticleBased = dim3((simulationParameters.numberOfParticles)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsForParticleBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksForContactBased = dim3((simulationParameters.maxContactConstraints)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsForContactBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksForGridBased = dim3((simulationParameters.maxGrid)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsForGridBased = dim3(threadsPerBlock, 1, 1);

  CUDA(cudaDeviceSynchronize());
}

// --------------------------------------------------------------------------

__global__ void applyForces() {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;
  const float deltaT = d_simulationParameters.deltaT;

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

    const float floorDiff = predictedPosition.y - 1.5;
    if( floorDiff < 0 ) {
      predictedPosition.y = predictedPosition.y + (-1.0f * floorDiff);
      position.y = position.y + (-1.0f * floorDiff);
      surf2Dwrite(position, positions4, x, y);
    }

    surf2Dwrite(predictedPosition, predictedPositions4, x, y);
  }
}

void cudaCallApplyForces() {
  applyForces<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>();
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

__global__ void initializeCellIds(unsigned int* cellIdsIn) {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

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
  initializeCellIds<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>(d_cellIds_in);
}

// --------------------------------------------------------------------------

void sortIds() {
  cub::DeviceRadixSort::SortPairs(d_sortTempStorage, 
                                  sortTempStorageBytes, 
                                  d_cellIds_in, 
                                  d_cellIds_out, 
                                  d_particleIds_in, 
                                  d_particleIds_out, 
                                  simulationParameters.numberOfParticles);
}

// --------------------------------------------------------------------------

__global__ void copy() {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

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
  copy<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>();
}

// --------------------------------------------------------------------------

__global__ void reorder(unsigned int* cellIdsOut,
                        unsigned int* particleIdsOut) {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

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
  reorder<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>(d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

void reorderStorage() {
  cudaCallCopy();
  cudaCallReorder();
}

// --------------------------------------------------------------------------

__global__ void resetCellInfo(unsigned int* cellStarts,
                              unsigned int* cellEndings) {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  cellStarts[idx] = UINT_MAX;
  cellEndings[idx] = numberOfParticles;
}

void cudaCallResetCellInfo() {
  resetCellInfo<<<cudaCallParameters.blocksForGridBased, cudaCallParameters.threadsForGridBased>>>(d_cellStarts, d_cellEndings);
}

// --------------------------------------------------------------------------

__global__ void computeCellInfo(unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* cellIdsOut,
                                unsigned int* particleIdsOut)  {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

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
      if( idx == numberOfParticles-1 ) {
        cellEndings[idx] = numberOfParticles;
      }
    }
  }
}

void cudaCallComputeCellInfo() {
  computeCellInfo<<<cudaCallParameters.blocksForGridBased, cudaCallParameters.threadsForGridBased>>>(d_cellStarts, d_cellEndings, d_cellIds_out, d_particleIds_out);
}

// --------------------------------------------------------------------------

__global__ void resetContacts(unsigned int* contacts) {
  const unsigned int maxContactConstraints = d_simulationParameters.maxContactConstraints;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contacts[idx] = UINT_MAX;
  }
}

void cudaCallResetContacts() {
  resetContacts<<<cudaCallParameters.blocksForContactBased, cudaCallParameters.threadsForContactBased>>>(d_contacts);
}

// --------------------------------------------------------------------------

__global__ void findContacts(unsigned int* cellStarts,
                             unsigned int* cellEndings,
                             unsigned int* contacts,
                             unsigned int* cellIds,
                             unsigned int* contactCounters,
                             float* densities) {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;
  const unsigned int maxGrid = d_simulationParameters.maxGrid;
  const unsigned int maxContactsPerParticle = d_simulationParameters.maxContactsPerParticle;
  const float particleDiameter = d_simulationParameters.particleDiameter;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x1 = (idx % textureWidth) * sizeof(float4);
  const unsigned int y1 = idx / textureWidth;

  contacts += maxContactsPerParticle * idx;
  unsigned int counter = 0;
  //float numberOfCloseNeighours = 0.0f;

  if( idx < numberOfParticles ) {
    float4 predictedPosition1;
    surf2Dread(&predictedPosition1, predictedPositions4, x1, y1);
    predictedPosition1.w = 0.0f;
    const float4 predictedPosition1Org = predictedPosition1;

    //float4 position1;
    //surf2Dread(&position1, positions4, x1, y1);

    const int searchWidth = 1;
    float4 tempPos;
    unsigned int morton;
    unsigned int start;
    unsigned int end;
    unsigned int index;
    unsigned int x2;
    unsigned int y2;
    float distance;
    float4 predictedPosition2; 
    for(int i=-searchWidth; i<=searchWidth; i++) {
      for(int j=-searchWidth; j<=searchWidth; j++) {
        for(int k=-searchWidth; k<=searchWidth; k++) {
          tempPos = predictedPosition1Org;
          tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
          morton = mortonCode(tempPos);
          if( morton < maxGrid ) {
            start = cellStarts[morton]; 
            end = cellEndings[morton];
            if( start != UINT_MAX ) {
              for(index=start; index<end && counter<maxContactsPerParticle; index++) {
                //numberOfCloseNeighours += 1.0f;
                if( idx != index && index < numberOfParticles ) {
                  x2 = (index % textureWidth) * sizeof(float4);
                  y2 = index / textureWidth;
                  surf2Dread(&predictedPosition2, predictedPositions4, x2, y2);
                  predictedPosition2.w = 0.0f;
                  distance = length(predictedPosition1 - predictedPosition2);
                  if( distance < particleDiameter ) {
                    contacts[counter++] = index;
                   /* if( counter >= maxContactsPerParticle ) {
                      contactCounters[idx] = counter;
                      return;
                    }*/
                  
                    /*
                    const float overlap = particleDiameter - distance;
                    if( overlap > 0 ) {
                      const float4 pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 
                      const float halfOverlap = overlap / 2.0f;
                   
                      //if( idx < index ) {
                        const float4 addTo1 = -1.0 * pos1ToPos2 * halfOverlap;
                        //surf2Dread(&predictedPosition1, predictedPositions4, x1, y1);
                        predictedPosition1 += addTo1/2.0f;
                        position1 += addTo1/2.0f;
                        
                      //} else {
                      //  const float4 addTo2 = pos1ToPos2 * halfOverlap;
                      //  predictedPosition2 += addTo2;
                      //  surf2Dwrite(predictedPosition2, predictedPositions4, x2, y2);
                      //  float4 position2;
                      //  surf2Dread(&position2, positions4, x2, y2);
                      //  position2 += addTo2;
                      //  surf2Dwrite(position2, positions4, x2, y2);
                      //}
                    }
                    */
                  } 
                }
              }
            }
          }

        }
      }
    }
    //surf2Dwrite(predictedPosition1, predictedPositions4, x1, y1);
    //surf2Dwrite(position1, positions4, x1, y1);
  }
  contactCounters[idx] = counter;
  //densities[idx] = numberOfCloseNeighours / 26.0f;
}

__global__ void copyPredictedPositions() {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  if( idx < numberOfParticles ) {
    float4 predictedPosition;
    surf2Dread(&predictedPosition, predictedPositions4, x, y);
    surf2Dwrite(predictedPosition, predictedPositions4Copy, x, y);
  }
}

void cudaCallFindContacts() {
  //copyPredictedPositions<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>();
  findContacts<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>(d_cellStarts, d_cellEndings, d_contacts , d_cellIds_out, d_contactCounters, d_densities);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintSuccess(int* contactConstraintSucces) {
  const unsigned int maxContactConstraints = d_simulationParameters.maxContactConstraints;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contactConstraintSucces[idx] = -1;
  } 
}

void cudaCallResetContactConstraintSuccess() {
  resetContactConstraintSuccess<<<cudaCallParameters.blocksForContactBased, cudaCallParameters.threadsForContactBased>>>(d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintParticleUsed(int* contactConstraintParticleUsed) {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfParticles  ) {
    contactConstraintParticleUsed[idx] = -1;
  }
}

void cudaCallResetContactConstraintParticleUsed() {
  resetContactConstraintParticleUsed<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>(d_contactConstraintParticleUsed);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatches(unsigned int* contacts,
                                                int* particleUsed,
                                                int* constraintSucces) {
  const unsigned int maxContactConstraints = d_simulationParameters.maxContactConstraints;
  const unsigned int maxContactsPerParticle = d_simulationParameters.maxContactsPerParticle;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    const int success = constraintSucces[idx];
    if( success < 0 ) { // If not success
      const unsigned int particle1 = idx / maxContactsPerParticle;
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
  setupCollisionConstraintBatches<<<cudaCallParameters.blocksForContactBased, cudaCallParameters.threadsForContactBased>>>(d_contacts, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatchesCheck(unsigned int* contacts,
                                                     int* particleUsed,
                                                     int* constraintSucces) {
  const unsigned int maxContactConstraints = d_simulationParameters.maxContactConstraints;
  const unsigned int maxContactsPerParticle = d_simulationParameters.maxContactsPerParticle;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;
  const float particleDiameter = d_simulationParameters.particleDiameter;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    const unsigned int particle1 = idx / maxContactsPerParticle;
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
  setupCollisionConstraintBatchesCheck<<<cudaCallParameters.blocksForContactBased, cudaCallParameters.threadsForContactBased>>>(d_contacts, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

void solveCollisions() {
  for(unsigned int i=0; i<1; i++) {
    cudaCallResetContacts();
    cudaCallFindContacts();
    cudaCallResetContactConstraintSuccess();
    const unsigned int maxBatches = simulationParameters.maxContactsPerParticle;
    for(unsigned int b=0; b<maxBatches; b++) {
      cudaCallResetContactConstraintParticleUsed();
      cudaCallSetupCollisionConstraintBatches();
      cudaCallSetupCollisionConstraintBatchesCheck();
    }
  }
}

// --------------------------------------------------------------------------

__global__ void updatePositions() {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;
  const float deltaT = d_simulationParameters.deltaT;

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
  updatePositions<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>();
}

// --------------------------------------------------------------------------

__global__ void initializeParticleIds(unsigned int* particleIdsIn) {
  const unsigned int numberOfParticles = d_simulationParameters.numberOfParticles;
  const unsigned int textureWidth = d_simulationParameters.textureWidth;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < numberOfParticles ) {
    particleIdsIn[idx] = idx;
  }
}

void cudaCallInitializeParticleIds() {
  initializeParticleIds<<<cudaCallParameters.blocksForParticleBased, cudaCallParameters.threadsForParticleBased>>>(d_particleIds_in);
}

// --------------------------------------------------------------------------

void initializeSort() {
  CUDA(cudaMalloc((void**)&d_cellIds_in, simulationParameters.maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_cellIds_out, simulationParameters.maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_particleIds_in, simulationParameters.maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&d_particleIds_out, simulationParameters.maxParticles * sizeof(unsigned int)));

  cudaCallInitializeParticleIds();
  
  cub::DeviceRadixSort::SortPairs(d_sortTempStorage, 
                                  sortTempStorageBytes,
		                              d_cellIds_in, 
                                  d_cellIds_out, 
                                  d_particleIds_in, 
                                  d_particleIds_out,
                                  simulationParameters.numberOfParticles);

  CUDA(cudaMalloc(&d_sortTempStorage, sortTempStorageBytes));
}

// --------------------------------------------------------------------------

void initializeCellInfo() {
  CUDA(cudaMalloc((void**)&d_cellStarts, simulationParameters.maxGrid * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_cellEndings, simulationParameters.maxGrid * sizeof(unsigned int)));
  cudaCallResetCellInfo();
}

// --------------------------------------------------------------------------

void initializeCollision() {
  CUDA(cudaMalloc((void**)&d_contacts, simulationParameters.maxContactConstraints * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactCounters, simulationParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintSucces, simulationParameters.maxContactConstraints * sizeof(int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintParticleUsed, simulationParameters.maxParticles * sizeof(int)));
}

// --------------------------------------------------------------------------

void cudaInitializeKernels() {
  initializeFrame();

  initializeSharedTexture(positions4, "positions4");
  initializeSharedTexture(predictedPositions4, "predictedPositions4");
  initializeSharedTexture(velocities4, "velocities4");
  initializeSharedTexture(colors4, "colors4");

  initializeSharedTexture(positions4Copy, "positions4Copy");
  initializeSharedTexture(predictedPositions4Copy, "predictedPositions4Copy");
  initializeSharedTexture(velocities4Copy, "velocities4Copy");
  initializeSharedTexture(colors4Copy, "colors4Copy");

  initializeSharedBuffer(d_densities, "d_densities");

  initializeSort();
  initializeCellInfo();
  initializeCollision();
}
