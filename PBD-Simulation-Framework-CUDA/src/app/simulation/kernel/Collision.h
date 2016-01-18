#ifndef COLLISION_H
#define COLLISION_H

#include "Kernels.h"
#include "Globals.h"


// --------------------------------------------------------------------------

__global__ void findNeighbours(unsigned int* neighbours,
                               unsigned int* contactCounters,
                               unsigned int* neighbourCounters,
                               unsigned int* cellStarts,
                               unsigned int* cellEndings) {
  GET_INDEX_X_Y

  const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int kernelWidth = params.kernelWidth;
  const float particleDiameter = params.particleDiameter;

  neighbours += maxNeighboursPerParticle * index;
  unsigned int counter = 0;

  if( index < numberOfParticles ) {
    float4 predictedPosition1;
    surf2Dread(&predictedPosition1, predictedPositions4, x, y);
    predictedPosition1.w = 0.0f;
    
    const float4 predictedPosition1Org = predictedPosition1;

    float4 tempPos;
    unsigned int hash;
    unsigned int start;
    unsigned int end;
    unsigned int index2;
    float distance;
    float overlap;
    float halfOverlap;
    float4 addTo1;
    float4 addTo2;
    float4 pos1ToPos2;
    float4 predictedPosition2; 
    
    if( counter == maxNeighboursPerParticle ) {
      both: 
      contactCounters[index] = counter;
      single:
      neighbourCounters[index] = counter;
      return;
    }
   
    for(int i=-1; i<=1; i++) {
      for(int j=-1; j<=1; j++) {
        for(int k=-1; k<=1; k++) {

          tempPos = predictedPosition1Org;
          tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
          hash = mortonCode(tempPos);
      
          if( hash < maxGrid ) {
            start = cellStarts[hash]; 
            if( start < numberOfParticles ) {
              end = cellEndings[hash];
              for(index2=start; index2<end; index2++) {
                if( index != index2 && index2 < numberOfParticles ) {
                  neighbours[counter++] = index2;
                  if( counter == maxNeighboursPerParticle ) {
                    goto both;
                  }
                }
              }
            }
          }

        }
      }
    }
     
    contactCounters[index] = counter;

    int abi = 0;
    int abj = 0;
    for(int shell=2; shell<=kernelWidth; shell++) {
      for(int i=-shell; i<=shell; i++) {
        abi = abs(i);
        for(int j=-shell; j<=shell; j++) {
          abj = abs(j);
          for(int k=-shell; k<=shell; k++) {
            if( (abi!=shell) && (abj!=shell) && (abs(k)!=shell) ) continue;

            tempPos = predictedPosition1Org;
            tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
            hash = getHash(tempPos);
            if( hash < maxGrid ) {
              start = cellStarts[hash]; 
              end = cellEndings[hash];
              if( start != UINT_MAX ) {
                for(index2=start; index2<end; index2++) {
                  if( index != index2 && index2 < numberOfParticles ) {
                    neighbours[counter++] = index2;
                    if( counter == maxNeighboursPerParticle ) {
                      goto single;
                    }
                  }
                }
              }
            }

          }
        }
      }
    }
    neighbourCounters[index] = counter;
  }
  
}


void cudaCallFindNeighbours() {
  findNeighbours<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_contactCounters, d_neighbourCounters, d_cellStarts, d_cellEndings);
}

// --------------------------------------------------------------------------

__global__ void solveCollisions(unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* neighbours,
                                unsigned int* contactCounters,
                                unsigned int* neighbourCounters) {
  GET_INDEX_X_Y

  const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int kernelWidth = params.kernelWidth;
  const float particleDiameter = params.particleDiameter;

  if( index < numberOfParticles ) {
    neighbours += maxNeighboursPerParticle * index;

    float4 predictedPosition1;
    surf2Dread(&predictedPosition1, predictedPositions4, x, y);
    predictedPosition1.w = 0.0f;

    float4 position1;
    surf2Dread(&position1, positions4, x, y);

    float4 predictedPosition2;
    float4 pos1ToPos2;
    float4 addTo1;
    unsigned int index2;
    float halfOverlap;
    unsigned int x2;
    unsigned int y2;

    const unsigned int numberOfContacts = contactCounters[index];

    for(unsigned int i=0; i<numberOfContacts; i++) {
      index2 = neighbours[i];
      
      x2 = (index2 % textureWidth) * sizeof(float4);
      y2 = index2 / textureWidth;     
      surf2Dread(&predictedPosition2, predictedPositions4, x2, y2);

      halfOverlap = (particleDiameter - length(predictedPosition2 - predictedPosition1)) / 2.0f;

      if( halfOverlap > 0 ) {
        pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 

        addTo1 = -1.0 * pos1ToPos2 * halfOverlap;

        predictedPosition1 += addTo1;
        position1 += addTo1;
        
        surf2Dwrite(predictedPosition1, predictedPositions4, x, y);
      
      }
    }
    surf2Dwrite(position1, positions4, x, y);
  }

}


void cudaCallSolveCollisions() {
    solveCollisions<<<FOR_EACH_PARTICLE>>>(d_cellStarts, d_cellEndings, d_neighbours, d_contactCounters, d_neighbourCounters);
}

// --------------------------------------------------------------------------

__global__ void resetContacts(unsigned int* contacts) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contacts[idx] = UINT_MAX;
  }
}

void cudaCallResetContacts() {
  resetContacts<<<FOR_EACH_CONTACT>>>(d_neighbours);
}

// --------------------------------------------------------------------------

__global__ void findContacts(unsigned int* cellStarts,
                             unsigned int* cellEndings,
                             unsigned int* contacts,
                             unsigned int* cellIds,
                             unsigned int* contactCounters,
                             float* densities) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int maxContactsPerParticle = params.maxNeighboursPerParticle;
  const float particleDiameter = params.particleDiameter;

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

    float4 position1;
    surf2Dread(&position1, positions4, x1, y1);

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

                  
                    
                    const float overlap = particleDiameter - distance;
                    if( overlap > 0 ) {
                      const float4 pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 
                      const float halfOverlap = overlap / 2.0f;
                   
                      //if( idx < index ) {
                        const float4 addTo1 = -1.0 * pos1ToPos2 * halfOverlap;
                        //surf2Dread(&predictedPosition1, predictedPositions4, x1, y1);
                        predictedPosition1 += addTo1;
                        position1 += addTo1;
                        surf2Dwrite(predictedPosition1, predictedPositions4, x1, y1);

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
                    
                  } 
                }
              }
            }
          }

        }
      }
    }
    
    surf2Dwrite(position1, positions4, x1, y1);
  }
  contactCounters[idx] = counter;
  //densities[idx] = numberOfCloseNeighours / 26.0f;
}

__global__ void copyPredictedPositions() {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;

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
  findContacts<<<FOR_EACH_PARTICLE>>>(d_cellStarts, d_cellEndings, d_neighbours , d_cellIds_out, d_contactCounters, d_densities);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintSuccess(int* contactConstraintSucces) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contactConstraintSucces[idx] = -1;
  } 
}

void cudaCallResetContactConstraintSuccess() {
  resetContactConstraintSuccess<<<FOR_EACH_CONTACT>>>(d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintParticleUsed(int* contactConstraintParticleUsed) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfParticles  ) {
    contactConstraintParticleUsed[idx] = -1;
  }
}

void cudaCallResetContactConstraintParticleUsed() {
  resetContactConstraintParticleUsed<<<FOR_EACH_PARTICLE>>>(d_contactConstraintParticleUsed);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatches(unsigned int* contacts,
                                                int* particleUsed,
                                                int* constraintSucces) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;
  const unsigned int maxContactsPerParticle = params.maxNeighboursPerParticle;

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
  setupCollisionConstraintBatches<<<FOR_EACH_CONTACT>>>(d_neighbours, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatchesCheck(unsigned int* contacts,
                                                     int* particleUsed,
                                                     int* constraintSucces) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;
  const unsigned int maxContactsPerParticle = params.maxNeighboursPerParticle;
  const unsigned int textureWidth = params.textureWidth;
  const float particleDiameter = params.particleDiameter;

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
  setupCollisionConstraintBatchesCheck<<<FOR_EACH_CONTACT>>>(d_neighbours, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

void collisionHandling() {
  unsigned int stabilizationIterations = 1;
  for(unsigned int i=0; i<stabilizationIterations; i++) {
    // cudaCallResetContacts();
    cudaCallFindNeighbours();
    cudaCallSolveCollisions();
    //cudaCallFindContacts();
    /*
    cudaCallResetContactConstraintSuccess();
    const unsigned int maxBatches = simulationParameters.maxNeighboursPerParticle;
    for(unsigned int b=0; b<maxBatches; b++) {
      cudaCallResetContactConstraintParticleUsed();
      cudaCallSetupCollisionConstraintBatches();
      cudaCallSetupCollisionConstraintBatchesCheck();
    }*/
  }
}

// --------------------------------------------------------------------------

void initializeCollision() {
  CUDA(cudaMalloc((void**)&d_neighbours, simulationParameters.maxContactConstraints * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactCounters, simulationParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_neighbourCounters, simulationParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintSucces, simulationParameters.maxContactConstraints * sizeof(int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintParticleUsed, simulationParameters.maxParticles * sizeof(int)));
}

// --------------------------------------------------------------------------



#endif // COLLISION_H