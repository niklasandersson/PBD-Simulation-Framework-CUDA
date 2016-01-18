#include "Collision.h"
#include "cub/cub.cuh"


__global__ void initializeParticleIds(const unsigned int numberOfParticles,
                                      unsigned int* particleIdsIn) {
  GET_INDEX

  if( index < numberOfParticles ) {
    particleIdsIn[index] = index;
  }
}


void cudaCallInitializeParticleIds(Parameters* parameters) {
  initializeParticleIds<<<PARTICLE_BASED>>>
                         (parameters->deviceParameters.numberOfParticles,
                          parameters->deviceBuffers.d_particleIds_in);
}


void initializeSort(Parameters* parameters) {
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_cellIds_in, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_cellIds_out, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_particleIds_in, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_particleIds_out, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));

  cudaCallInitializeParticleIds(parameters);
  
  cub::DeviceRadixSort::SortPairs(parameters->deviceBuffers.d_sortTempStorage, 
                                  parameters->deviceBuffers.sortTempStorageBytes,
		                              parameters->deviceBuffers.d_cellIds_in, 
                                  parameters->deviceBuffers.d_cellIds_out, 
                                  parameters->deviceBuffers.d_particleIds_in, 
                                  parameters->deviceBuffers.d_particleIds_out,
                                  parameters->deviceParameters.numberOfParticles);

  CUDA(cudaMalloc(&parameters->deviceBuffers.d_sortTempStorage, parameters->deviceBuffers.sortTempStorageBytes));
}


__global__ void resetCellInfo(const unsigned int numberOfParticles,
                              const unsigned int maxGrid,
                              unsigned int* cellStarts,
                              unsigned int* cellEndings) {
  GET_INDEX

  if( index < maxGrid ) {
    cellStarts[index] = UINT_MAX;
    cellEndings[index] = numberOfParticles;
  }
}


void cudaCallResetCellInfo(Parameters* parameters) {
  resetCellInfo<<<GRID_BASED>>>
                 (parameters->deviceParameters.numberOfParticles,
                  parameters->deviceParameters.maxGrid,
                  parameters->deviceBuffers.d_cellStarts, 
                  parameters->deviceBuffers.d_cellEndings);
}


void initializeCellInfo(Parameters* parameters) {
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_cellStarts, parameters->deviceParameters.maxGrid * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_cellEndings, parameters->deviceParameters.maxGrid * sizeof(unsigned int)));
  cudaCallResetCellInfo(parameters);
}


void initializeContacts(Parameters* parameters) {
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_neighbours, parameters->deviceParameters.maxContactConstraints * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contactCounters, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_neighbourCounters, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contactConstraintSucces, parameters->deviceParameters.maxContactConstraints * sizeof(int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contactConstraintParticleUsed, parameters->deviceParameters.maxParticles * sizeof(int)));
}


void initializeCollision(Parameters* parameters) {
  initializeSort(parameters);
  initializeCellInfo(parameters);
  initializeContacts(parameters);
}


__global__ void resetContacts(const unsigned int maxContactConstraints,
                              unsigned int* neighbours) {
  GET_INDEX

  if( index < maxContactConstraints ) {
    neighbours[index] = UINT_MAX;
  }
}


void cudaCallResetContacts(Parameters* parameters) {
  resetContacts<<<CONTACT_BASED>>>
                 (parameters->deviceParameters.maxContactConstraints,
                  parameters->deviceBuffers.d_neighbours);
}


__global__ void initializeCellIds(const unsigned int numberOfParticles,
                                  unsigned int* cellIdsIn,
                                  float4* predictedPositions) {
  GET_INDEX
  
  if( index < numberOfParticles ) {
    cellIdsIn[index] = getHash(predictedPositions[index]);
  } else {
    cellIdsIn[index] = UINT_MAX;
  }
}


void cudaCallInitializeCellIds(Parameters* parameters) {
  initializeCellIds<<<PARTICLE_BASED>>>
                     (parameters->deviceParameters.numberOfParticles,
                      parameters->deviceBuffers.d_cellIds_in,
                      parameters->deviceBuffers.d_predictedPositions);
}


void sortIds(Parameters* parameters) {
  cub::DeviceRadixSort::SortPairs(parameters->deviceBuffers.d_sortTempStorage, 
                                  parameters->deviceBuffers.sortTempStorageBytes,
		                              parameters->deviceBuffers.d_cellIds_in, 
                                  parameters->deviceBuffers.d_cellIds_out, 
                                  parameters->deviceBuffers.d_particleIds_in, 
                                  parameters->deviceBuffers.d_particleIds_out,
                                  parameters->deviceParameters.numberOfParticles);
}


__global__ void copy(const unsigned int numberOfParticles,
                     float4* positions,
                     float4* positionsCopy,
                     float4* predictedPositions,
                     float4* predictedPositionsCopy,
                     float4* velocities,
                     float4* velocitiesCopy,
                     float4* colors,
                     float4* colorsCopy) {
  GET_INDEX
        
  if( index < numberOfParticles ) {
    positionsCopy[index] = positions[index];
    predictedPositionsCopy[index] = predictedPositions[index];
    velocitiesCopy[index] = velocities[index];
    colorsCopy[index] = colors[index]; 
  } 
}


void cudaCallCopy(Parameters* parameters) {
  copy<<<PARTICLE_BASED>>>
        (parameters->deviceParameters.numberOfParticles,
         parameters->deviceBuffers.d_positions,
         parameters->deviceBuffers.d_positionsCopy,
         parameters->deviceBuffers.d_predictedPositions,
         parameters->deviceBuffers.d_predictedPositionsCopy,
         parameters->deviceBuffers.d_velocities,
         parameters->deviceBuffers.d_velocitiesCopy,
         parameters->deviceBuffers.d_colors,
         parameters->deviceBuffers.d_colorsCopy);
}


__global__ void reorder(unsigned int* particleIdsOut,
                        const unsigned int numberOfParticles,
                        float4* positions,
                        float4* positionsCopy,
                        float4* predictedPositions,
                        float4* predictedPositionsCopy,
                        float4* velocities,
                        float4* velocitiesCopy,
                        float4* colors,
                        float4* colorsCopy) {
  GET_INDEX

  if( index < numberOfParticles ) {
    const unsigned int previousIndex = particleIdsOut[index];
    positions[index] = positionsCopy[previousIndex];
    predictedPositions[index] = predictedPositionsCopy[previousIndex];
    velocities[index] = velocitiesCopy[previousIndex];
    colors[index] = colorsCopy[previousIndex];
  }
}


void cudaCallReorder(Parameters* parameters) {
  reorder<<<PARTICLE_BASED>>>
           (parameters->deviceBuffers.d_particleIds_out,
            parameters->deviceParameters.numberOfParticles,
            parameters->deviceBuffers.d_positions,
            parameters->deviceBuffers.d_positionsCopy,
            parameters->deviceBuffers.d_predictedPositions,
            parameters->deviceBuffers.d_predictedPositionsCopy,
            parameters->deviceBuffers.d_velocities,
            parameters->deviceBuffers.d_velocitiesCopy,
            parameters->deviceBuffers.d_colors,
            parameters->deviceBuffers.d_colorsCopy);
}


void reorderStorage(Parameters* parameters) {
  cudaCallCopy(parameters);
  cudaCallReorder(parameters);
}


void hashSortReorder(Parameters* parameters) {
  cudaCallInitializeCellIds(parameters);
  sortIds(parameters);
  reorderStorage(parameters);
}


__global__ void computeCellInfo(const unsigned int numberOfParticles,
                                unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* cellIdsOut,
                                unsigned int* particleIdsOut)  {
  GET_INDEX

  const unsigned int cellId = cellIdsOut[index];
  const unsigned int particleId = particleIdsOut[index];

  if( index < numberOfParticles ) {
    if( index == 0 ) {
      cellStarts[cellId] = 0; 
    } else {
      const unsigned int previousCellId = cellIdsOut[index-1];
      if( previousCellId != cellId ) {
        cellStarts[cellId] = index;
        cellEndings[previousCellId] = index;
      }
      if( index == numberOfParticles-1 ) {
        cellEndings[index] = numberOfParticles;
      }
    }
  }
}


void cudaCallComputeCellInfo(Parameters* parameters) {
  computeCellInfo<<<GRID_BASED>>>(parameters->deviceParameters.numberOfParticles,
                                  parameters->deviceBuffers.d_cellStarts, 
                                  parameters->deviceBuffers.d_cellEndings, 
                                  parameters->deviceBuffers.d_cellIds_out, 
                                  parameters->deviceBuffers.d_particleIds_out);
}


__global__ void findNeighbours(const unsigned int numberOfParticles,
                               const unsigned int maxGrid,
                               const unsigned int maxNeighboursPerParticle,
                               const float particleDiameter,
                               const unsigned int kernelWidth,
                               float4* predictedPositions,
                               float4* positions,
                               unsigned int* cellStarts,
                               unsigned int* cellEndings,
                               unsigned int* neighbours,
                               unsigned int* cellIds,
                               unsigned int* contactCounters,
                               unsigned int* neighbourCounters,
                               float* densities) {
  GET_INDEX

  neighbours += maxNeighboursPerParticle * index;
  unsigned int counter = 0;

  if( index < numberOfParticles ) {
    float4 predictedPosition1 = predictedPositions[index]; 
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
          hash = getHash(tempPos);
          if( hash < maxGrid ) {
            start = cellStarts[hash]; 
            if( start != UINT_MAX ) {
              end = cellEndings[hash];
              for(index2=start; index2<end; index2++) {
                if( index != index2 && index2 < numberOfParticles ) {
                  neighbours[counter++] = index2;
                  /*
                  predictedPosition1 = predictedPositions[index]; 
                  predictedPosition2 = predictedPositions[index2]; 

                  distance = length(predictedPosition2 - predictedPosition1);
                  overlap = particleDiameter - distance;

                  if( overlap > 0 ) {
                    pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 
                    halfOverlap = overlap / 2.0f;

                    addTo1 = -1.0 * pos1ToPos2 * halfOverlap;
                    // addTo2 = pos1ToPos2 * halfOverlap;

                    //predictedPosition1 += addTo1;
                    //predictedPosition2 += addTo2;

                    //predictedPositions[index] = predictedPosition1;
                    //predictedPositions[index2] = predictedPosition2;

                    atomicAdd(&(predictedPositions[index].x), addTo1.x);
                    atomicAdd(&(predictedPositions[index].y), addTo1.y);
                    atomicAdd(&(predictedPositions[index].z), addTo1.z);
       
                    //atomicAdd(&(predictedPositions[index2].x), addTo2.x);
                    //atomicAdd(&(predictedPositions[index2].y), addTo2.y);
                    //atomicAdd(&(predictedPositions[index2].z), addTo2.z);

                    //float4 position1 = positions[index]; 
                    //float4 position2 = positions[index2];

                    //position1 += addTo1;
                    //position2 += addTo2;

                    //positions[index] = position1;
                    //positions[index2] = position2;
               
                    atomicAdd(&(positions[index].x), addTo1.x);
                    atomicAdd(&(positions[index].y), addTo1.y);
                    atomicAdd(&(positions[index].z), addTo1.z);
       
                    //atomicAdd(&(positions[index2].x), addTo2.x);
                    //atomicAdd(&(positions[index2].y), addTo2.y);
                    //atomicAdd(&(positions[index2].z), addTo2.z);
                  }*/
                  
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


void cudaCallFindNeighbours(Parameters* parameters) {
  findNeighbours<<<PARTICLE_BASED>>>
                  (parameters->deviceParameters.numberOfParticles,
                   parameters->deviceParameters.maxGrid,
                   parameters->deviceParameters.maxNeighboursPerParticle,
                   parameters->deviceParameters.particleDiameter,
                   parameters->deviceParameters.kernelWidth,
                   parameters->deviceBuffers.d_predictedPositions, 
                   parameters->deviceBuffers.d_positions, 
                   parameters->deviceBuffers.d_cellStarts, 
                   parameters->deviceBuffers.d_cellEndings, 
                   parameters->deviceBuffers.d_neighbours , 
                   parameters->deviceBuffers.d_cellIds_out, 
                   parameters->deviceBuffers.d_contactCounters, 
                   parameters->deviceBuffers.d_neighbourCounters,
                   parameters->deviceBuffers.d_densities);
}


__global__ void solveCollisions(const unsigned int numberOfParticles,
                                const unsigned int maxGrid,
                                const unsigned int maxNeighboursPerParticle,
                                const float particleDiameter,
                                const unsigned int kernelWidth,
                                float4* predictedPositions,
                                float4* positions,
                                unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* neighbours,
                                unsigned int* cellIds,
                                unsigned int* contactCounters,
                                unsigned int* neighbourCounters,
                                float* densities) {
  GET_INDEX

  if( index < numberOfParticles ) {
    neighbours += maxNeighboursPerParticle * index;

    float4 predictedPosition1 = predictedPositions[index]; 
    float4 predictedPosition2;
    float4 pos1ToPos2;
    float4 addTo1;
    unsigned int index2;
    float halfOverlap;

    const unsigned int numberOfContacts = contactCounters[index];

    for(unsigned int i=0; i<numberOfContacts; i++) {
      index2 = neighbours[i];
      
      predictedPosition2 = predictedPositions[index2]; 

      halfOverlap = (particleDiameter - length(predictedPosition2 - predictedPosition1)) / 2.0f;

      if( halfOverlap > 0 ) {
        pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 

        addTo1 = -1.0 * pos1ToPos2 * halfOverlap;

        predictedPosition1 += addTo1;

        predictedPositions[index] += addTo1;
        positions[index] += addTo1;

        /*
        atomicAdd(&(predictedPositions[index].x), addTo1.x);
        atomicAdd(&(predictedPositions[index].y), addTo1.y);
        atomicAdd(&(predictedPositions[index].z), addTo1.z);
  
        atomicAdd(&(positions[index].x), addTo1.x);
        atomicAdd(&(positions[index].y), addTo1.y);
        atomicAdd(&(positions[index].z), addTo1.z);
        */
      }
    }
  }

}


void cudaCallSolveCollisions(Parameters* parameters) {
    solveCollisions<<<PARTICLE_BASED>>>
                  (parameters->deviceParameters.numberOfParticles,
                   parameters->deviceParameters.maxGrid,
                   parameters->deviceParameters.maxNeighboursPerParticle,
                   parameters->deviceParameters.particleDiameter,
                   parameters->deviceParameters.kernelWidth,
                   parameters->deviceBuffers.d_predictedPositions, 
                   parameters->deviceBuffers.d_positions, 
                   parameters->deviceBuffers.d_cellStarts, 
                   parameters->deviceBuffers.d_cellEndings, 
                   parameters->deviceBuffers.d_neighbours , 
                   parameters->deviceBuffers.d_cellIds_out, 
                   parameters->deviceBuffers.d_contactCounters, 
                   parameters->deviceBuffers.d_neighbourCounters,
                   parameters->deviceBuffers.d_densities);
}


__global__ void resetContactConstraintSuccess(const unsigned int maxContactConstraints,
                                              int* contactConstraintSucces) {
  GET_INDEX

  if( index < maxContactConstraints ) {
    contactConstraintSucces[index] = -1;
  } 
}

void cudaCallResetContactConstraintSuccess(Parameters* parameters) {
  resetContactConstraintSuccess<<<CONTACT_BASED>>>
                                 (parameters->deviceParameters.maxContactConstraints,
                                  parameters->deviceBuffers.d_contactConstraintSucces);
}

__global__ void resetContactConstraintParticleUsed(const unsigned int numberOfParticles,
                                                   int* contactConstraintParticleUsed) {
  GET_INDEX

  if( index < numberOfParticles  ) {
    contactConstraintParticleUsed[index] = -1;
  }
}

void cudaCallResetContactConstraintParticleUsed(Parameters* parameters) {
  resetContactConstraintParticleUsed<<<PARTICLE_BASED>>>
                                      (parameters->deviceParameters.numberOfParticles,
                                       parameters->deviceBuffers.d_contactConstraintParticleUsed);
}


__global__ void setupCollisionConstraintBatches(const unsigned int maxContactConstraints,
                                                const unsigned int maxContactsPerParticle,
                                                unsigned int* contacts,
                                                int* particleUsed,
                                                int* constraintSucces) {
  GET_INDEX
    
  if( index < maxContactConstraints ) {
    const int success = constraintSucces[index];
    if( success < 0 ) { // If not success
      const unsigned int particle1 = index / maxContactsPerParticle;
      const unsigned int particle2 = contacts[index];
      if( particle2 != UINT_MAX ) {
        const unsigned int localId = threadIdx.x;
        const unsigned int localWorkSize = blockDim.x;
        for(unsigned int i=0; i<localWorkSize; i++) {
          if( (i == localId) && (particleUsed[particle1] < 0) && (particleUsed[particle2] < 0) ) {
            if( particleUsed[particle1] == -1 ) {
              particleUsed[particle1] = index;
            }
            if( particleUsed[particle2] == -1 ) {
              particleUsed[particle2] = index;
            }
          }
          __syncthreads();
        }
      }

    }
  }
  
}


void cudaCallSetupCollisionConstraintBatches(Parameters* parameters) {
  setupCollisionConstraintBatches<<<CONTACT_BASED>>>
                                   (parameters->deviceParameters.maxContactConstraints,
                                    parameters->deviceParameters.maxNeighboursPerParticle,
                                    parameters->deviceBuffers.d_neighbours, 
                                    parameters->deviceBuffers.d_contactConstraintParticleUsed, 
                                    parameters->deviceBuffers.d_contactConstraintSucces);

}


__global__ void setupCollisionConstraintBatchesCheck(const unsigned int maxContactConstraints,
                                                     const unsigned int maxContactsPerParticle,
                                                     const float particleDiameter,
                                                     unsigned int* contacts,
                                                     int* particleUsed,
                                                     int* constraintSucces,
                                                     float4* positions,
                                                     float4* predictedPositions) {
  GET_INDEX

  if( index < maxContactConstraints ) {
    const unsigned int particle1 = index / maxContactsPerParticle;
    const unsigned int particle2 = contacts[index];
    if( particle2 != UINT_MAX ) {
       if( (particleUsed[particle1] == index) && (particleUsed[particle2] == index) ) {
        constraintSucces[index] = 1;

        float4 predictedPosition1 = predictedPositions[particle1]; 
        float4 predictedPosition2 = predictedPositions[particle2]; 

        const float distance = length(predictedPosition2 - predictedPosition1);
        const float  overlap = particleDiameter - distance;

        if( overlap > 0 ) {
          const float4 pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 
          const float halfOverlap = overlap / 2.0f;

          const float4 addTo1 = -1.0 * pos1ToPos2 * halfOverlap;
          const float4 addTo2 = pos1ToPos2 * halfOverlap;

          predictedPositions[particle1] = predictedPosition1 + addTo1;
          predictedPositions[particle2] = predictedPosition2 + addTo2;

          positions[particle1] = positions[particle1] + addTo1;
          positions[particle2] = positions[particle2] + addTo2;
        }
        
      } 
    }
  }
}


void cudaCallSetupCollisionConstraintBatchesCheck(Parameters* parameters) {
  setupCollisionConstraintBatchesCheck<<<CONTACT_BASED>>>
                                        (parameters->deviceParameters.maxContactConstraints,
                                         parameters->deviceParameters.maxNeighboursPerParticle,
                                         parameters->deviceParameters.particleDiameter,
                                         parameters->deviceBuffers.d_neighbours, 
                                         parameters->deviceBuffers.d_contactConstraintParticleUsed, 
                                         parameters->deviceBuffers.d_contactConstraintSucces,
                                         parameters->deviceBuffers.d_positions,
                                         parameters->deviceBuffers.d_predictedPositions);
}


void findNeighboursAndSolveCollisions(Parameters* parameters) {
  //cudaCallResetContacts(parameters);

  cudaCallResetCellInfo(parameters);
  cudaCallComputeCellInfo(parameters);
  cudaCallFindNeighbours(parameters);
  cudaCallSolveCollisions(parameters);

  /*
  cudaCallResetContactConstraintSuccess(parameters);
  const unsigned int maxBatches = parameters->deviceParameters.maxNeighboursPerParticle;
  for(unsigned int b=0; b<maxBatches; b++) {
    cudaCallResetContactConstraintParticleUsed(parameters);
    cudaCallSetupCollisionConstraintBatches(parameters);
    cudaCallSetupCollisionConstraintBatchesCheck(parameters);
  }
  */

}

