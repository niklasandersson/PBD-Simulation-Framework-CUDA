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
    float4 predictedPosition2; 

    // Find contacts
    for(int i=-1; i<=1; i++) {
      for(int j=-1; j<=1; j++) {
        for(int k=-1; k<=1; k++) {

          tempPos = predictedPosition1Org;
          tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
          hash = getHash(tempPos);
          if( hash < maxGrid ) {
            start = cellStarts[hash]; 
            end = cellEndings[hash];
            if( start != UINT_MAX ) {
              for(index2=start; index2<end && counter<maxNeighboursPerParticle; index2++) {
                if( index != index2 && index2 < numberOfParticles ) {
                  neighbours[counter++] = index2;
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
    for(int shell=2; shell<kernelWidth; shell++) {
      for(int i=-shell; i<=shell; i++) {
        abi = abs(i);
        for(int j=-shell; j<=shell; j++) {
          abj = abs(j);
          for(int k=-shell; k<=shell; k++) {
            if( (abi!=shell) || (abj!=shell) || (abs(k)!=shell) ) continue;

            tempPos = predictedPosition1Org;
            tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
            hash = getHash(tempPos);
            if( hash < maxGrid ) {
              start = cellStarts[hash]; 
              end = cellEndings[hash];
              if( start != UINT_MAX ) {
                for(index2=start; index2<end && counter<maxNeighboursPerParticle; index2++) {
                  if( index != index2 && index2 < numberOfParticles ) {
                    neighbours[counter++] = index2;
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


void cudaCallFindContacts(Parameters* parameters) {
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


void solveCollisions(Parameters* parameters) {

  for(unsigned int i=0; i<1; i++) {
    //cudaCallResetContacts(parameters);
    cudaCallFindContacts(parameters);
  }

}

