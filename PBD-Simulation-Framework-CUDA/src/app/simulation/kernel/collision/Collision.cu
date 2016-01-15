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
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contacts, parameters->deviceParameters.maxContactConstraints * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contactCounters, parameters->deviceParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contactConstraintSucces, parameters->deviceParameters.maxContactConstraints * sizeof(int)));
  CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_contactConstraintParticleUsed, parameters->deviceParameters.maxParticles * sizeof(int)));
}


void initializeCollision(Parameters* parameters) {
  initializeSort(parameters);
  initializeCellInfo(parameters);
  initializeContacts(parameters);
}

