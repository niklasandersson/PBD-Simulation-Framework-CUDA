#ifndef SORTREORDER_H
#define SORTREORDER_H

#include "Kernels.h"
#include "Globals.h"
#include "cub/cub.cuh"
#include "Hash.h"


// --------------------------------------------------------------------------


__global__ void initializeCellIds(unsigned int* cellIdsIn) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;

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
  initializeCellIds<<<FOR_EACH_PARTICLE>>>(d_cellIds_in);
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
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;

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
  copy<<<FOR_EACH_PARTICLE>>>();
}


// --------------------------------------------------------------------------


__global__ void reorder(unsigned int* cellIdsOut,
                        unsigned int* particleIdsOut) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;

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
  reorder<<<FOR_EACH_PARTICLE>>>(d_cellIds_out, d_particleIds_out);
}


// --------------------------------------------------------------------------


void reorderStorage() {
  cudaCallCopy();
  cudaCallReorder();
}


// --------------------------------------------------------------------------


__global__ void resetCellInfo(unsigned int* cellStarts,
                              unsigned int* cellEndings) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;
  const unsigned int maxGrid = params.maxGrid;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < maxGrid ) {
    cellStarts[idx] = UINT_MAX;
    cellEndings[idx] = numberOfParticles;
  }
}


void cudaCallResetCellInfo() {
  resetCellInfo<<<FOR_EACH_CELL>>>(d_cellStarts, d_cellEndings);
}


// --------------------------------------------------------------------------


__global__ void computeCellInfo(unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* cellIdsOut,
                                unsigned int* particleIdsOut)  {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  if( idx < numberOfParticles ) {
    const unsigned int cellId = cellIdsOut[idx];
    const unsigned int particleId = particleIdsOut[idx];

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
  computeCellInfo<<<FOR_EACH_PARTICLE>>>(d_cellStarts, d_cellEndings, d_cellIds_out, d_particleIds_out);
}


// --------------------------------------------------------------------------


__global__ void initializeParticleIds(unsigned int* particleIdsIn) {
  const unsigned int maxParticles = params.maxParticles;
  const unsigned int textureWidth = params.textureWidth;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < maxParticles ) {
    particleIdsIn[idx] = idx;
  }
}


void cudaCallInitializeParticleIds() {
  initializeParticleIds<<<FOR_ALL_POSSIBLE_PARTICLES>>>(d_particleIds_in);
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
                                  simulationParameters.maxParticles);

  CUDA(cudaMalloc(&d_sortTempStorage, sortTempStorageBytes));
}


// --------------------------------------------------------------------------


void cleanupSort() {
  CUDA(cudaFree(d_cellIds_in));
  CUDA(cudaFree(d_cellIds_out));
  CUDA(cudaFree(d_particleIds_in));
  CUDA(cudaFree(d_particleIds_out));
  CUDA(cudaFree(d_sortTempStorage));
}


// --------------------------------------------------------------------------


void initializeCellInfo() {
  CUDA(cudaMalloc((void**)&d_cellStarts, simulationParameters.maxGrid * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_cellEndings, simulationParameters.maxGrid * sizeof(unsigned int)));
  cudaCallResetCellInfo();
}


// --------------------------------------------------------------------------


void cleanupCellInfo() {
  CUDA(cudaFree(d_cellStarts));
  CUDA(cudaFree(d_cellEndings));
}


// --------------------------------------------------------------------------


void initializeSortReorder() {
  initializeSort();
  initializeCellInfo();
}


// --------------------------------------------------------------------------


void cleanupSortReorder() {
  cleanupSort();
  cleanupCellInfo();
}
 

// --------------------------------------------------------------------------


#endif // SORTREORDER_H
