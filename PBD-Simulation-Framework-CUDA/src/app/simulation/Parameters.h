#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <string>
#include <limits>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"

#include "Util.h"

#define GET_INDEX const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

#define PARTICLE_BASED parameters->cudaCallParameters.blocksParticleBased,parameters->cudaCallParameters.threadsParticleBased
#define CONTACT_BASED parameters->cudaCallParameters.blocksContactBased,parameters->cudaCallParameters.threadsContactBased
#define GRID_BASED parameters->cudaCallParameters.blocksGridBased,parameters->cudaCallParameters.threadsGridBased

#define M_PI 3.14159265359
#define MAX_NEIGHBOURS_PER_PARTICLE 32
#define KERNEL_WIDTH 3

struct DeviceParameters{
  unsigned int numberOfParticles;
  unsigned int textureWidth;
  unsigned int maxNeighboursPerParticle;
  unsigned int maxContactConstraints;
  unsigned int maxGrid;
  unsigned int maxParticles;
  float particleRadius;
  float particleDiameter;
  float deltaT;
  unsigned int kernelWidth;
};

struct DeviceBuffers {
  // Shared buffers
  float* d_densities;
  float4* d_positions;
  float4* d_predictedPositions;
  float4* d_velocities;
  float4* d_colors;

  float* d_densitiesCopy;
  float4* d_positionsCopy;
  float4* d_predictedPositionsCopy;
  float4* d_velocitiesCopy;
  float4* d_colorsCopy;

  // Only for simulation
  unsigned int* d_cellIds_in;
  unsigned int* d_cellIds_out;

  unsigned int* d_particleIds_in;
  unsigned int* d_particleIds_out;

  void* d_sortTempStorage = nullptr;
  size_t sortTempStorageBytes = 0;

  unsigned int* d_cellStarts;
  unsigned int* d_cellEndings;

  unsigned int* d_neighbours;
  unsigned int* d_contactCounters;
  unsigned int* d_neighbourCounters;

  int* d_contactConstraintSucces;
  int* d_contactConstraintParticleUsed;
};

struct CudaCallParameters {
  dim3 blocksParticleBased;
  dim3 threadsParticleBased;

  dim3 blocksContactBased;
  dim3 threadsContactBased;

  dim3 blocksGridBased;
  dim3 threadsGridBased;
};

struct Parameters {
  DeviceParameters deviceParameters;
  DeviceBuffers deviceBuffers;
  CudaCallParameters cudaCallParameters;

  Parameters();
  ~Parameters();

  void update();
};

#endif // PARAMETERS_H