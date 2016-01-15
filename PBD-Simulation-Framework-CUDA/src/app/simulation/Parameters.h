#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <iostream>
#include <string>
#include <limits>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"

#include "Util.h"

#define PARTICLE_BASED parameters->cudaCallParameters.blocksParticleBased,parameters->cudaCallParameters.threadsParticleBased
#define CONTACTS_BASED parameters->cudaCallParameters.blocksContactBased,parameters->cudaCallParameters.threadsContactBased
#define GRID_BASED parameters->cudaCallParameters.blocksGridBased,parameters->cudaCallParameters.threadsGridBased

#define M_PI 3.14159265359

struct DeviceParameters{
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

struct DeviceBuffers {
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