#ifndef DENSITY_H
#define DENSITY_H

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "../../Parameters.h"
//#include "DensityKernels.h"

__global__ void computeLambda(const unsigned int numberOfParticles,
  float4* predictedPositions);
void cudaCallComputeLamdbda(Parameters* parameters);

__device__ float computeConstraintValue(const unsigned int index,
  float4 pi,
  float4* predictedPositions,
  float restDensity,
  float kernelWidth,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors);

__device__ float4 computeGradientAtSelf(const unsigned int index,
  float4 pi,
  float4* predictedPositions,
  float restDensity,
  float kernelWidth,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors);

#endif