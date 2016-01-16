#ifndef DENSITY_H
#define DENSITY_H

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "../../Parameters.h"
//#include "DensityKernels.h"

void initilizeDensity(Parameters* parameters);

__global__ void computeLambda(const unsigned int numberOfParticles,
	float4* predictedPositions,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	unsigned int maxNumberOfNeighbors,
	float restDensity,
	float kernelWidth,
	float* lambdas);

void cudaCallComputeLambda(Parameters* parameters);

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

void cudaCallComputeDeltaPositions(Parameters* parameters);
__global__ void computeDeltaPositions(const unsigned int numberOfParticles,
	float4* predictedPositions,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	unsigned int maxNumberOfNeighbors,
	float restDensity,
	float kernelWidth,
	float* lambdas,
	float4* deltaPositions
	);

void cudaCallComputeOmega(Parameters* parameters);
__global__ void computeOmega(const unsigned int numberOfParticles,
														float4* predictedPositions,
														unsigned int* neighbors,
														unsigned int* numberOfNeighbors,
														unsigned int maxNumberOfNeighbors,
														float kernelWidth,
														float4* velocity,
														float3* omegas);

#endif