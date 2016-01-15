#ifndef DENSITY_H
#define DENSITY_H

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "../../Parameters.h"
#include "DensityKernels.h"

__global__ void computeLambda(const unsigned int numberOfParticles,
  float4* predictedPositions);
void cudaCallComputeLamdbda(Parameters* parameters);

#endif