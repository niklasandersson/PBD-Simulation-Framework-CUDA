#ifndef DENSITYKERNELS_H
#define DENSITYKERNELS_H

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "../../Parameters.h"

__device__ float poly6(float4 pi,
  float4 pj,
  float kernelWidth);

__device__ float4 spiky(float4 pi,
  float4 pj,
  float kernelWidth);

#endif