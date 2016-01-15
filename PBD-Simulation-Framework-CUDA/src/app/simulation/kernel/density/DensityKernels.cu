#include "DensityKernels.h"

__device__ float poly6(float4 pi,
  float4 pj,
  float kernelWidth)
{
  pi.w = 0.0f;
  pj.w = 0.0f;
  float distance = length(pi - pj);

  float numeratorTerm = kernelWidth * kernelWidth - distance * distance;
  return (315.0f * numeratorTerm * numeratorTerm) / (64.0f * M_PI * pow(kernelWidth, 9));
}

__device__ float4 spiky(float4 pi,
  float4 pj,
  float kernelWidth) {

  pi.w = 0.0f;
  pj.w = 0.0f;
  float4 r = pi - pj;
  float distance = length(r);

  float numeratorTerm = kernelWidth - distance;
  float denominatorTerm = M_PI * pow(kernelWidth, 6) * (distance + 0.001f);
  return 45.0f * numeratorTerm / denominatorTerm * r;
}