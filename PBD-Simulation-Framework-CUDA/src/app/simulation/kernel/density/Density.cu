#include "Density.h"

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

__global__ void computeLambda(const unsigned int numberOfParticles,
  float4* predictedPositions,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors,
  float restDensity,
  float kernelWidth
  ) {
  GET_INDEX

    float4 pi = predictedPositions[index];
  float ci = computeConstraintValue(index, pi, predictedPositions,
    restDensity, kernelWidth, neighbors, numberOfNeighbors, maxNumberOfNeighbors);

  float gradientValue = 0.0f;
  unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

  for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
    float4 pj = predictedPositions[neighborIndex];
    float4 gradient = -1.0f * spiky(pi, pj, kernelWidth) / restDensity;
    float gradientLength = length(gradient);
    gradientValue += gradientLength * gradientLength;
  }

  float4 gradientAtSelf = computeGradientAtSelf(index, pi, predictedPositions,
    restDensity, kernelWidth, neighbors, numberOfNeighbors, maxNumberOfNeighbors);

  float gradientAtSelfLength = length(gradientAtSelf);
  gradientValue += gradientAtSelfLength * gradientAtSelfLength;

  float lambda = -1.0f * ci / gradientValue;
}

void cudaCallComputeLamdbda(Parameters* parameters) {

}

__device__ float computeConstraintValue(const unsigned int index,
  float4 pi,
  float4* predictedPositions,
  float restDensity,
  float kernelWidth,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors) {

  float density = 0.0f;
  unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

  for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
    float4 pj = predictedPositions[neighborIndex];
    density += poly6(pi, pj, kernelWidth);
  }

  return (density / restDensity) - 1.0f;
}

__device__ float4 computeGradientAtSelf(const unsigned int index,
  float4 pi,
  float4* predictedPositions,
  float restDensity,
  float kernelWidth,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors) {

  float4 gradient = make_float4(0, 0, 0, 0);
  unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

  for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
    float4 pj = predictedPositions[neighborIndex];
    gradient += spiky(pi, pj, kernelWidth);
  }

  return gradient / restDensity;
}