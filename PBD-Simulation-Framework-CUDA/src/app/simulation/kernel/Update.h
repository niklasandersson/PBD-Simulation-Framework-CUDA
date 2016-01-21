#ifndef UPDATE_H
#define UPDATE_H

#include "Kernels.h"
#include "Globals.h"


// --------------------------------------------------------------------------

__global__ void updatePositions() {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;
  const float deltaT = params.deltaT;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;
  
  if( idx < numberOfParticles ) {
    float4 position;
    surf2Dread(&position, positions4, x, y);

    float4 predictedPosition;
    surf2Dread(&predictedPosition, predictedPositions4, x, y);

    float4 dir = predictedPosition - position;
    float4 velocity = dir / deltaT;

    //if( length(make_float3(dir)) > 0.05 ) {
      surf2Dwrite(predictedPosition, positions4, x, y);
    //}
    surf2Dwrite(velocity, velocities4, x, y);
  }
}

void cudaCallUpdatePositions() {
  updatePositions<<<FOR_EACH_PARTICLE>>>();
}

// --------------------------------------------------------------------------


#endif // UPDATE_H