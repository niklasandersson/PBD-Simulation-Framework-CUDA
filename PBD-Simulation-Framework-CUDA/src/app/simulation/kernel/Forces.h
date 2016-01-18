#ifndef FORCES_H
#define FORCES_H

#include "Kernels.h"
#include "Globals.h"


// --------------------------------------------------------------------------

__global__ void applyForces() {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;
  const float deltaT = params.deltaT;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  if( idx < numberOfParticles ) {
    const float inverseMass = 1.0f;
    const float gravity = -9.82;

    float4 velocity;
    surf2Dread(&velocity, velocities4, x, y);
    velocity.y += inverseMass * gravity * deltaT; 

    float4 position;
    surf2Dread(&position, positions4, x, y);

    float4 predictedPosition = position + velocity * deltaT;

    const float floorDiff = predictedPosition.y - 1.5;
    if( floorDiff < 0 ) {
      predictedPosition.y = predictedPosition.y + (-1.0f * floorDiff);
      position.y = position.y + (-1.0f * floorDiff);
      surf2Dwrite(position, positions4, x, y);
    }

    surf2Dwrite(predictedPosition, predictedPositions4, x, y);
  }
}

void cudaCallApplyForces() {
  applyForces<<<FOR_EACH_PARTICLE>>>();
}

// --------------------------------------------------------------------------


#endif // FORCES_H