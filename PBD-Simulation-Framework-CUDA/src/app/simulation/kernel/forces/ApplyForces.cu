#include "ApplyForces.h"


__global__ void applyForces(const unsigned int numberOfParticles,
                            const float deltaT,
                            float4* positions,
                            float4* predictedPositions,
                            float4* velocities) {
  const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

  if( index < numberOfParticles ) {
    const float inverseMass = 1.0f;
    const float gravity = -9.82;

    float4 velocity = velocities[index];
    velocity.y += inverseMass * gravity * deltaT; 

    float4 position = positions[index];

    float4 predictedPosition = position + velocity * deltaT;

    const float floorDiff = predictedPosition.y - 1.5;
    if( floorDiff < 0 ) {
      predictedPosition.y = predictedPosition.y + (-1.0f * floorDiff);
      position.y = position.y + (-1.0f * floorDiff);
      positions[index] = position;
    }

    predictedPositions[index] = predictedPosition;
  }
  
}


void cudaCallApplyForces(Parameters* parameters) {
  applyForces<<<PARTICLE_BASED>>>(parameters->deviceParameters.numberOfParticles,
                                  parameters->deviceParameters.deltaT,
                                  parameters->deviceBuffers.d_positions,
                                  parameters->deviceBuffers.d_predictedPositions,
                                  parameters->deviceBuffers.d_velocities);
}