#include "UpdatePositions.h"


__global__ void updatePositions(const unsigned int numberOfParticles,
                                const float deltaT,
                                float4* positions,
                                float4* predictedPositions,
                                float4* velocities) {
  const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
  
  if( index < numberOfParticles ) {
    float4 position = positions[index];
    float4 predictedPosition = predictedPositions[index];
    float4 velocity = (predictedPosition - position) / deltaT;
  
    positions[index] = predictedPosition;
    velocities[index] = velocity;
  }
}


void cudaCallUpdatePositions(Parameters* parameters) {
  updatePositions<<<PARTICLE_BASED>>>(parameters->deviceParameters.numberOfParticles,
                                      parameters->deviceParameters.deltaT,
                                      parameters->deviceBuffers.d_positions,
                                      parameters->deviceBuffers.d_predictedPositions,
                                      parameters->deviceBuffers.d_velocities);
}
