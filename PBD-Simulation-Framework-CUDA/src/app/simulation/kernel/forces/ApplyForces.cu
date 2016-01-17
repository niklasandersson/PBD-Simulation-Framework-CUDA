#include "ApplyForces.h"


__global__ void applyForces(const unsigned int numberOfParticles,
                            const float deltaT,
                            float4* positions,
                            float4* predictedPositions,
                            float4* velocities,
														float4* externalForces) {
  const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

  if( index < numberOfParticles ) {
    const float inverseMass = 1.0f;
		const float4 gravity = make_float4(0.0f, -9.82, 0.0f, 0.0f);
	
    float4 velocity = velocities[index];
		velocity += (inverseMass * gravity + externalForces[index]) * deltaT;
		//printf("at index = %i : velocity.x = %f , velocity.y = %f, velocity.z = %f \n", index, velocity.x, velocity.y, velocity.z);
		
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
                                  parameters->deviceBuffers.d_velocities,
																	parameters->deviceBuffers.d_externalForces);
}