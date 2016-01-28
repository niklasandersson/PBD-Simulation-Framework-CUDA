#ifndef FORCES_H
#define FORCES_H

#include "Kernels.h"
#include "Globals.h"


// --------------------------------------------------------------------------


__device__ __forceinline__ void applyMassScaling(float4& predictedPosition) {
  predictedPosition.w = (M_E, -5 * predictedPosition.y);
}


// --------------------------------------------------------------------------


__device__ __forceinline__ void confineToBox(float4& position, 
                                             float4& predictedPosition, 
                                             float4& velocity,
                                             bool& updatePosition) {
  const float velocityDamping = params.forcesVelocityDamping;
  const float positionDamping = params.forcesPositionDamping;

	if( predictedPosition.x < params.bounds.x.min ) {
		velocity.x = velocityDamping * velocity.x;
		predictedPosition.x = params.bounds.x.min + 0.001f;
    updatePosition = true;
	} else if( predictedPosition.x > params.bounds.x.max ) {
		velocity.x = velocityDamping * velocity.x;
		predictedPosition.x = params.bounds.x.max - 0.001f;
    updatePosition = true;
	}

	if( predictedPosition.y < params.bounds.y.min ) {
		velocity.y = velocityDamping * velocity.y;
    predictedPosition.y = params.bounds.y.min + 0.001f;
    updatePosition = true;
	} else if( predictedPosition.y > params.bounds.y.max ) {
		velocity.y = velocityDamping * velocity.y;
		predictedPosition.y = params.bounds.y.max - 0.001f;
    updatePosition = true;
	}

	if( predictedPosition.z < params.bounds.z.min ) {
		velocity.z = velocityDamping * velocity.z;
		predictedPosition.z = params.bounds.z.min + 0.001f;
    updatePosition = true;
	} else if( predictedPosition.z > params.bounds.z.max ) {
		velocity.z = velocityDamping * velocity.z;
		predictedPosition.z = params.bounds.z.max - 0.001f;
    updatePosition = true;
	}

  if( updatePosition ) {
    position += positionDamping * (predictedPosition - position);
  }
}


// --------------------------------------------------------------------------


__global__ void applyForces(float4* externalForces) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;
  const float deltaT = params.deltaT;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  const unsigned int x = (idx % textureWidth) * sizeof(float4);
  const unsigned int y = idx / textureWidth;

  if( idx < numberOfParticles ) {
    const float inverseMass = 1.0f;
    const float gravity = params.gravity;

    float4 velocity;
    surf2Dread(&velocity, velocities4, x, y);
   
    velocity.y += inverseMass * gravity * deltaT; 
    velocity += externalForces[idx] * deltaT;

    float4 position;
    surf2Dread(&position, positions4, x, y);

    float4 predictedPosition = position + velocity * deltaT;

    bool updatePosition = false;
    confineToBox(position, predictedPosition, velocity, updatePosition);

    surf2Dwrite(velocity, velocities4, x, y);

    predictedPosition.w = 1.0f; // Set mass
    surf2Dwrite(predictedPosition, predictedPositions4, x, y);

    if( updatePosition ) {
      surf2Dwrite(position, positions4, x, y);
    }      
  }
}


void cudaCallApplyForces() {
  applyForces<<<FOR_EACH_PARTICLE>>>(d_externalForces);
}


// --------------------------------------------------------------------------


#endif // FORCES_H