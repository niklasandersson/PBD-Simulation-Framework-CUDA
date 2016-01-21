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
                                             bool& update) {
  float velocityDamping = 1.0f;
	if( predictedPosition.x < params.bounds.x.min ) {
		velocity.x = velocityDamping * velocity.x;
		predictedPosition.x = params.bounds.x.min + 0.001f;
    position = predictedPosition;
    update = true;
	} else if( predictedPosition.x > params.bounds.x.max ) {
		velocity.x = velocityDamping * velocity.x;
		predictedPosition.x = params.bounds.x.max - 0.001f;
    position = predictedPosition;
    update = true;
	}

	if( predictedPosition.y < params.bounds.y.min ) {
		velocity.y = velocityDamping * velocity.y;
    predictedPosition.y = params.bounds.y.min + 0.001f;
    position = predictedPosition;
    update = true;
	} else if( predictedPosition.y > params.bounds.y.max ) {
		velocity.y = velocityDamping * velocity.y;
		predictedPosition.y = params.bounds.y.max - 0.001f;
    position = predictedPosition;
    update = true;
	}

	if( predictedPosition.z < params.bounds.z.min ) {
		velocity.z = velocityDamping * velocity.z;
		predictedPosition.z = params.bounds.z.min + 0.001f;
    position = predictedPosition;
    update = true;
	} else if( predictedPosition.z > params.bounds.z.max ) {
		velocity.z = velocityDamping * velocity.z;
		predictedPosition.z = params.bounds.z.max - 0.001f;
    position = predictedPosition;
    update = true;
	}
}

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
    if( isnan(velocity.x) || isnan(velocity.y) || isnan(velocity.z) ) {
        printf("velocity: %f, %f, %f\n", velocity.x, velocity.y, velocity.z);
      }
   // velocity.y += inverseMass * gravity * deltaT; 

    float4 position;
    surf2Dread(&position, positions4, x, y);
    if( isnan(position.x) || isnan(position.y) || isnan(position.z) ) {
      printf("position: %f, %f, %f\n", position.x, position.y, position.z);
    }


    float4 predictedPosition = position + velocity * deltaT;

    bool update = false;
    confineToBox(position, predictedPosition, velocity, update);

    surf2Dwrite(velocity, velocities4, x, y);

    //applyMassScaling(predictedPosition);
    surf2Dwrite(predictedPosition, predictedPositions4, x, y);

    if( update ) {
      surf2Dwrite(position, positions4, x, y);
    }
    

      
      if( isnan(predictedPosition.x) || isnan(predictedPosition.y) || isnan(predictedPosition.z) ) {
        printf("predictedPosition: %f, %f, %f\n", predictedPosition.x, predictedPosition.y, predictedPosition.z);
      }

      
    

  }
}

void cudaCallApplyForces() {
  applyForces<<<FOR_EACH_PARTICLE>>>();
}

// --------------------------------------------------------------------------


#endif // FORCES_H