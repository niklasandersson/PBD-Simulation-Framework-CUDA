#ifndef APPLYFORCES_H
#define APPLYFORCES_H

#include "app/simulation/Parameters.h"

__global__ void applyForces(const unsigned int numberOfParticles,
                            const float deltaT,
                            float4* positions,
                            float4* predictedPositions,
                            float4* velocities);

void cudaCallApplyForces(Parameters* parameters);

#endif // APPLYFORCES