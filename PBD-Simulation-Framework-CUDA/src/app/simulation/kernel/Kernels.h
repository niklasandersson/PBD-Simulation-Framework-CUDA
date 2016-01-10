#ifndef KERNELS_H
#define KERNELS_H

#include <iostream>
#include <string>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"

void cudaInitializeKernels();

__global__ void applyForces(const unsigned int numberOfParticles,
                            const unsigned int textureWidth);

void cudaCallApplyForces();

#endif // KERNELS_H 