#ifndef KERNELS_H
#define KERNELS_H

#include <iostream>
#include <string>
#include <limits>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"
#include "../Util.h"

void initializeFrame();
void cudaInitializeKernels();

void cudaCallApplyForces();
void cudaCallInitializeCellIds();
void sortIds();
void reorderStorage();
void cudaCallResetCellInfo();
void cudaCallComputeCellInfo();
void collisionHandling();
void cudaCallUpdatePositions();
void cudaCallComputeLambda();
void cudaCallComputeDeltaPositions();
void cudaCallApplyDeltaPositions();
void cudaCallComputeOmegas();
void cudaCallComputeVorticity();
void cudaComputeViscosity();
void callClearAllTheCrap();

#endif // KERNELS_H 