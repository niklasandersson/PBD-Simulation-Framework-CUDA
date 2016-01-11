#ifndef KERNELS_H
#define KERNELS_H

#include <iostream>
#include <string>
#include <limits>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"

void cudaInitializeKernels();

void cudaCallApplyForces();
void cudaCallInitializeCellIds();
void sortIds();
void reorderStorage();
void cudaCallResetCellStarts();
void cudaCallComputeCellStarts();
void cudaCallUpdatePositions();


#endif // KERNELS_H 