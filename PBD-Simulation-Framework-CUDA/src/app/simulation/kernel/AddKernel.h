#ifndef ADDKERNEL_H
#define ADDKERNEL_H

#include "cuda/Cuda.h"

//#include <thrust\device_vector.h>

/*
// --- 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// --- 2D surface memory
surface<void, 2> surf2D;
*/

__global__ void addKernel();

#endif // ADDKERNEL_H