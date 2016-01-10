#include "Collision.h"

Collision::Collision() {

  cudaInitializeKernels();

}


void Collision::compute() {

  cudaCallApplyForces();

  cudaCallInitializeCellIds();
  
  sortIds();

  cudaCallUpdatePositions();

}