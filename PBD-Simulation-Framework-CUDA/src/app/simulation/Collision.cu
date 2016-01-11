#include "Collision.h"

Collision::Collision() {

  cudaInitializeKernels();

}


void Collision::compute() {

  cudaCallApplyForces();

  cudaCallInitializeCellIds();
  
  sortIds();

  reorderStorage();

  cudaCallResetCellInfo();

  cudaCallComputeCellInfo();

  solveCollisions();

  cudaCallUpdatePositions();

}