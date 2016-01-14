#include "Collision.h"

Collision::Collision() {

  cudaInitializeKernels();

}


void Collision::compute() {

  initializeFrame();

  cudaCallApplyForces();
  
  cudaCallInitializeCellIds();
  
  sortIds();

  reorderStorage();

  cudaCallResetCellInfo();

  cudaCallComputeCellInfo();

  solveCollisions();

  cudaCallUpdatePositions();

}