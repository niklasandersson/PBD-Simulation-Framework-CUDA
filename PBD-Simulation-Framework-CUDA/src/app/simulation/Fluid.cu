#include "Fluid.h"


Fluid::Fluid(Parameters* parameters) : parameters_(parameters){

  //cudaInitializeKernels();
  initializeCollision(parameters_);
	initilizeDensity(parameters_);
}


void Fluid::compute() {

  cudaCallApplyForces(parameters_);

  solveCollisions(parameters_);

  cudaCallUpdatePositions(parameters_);

	cudaCallComputeLambda(parameters_);
	
	cudaCallComputeDeltaPositions(parameters_);

/*
  initializeFrame();

  cudaCallApplyForces();
 
  cudaCallInitializeCellIds();
  
  sortIds();

  reorderStorage();

  cudaCallResetCellInfo();

  cudaCallComputeCellInfo();

  solveCollisions();

  cudaCallUpdatePositions();
  */

}