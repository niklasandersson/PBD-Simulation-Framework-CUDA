#include "Fluid.h"


Fluid::Fluid(Parameters* parameters) : parameters_(parameters){

  cudaInitializeKernels();
  //initializeCollision(parameters_);
	//initilizeDensity(parameters_);
}

#include <iostream>
void Fluid::compute() {
  /*
  cudaCallApplyForces(parameters_);

  hashSortReorder(parameters_);

  findNeighboursAndSolveCollisions(parameters_);

  cudaCallUpdatePositions(parameters_);

	cudaCallComputeLambda(parameters_);
	
	cudaCallComputeDeltaPositions(parameters_);

    */
  initializeFrame();

  cudaCallApplyForces();
 
  cudaCallInitializeCellIds();

  sortIds();

  reorderStorage();

  cudaCallResetCellInfo();

  cudaCallComputeCellInfo();

  collisionHandling();

  cudaCallUpdatePositions();

	cudaCallComputeLambda();

	cudaCallComputeDeltaPositions();
	
	cudaCallApplyDeltaPositions();

	cudaCallComputeOmegas();

	cudaCallComputeVorticity();

	cudaComputeViscosity();
}