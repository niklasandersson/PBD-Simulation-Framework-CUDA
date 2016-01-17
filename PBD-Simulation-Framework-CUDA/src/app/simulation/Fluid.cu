#include "Fluid.h"


Fluid::Fluid(Parameters* parameters) : parameters_(parameters){

  //cudaInitializeKernels();
  initializeCollision(parameters_);
	initilizeDensity(parameters_);
}


void Fluid::compute() {

  cudaCallApplyForces(parameters_);

  hashSortReorder(parameters_);

  solveCollisions(parameters_); // find neighbors && solve collisions

	cudaCallComputeLambda(parameters_);
	
	cudaCallComputeDeltaPositions(parameters_);

	//cudaApplyCollision()

	cudaApplyDeltaPositions(parameters_);

	cudaCallComputeOmega(parameters_);

	cudaCallComputeVorticity(parameters_);

	//cudaCallComputeViscosity(parameters_);

	cudaCallUpdatePositions(parameters_); // vel = (pj - pi)/dt && pos = predPos


	char a;
	std::cin >> a;

	

	

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