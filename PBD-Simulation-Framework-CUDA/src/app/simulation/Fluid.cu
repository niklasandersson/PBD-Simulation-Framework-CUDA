#include "Fluid.h"


Fluid::Fluid(Parameters* parameters) : parameters_(parameters){

  //cudaInitializeKernels();
  initializeCollision(parameters_);
	initilizeDensity(parameters_);
}


void Fluid::compute() {

  cudaCallApplyForces(parameters_);
		//std::cout << "applyForces" << std::endl;

  hashSortReorder(parameters_);
	//std::cout << "hashSortReorder" << std::endl;

  solveCollisions(parameters_); // find neighbors && solve collisions
	//std::cout << "solveCollisions" << std::endl;

	cudaCallComputeLambda(parameters_);
	//std::cout << "cudaCallComputeLambda" << std::endl;

	//cudaCallComputeDeltaPositions(parameters_);
	//std::cout << "cudaCallComputeDeltaPositions" << std::endl;

	//cudaApplyCollision()

	//cudaApplyDeltaPositions(parameters_);
	//std::cout << "cudaApplyDeltaPositions" << std::endl;

	//cudaCallComputeOmega(parameters_);

	//cudaCallComputeVorticity(parameters_);

	//cudaCallComputeViscosity(parameters_);

	cudaCallUpdatePositions(parameters_); // vel = (pj - pi)/dt && pos = predPos
	//std::cout << "cudaCallUpdatePositions" << std::endl;

	//char a;
	//std::cin >> a;

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