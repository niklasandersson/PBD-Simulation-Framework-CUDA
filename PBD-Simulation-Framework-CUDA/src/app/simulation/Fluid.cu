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

  cudaCallFindContacts();

  cudaCallFindNeighbours();
	
  const unsigned int solverIterations = 1;
  for(unsigned int i=0; i<solverIterations; i++) {

	  cudaCallComputeLambda();
		char a;
		//std::cin >> a;
	  cudaCallComputeDeltaPositions();	
    
    const unsigned int stabilizationIterations = 1;
    for(unsigned int j=0; j<stabilizationIterations; j++) {
      cudaCallSolveCollisions();
    }
  
    cudaCallApplyDeltaPositions();

  }
  cudaCallUpdatePositions(); 


	cudaCallComputeOmegas();

	cudaCallComputeVorticity();

	cudaComputeViscosity();
    
  
}