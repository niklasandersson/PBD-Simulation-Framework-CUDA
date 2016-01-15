#include "Fluid.h"


Fluid::Fluid(Parameters* parameters) : parameters_(parameters){

  //cudaInitializeKernels();

}


void Fluid::compute() {

  cudaCallApplyForces(parameters_);

  cudaCallUpdatePositions(parameters_);

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