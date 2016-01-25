#include "Fluid.h"


Fluid::Fluid(Parameters* parameters) : parameters_(parameters){
  cudaInitializeKernels();
}


void Fluid::compute() {
  Config& config = Config::getInstance();

  initializeFrame();

  if( config.getValue<bool>("Application.Sim.applyForces") ) {
    cudaCallApplyForces();
  }
  
  cudaCallInitializeCellIds();

  sortIds();

  reorderStorage();
   
  cudaCallResetCellInfo();

  cudaCallComputeCellInfo();

  const int collisionType = config.getValue<int>("Application.Sim.collisionType"); 

  if( collisionType == 1 ) {
    cudaCallResetContacts();
  }

  if( config.getValue<bool>("Application.Sim.findContacts") ) {
    cudaCallFindContacts();
  }
  
  if( config.getValue<bool>("Application.Sim.findNeighbours") ) {
    cudaCallFindNeighbours();
  }

  const unsigned int solverIterations = config.getValue<unsigned int>("Application.Sim.solverIterations");
  for(unsigned int i=0; i<solverIterations; i++) {
     
    if( config.getValue<bool>("Application.Sim.computeLambda") ) {
	    cudaCallComputeLambda();
    }

    if( config.getValue<bool>("Application.Sim.computeDeltaPositions") ) {
	    cudaCallComputeDeltaPositions();
    } 

    const unsigned int stabilizationIterations = config.getValue<unsigned int>("Application.Sim.stabilizationIterations");
    for(unsigned int j=0; j<stabilizationIterations; j++) {

      if( collisionType == 0 ) {
        cudaCallSolveCollisions();
      } else if( collisionType == 1 ) {
        cudaCallResetContactConstraintSuccess();
        const unsigned int maxBatches = config.getValue<unsigned int>("Application.Sim.maxCollisionBatches");
        for(unsigned int b=0; b<maxBatches; b++) {
          cudaCallResetContactConstraintParticleUsed();
          cudaCallSetupCollisionConstraintBatches();
          cudaCallSetupCollisionConstraintBatchesCheck();
        }
      }
    }
    
    if( config.getValue<bool>("Application.Sim.applyDeltaPositions") ) {
	    cudaCallApplyDeltaPositions();
    }

  }

  if( config.getValue<bool>("Application.Sim.updatePositions") ) {
	  cudaCallUpdatePositions();
  }

  if( config.getValue<bool>("Application.Sim.computeOmegas") ) {
	  cudaCallComputeOmegas();
  } 

	if( config.getValue<bool>("Application.Sim.computeVorticity") ) {
	  cudaCallComputeVorticity();
  }
	
	if( config.getValue<bool>("Application.Sim.computeViscosity") ) {
	  cudaComputeViscosity();
  }
  
  if( config.getValue<bool>("Application.Sim.break") ) {
    std::cout << "Break, type character: ";
    std::cin.get();
  } 

}