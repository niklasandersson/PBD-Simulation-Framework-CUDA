#include "Fluid.h"


Fluid::Fluid() {
  cudaInitializeKernels();
}


Fluid::~Fluid() {
  cudaCleanupKernels();
}


void Fluid::compute() {
  Config& config = Config::getInstance();

  initializeFrame();

  if( *GL_Shared::getInstance().get_unsigned_int_value("numberOfParticles") == 0 ) return;

  if( config.getValue<bool>("Application.Simulation.Active.applyForces") ) {
    cudaCallApplyForces();
  }
  
  cudaCallInitializeCellIds();

  sortIds();

  reorderStorage();
   
  cudaCallResetCellInfo();

  cudaCallComputeCellInfo();

  const int collisionType = config.getValue<int>("Application.Simulation.Collision.collisionType"); 

  if( collisionType == 1 ) {
    cudaCallResetContacts();
  }

  if( config.getValue<bool>("Application.Simulation.Active.findContacts") ) {
    cudaCallFindContacts();
  }
  
  if( config.getValue<bool>("Application.Simulation.Active.findNeighbours") ) {
    cudaCallFindNeighbours();
  }

  const unsigned int solverIterations = config.getValue<unsigned int>("Application.Simulation.Iterations.solverIterations");
  for(unsigned int i=0; i<solverIterations; i++) {
     
    if( config.getValue<bool>("Application.Simulation.Active.computeLambda") ) {
	    cudaCallComputeLambda();
    }

    if( config.getValue<bool>("Application.Simulation.Active.computeDeltaPositions") ) {
	    cudaCallComputeDeltaPositions();
    } 

    const unsigned int stabilizationIterations = config.getValue<unsigned int>("Application.Simulation.Iterations.stabilizationIterations");
    for(unsigned int j=0; j<stabilizationIterations; j++) {

      if( collisionType == 0 ) {
        cudaCallSolveCollisions();
      } else if( collisionType == 1 ) {
        cudaCallResetContactConstraintSuccess();
        const unsigned int maxBatches = config.getValue<unsigned int>("Application.Simulation.Collision.maxCollisionBatches");
        for(unsigned int b=0; b<maxBatches; b++) {
          cudaCallResetContactConstraintParticleUsed();
          cudaCallSetupCollisionConstraintBatches();
          cudaCallSetupCollisionConstraintBatchesCheck();
        }
      }
    }
    
    if( config.getValue<bool>("Application.Simulation.Active.applyDeltaPositions") ) {
	    cudaCallApplyDeltaPositions();
    }

  }

  if( config.getValue<bool>("Application.Simulation.Active.updatePositions") ) {
	  cudaCallUpdatePositions();
  }

  if( config.getValue<bool>("Application.Simulation.Active.computeOmegas") ) {
	  cudaCallComputeOmegas();
  } 

	if( config.getValue<bool>("Application.Simulation.Active.computeVorticity") ) {
	  cudaCallComputeVorticity();
  }
	
	if( config.getValue<bool>("Application.Simulation.Active.computeViscosity") ) {
	  cudaComputeViscosity();
  }
  
}