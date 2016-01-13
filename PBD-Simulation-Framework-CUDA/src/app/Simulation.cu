#include "Simulation.h"


Simulation::Simulation() {
  
}


Simulation::~Simulation() {
 
}


void Simulation::initialize() {
  cuda_ = new Cuda(5, 0);
  collision_ = new Collision();
	density_ = new Density();
}


void Simulation::cleanup() {
  delete collision_;
	delete density_;
  delete cuda_;
}


void Simulation::step() {

  collision_->compute();
	
  cudaError_t error = cudaDeviceSynchronize();
	if (error != cudaSuccess)
	{
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}

}
