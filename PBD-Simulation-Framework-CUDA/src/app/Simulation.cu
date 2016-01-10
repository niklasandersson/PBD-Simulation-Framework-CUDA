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
	density_->compute();

  CUDA(cudaDeviceSynchronize());

}
