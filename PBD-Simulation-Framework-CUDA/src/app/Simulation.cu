#include "Simulation.h"


Simulation::Simulation() {
  
}


Simulation::~Simulation() {
 
}


void Simulation::initialize() {
  cuda_ = new Cuda(3, 0);
  collision_ = new Collision();
}


void Simulation::cleanup() {
  delete collision_;
  delete cuda_;
}


void Simulation::step() {

  collision_->compute();

  CUDA(cudaDeviceSynchronize());

}
