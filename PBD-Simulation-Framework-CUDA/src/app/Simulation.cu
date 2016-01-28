#include "Simulation.h"


void Simulation::initialize() {
  cuda_ = new Cuda(5, 0);
  fluid_ = new Fluid();
}


void Simulation::cleanup() {
  delete fluid_;
  delete cuda_;
}


void Simulation::step() {
  fluid_->compute();
  CUDA(cudaDeviceSynchronize());
}
