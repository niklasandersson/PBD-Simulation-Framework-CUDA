#include "Simulation.h"


Simulation::Simulation() {
  
}


Simulation::~Simulation() {
 
}


void Simulation::initialize() {
  cuda_ = new Cuda(5, 0);
  parameters_ = new Parameters();
  fluid_ = new Fluid(parameters_);
}


void Simulation::cleanup() {
  //delete collision_;
  //delete parameters_;
  delete parameters_;
  delete cuda_;
}


void Simulation::step() {
  
  parameters_->update();

  fluid_->compute();

  CUDA(cudaDeviceSynchronize());

}
