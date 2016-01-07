#include "Simulation.h"


Simulation::Simulation() {

}


Simulation::~Simulation() {

}


void Simulation::initialize() {
  cuda_ = new Cuda(2, 0);
  cuda_->initialize();
}


void Simulation::cleanup() {
  cuda_->cleanup();
  delete cuda_;
}


void Simulation::step() {

}
