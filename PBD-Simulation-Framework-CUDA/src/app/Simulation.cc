#include "Simulation.h"


Simulation::Simulation() {

}


Simulation::~Simulation() {

}


void Simulation::initialize() {
  cuda_ = new Cuda(2, 0);
}


void Simulation::cleanup() {
  delete cuda_;
}


void Simulation::step() {

}
