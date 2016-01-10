#ifndef SIMULATION_H
#define SIMULATION_H

#include "cuda/Cuda.h"

#include "simulation/Collision.h"
#include "simulation\Density.h"

class Simulation {

public:
  Simulation();
  ~Simulation();

  void initialize();
  void step();
  void cleanup();

protected:

private:
  Cuda* cuda_;
  Collision* collision_;
	Density* density_;

};

#endif // SIMULATION_H