#ifndef SIMULATION_H
#define SIMULATION_H

#include "cuda/Cuda.h"

#include "simulation/Parameters.h"
#include "simulation/Fluid.h"


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
  Parameters* parameters_;
  Fluid* fluid_;

};

#endif // SIMULATION_H