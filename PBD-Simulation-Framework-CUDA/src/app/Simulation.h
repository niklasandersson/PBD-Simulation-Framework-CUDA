#ifndef SIMULATION_H
#define SIMULATION_H

#include "cuda/Cuda.h"

#include "simulation/Fluid.h"


class Simulation {

public:
  Simulation() = default;
  ~Simulation() = default;

  void initialize();
  void cleanup();
  void step();

protected:

private:
  Cuda* cuda_;
  Fluid* fluid_;

};


#endif // SIMULATION_H