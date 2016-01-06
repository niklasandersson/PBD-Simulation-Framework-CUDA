#ifndef SIMULATION_H
#define SIMULATION_H

#include "cuda/Cuda.h"

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

};

#endif 