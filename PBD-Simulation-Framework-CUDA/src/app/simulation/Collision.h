#ifndef COLLISION_H
#define COLLISION_H

#include <iostream>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Computable.h"
#include "cuda/Cuda_Helper_Math.h"

#include "kernel/AddKernel.h"

class Collision : public Cuda_Computable {

public:
  Collision();

  void compute() override;

protected:

private:

};

#endif // COLLISION_H