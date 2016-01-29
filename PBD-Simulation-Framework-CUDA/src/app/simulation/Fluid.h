#ifndef COLLISION_H
#define COLLISION_H

#include <iostream>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Computable.h"
#include "cuda/Cuda_Helper_Math.h"

#include "opengl/GL_Shared.h"

#include "parser/Config.h"
#include "kernel/kernels.h"

class Fluid : public Cuda_Computable {

public:
  Fluid();
  ~Fluid();
  
  void compute() override;

protected:

private:

};

#endif // COLLISION_H