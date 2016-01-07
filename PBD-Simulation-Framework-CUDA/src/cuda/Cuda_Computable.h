#ifndef CUDA_COMPUTABLE_H
#define CUDA_COMPUTABLE_H

#include "cuda/Cuda.h"

#include "opengl/GL_Shared.h"

class Cuda_Computable {

public:
  Cuda_Computable() = default;

  virtual ~Cuda_Computable() = default;

  virtual void compute() = 0;

protected:
  GL_Shared& glShared_ = GL_Shared::getInstance();

private:
  

};

#endif // CUDA_COMPUTABLE_H