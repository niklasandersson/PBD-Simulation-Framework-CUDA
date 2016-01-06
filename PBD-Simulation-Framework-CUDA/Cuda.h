#ifndef CUDA_H
#define CUDA_H

#include <iostream>

#include <GL/glew.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

class Cuda {

public:
  Cuda(const int computeCapabilityMajor = 2, const int computeCapabilityMinor = 0);

protected:

private:
  int deviceId_;
  cudaDeviceProp properties_;

  int computeCapabilityMajor_;
  int computeCapabilityMinor_;

};

#endif