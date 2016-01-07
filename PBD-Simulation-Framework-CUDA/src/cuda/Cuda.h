#ifndef CUDA_H
#define CUDA_H

#include <iostream>
#include <sstream>
#include <stdexcept>

#include <GL/glew.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_gl_interop.h"

#include "Cuda_Call_Error.h"

#ifdef _DEBUG
#define CUDA_CALL_IMPL(file, line, cerr) if(cerr != cudaSuccess)\
                             {\
                              std::ostringstream os;\
                              os << "Cuda call failed @" << file << " " << line << ", Error code: " << cerr << std::endl;\
                              std::cout << os.str() << std::endl;\
                              throw cuda_call_error(os.str());\
                             }
#define CUDA(cerr) CUDA_CALL_IMPL(__FILE__, __LINE__, cerr)
#else
#define CUDA(cerr) cerr
#endif

class Cuda {

public:
  Cuda(const int computeCapabilityMajor, const int computeCapabilityMinor);
  ~Cuda();

protected:

private:
  int deviceId_;
  cudaDeviceProp properties_;

  int computeCapabilityMajor_;
  int computeCapabilityMinor_;

};

#endif // CUDA_H