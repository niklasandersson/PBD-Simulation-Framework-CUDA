#include "Cuda.h"


Cuda::Cuda(const int computeCapabilityMajor,
           const int computeCapabilityMinor)
  : computeCapabilityMajor_(computeCapabilityMajor)
  , computeCapabilityMinor_(computeCapabilityMinor)
{
  memset(&properties_, 0, sizeof(properties_));

  properties_.major = computeCapabilityMajor_;
  properties_.minor = computeCapabilityMinor_;

  CUDA(cudaChooseDevice(&deviceId_, &properties_));

  CUDA(cudaGLSetGLDevice(deviceId_));
}


Cuda::~Cuda() {
  CUDA(cudaDeviceReset());
}

