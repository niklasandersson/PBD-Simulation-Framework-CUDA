#include "Cuda.h"

Cuda::Cuda(const int computeCapabilityMajor,
           const int computeCapabilityMinor)
  : computeCapabilityMajor_(computeCapabilityMajor)
  , computeCapabilityMinor_(computeCapabilityMinor)
{

  memset(&properties_, 0, sizeof(properties_));

  properties_.major = computeCapabilityMajor;
  properties_.minor = computeCapabilityMinor;

  cudaChooseDevice(&deviceId_, &properties_);

  cudaGLSetGLDevice(deviceId_);
 
}