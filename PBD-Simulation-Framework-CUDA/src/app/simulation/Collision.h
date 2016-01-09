#ifndef COLLISION_H
#define COLLISION_H

#include <iostream>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Computable.h"
#include "cuda/Cuda_Helper_Math.h"

#include "kernel/AddKernel.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

class Collision : public Cuda_Computable {

public:
  Collision();

  void compute() override;

protected:

private:

};

#endif // COLLISION_H