#ifndef COLLISION_H
#define COLLISION_H

#include <iostream>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Computable.h"
#include "cuda/Cuda_Helper_Math.h"

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/iterator/counting_iterator.h>

#include "Parameters.h"

#include "kernel/forces/ApplyForces.h"
#include "kernel/update/UpdatePositions.h"


class Fluid : public Cuda_Computable {

public:
  Fluid(Parameters* parameters);

  void compute() override;

protected:

private:
  Parameters* parameters_;

};


#endif // COLLISION_H