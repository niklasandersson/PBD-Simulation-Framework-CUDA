#ifndef DENSITY_H
#define DENSITY_H

#include <iostream>

#include "cuda/Cuda.h"
#include "cuda/Cuda_Computable.h"
#include "cuda/Cuda_Helper_Math.h"

#include "kernel/kernels.h"

class Density : public Cuda_Computable {

public:
	Density();

	void compute() override;

protected:

private:

};

#endif // COLLISION_H