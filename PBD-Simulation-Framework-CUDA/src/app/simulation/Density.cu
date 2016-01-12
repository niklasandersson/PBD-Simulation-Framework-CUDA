#include "Density.h"


Density::Density()
{
	// Separat/constraint?
	cudaInitializeKernels();
}


void Density::compute()
{
	cudaCallComputeLambda();
	cudaCallComputePositionDeltas();

}