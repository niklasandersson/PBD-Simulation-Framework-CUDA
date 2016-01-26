#ifndef GLOBALS_H
#define GLOBALS_H


surface<void, cudaSurfaceType2D> positions4;
surface<void, cudaSurfaceType2D> predictedPositions4;
surface<void, cudaSurfaceType2D> velocities4;
surface<void, cudaSurfaceType2D> colors4;

surface<void, cudaSurfaceType2D> positions4Copy;
surface<void, cudaSurfaceType2D> predictedPositions4Copy;
surface<void, cudaSurfaceType2D> velocities4Copy;
surface<void, cudaSurfaceType2D> colors4Copy;

unsigned int* d_cellIds_in;
unsigned int* d_cellIds_out;

unsigned int* d_particleIds_in;
unsigned int* d_particleIds_out;

void* d_sortTempStorage = nullptr;
size_t sortTempStorageBytes = 0;

unsigned int* d_cellStarts;
unsigned int* d_cellEndings;

unsigned int* d_neighbours;
unsigned int* d_neighbourCounters;
unsigned int* d_contactCounters;
int* d_contactConstraintSucces;
int* d_contactConstraintParticleUsed;

float* d_densities;
float* d_lambdas;
float4* d_deltaPositions;
float4* d_externalForces;
float3* d_omegas;

struct Bound {
  float min;
  float max;
};

struct Bounds {
  Bound x;
  Bound y;
  Bound z;
};

struct SimulationParameters{
  unsigned int numberOfParticles;
  unsigned int textureWidth;
  unsigned int maxNeighboursPerParticle;
  unsigned int maxContactConstraints;
  unsigned int maxPossibleContactConstraints;
  unsigned int maxGrid;
  unsigned int maxParticles;
  float particleRadius;
  float particleDiameter;
  float deltaT;
  unsigned int kernelWidth;
	float restDensity;
  Bounds bounds;
  unsigned int randomStart;
  float kernelWidthDensity;
  float kernelWidthViscosity;
  float kSCorr;
  int nSCorr;
  float qSCorr;
  float cViscosity;
  float forcesVelocityDamping;
  float forcesPositionDamping;
  float gravity;
};

__constant__ SimulationParameters params;

SimulationParameters simulationParameters;

struct DeviceBuffers {
  // Shared buffers
  float* d_densities;
  float4* d_positions;
  float4* d_predictedPositions;
  float4* d_velocities;
  float4* d_colors;

  float* d_densitiesCopy;
  float4* d_positionsCopy;
  float4* d_predictedPositionsCopy;
  float4* d_velocitiesCopy;
  float4* d_colorsCopy;

  float4* d_collisionDeltas;
};

DeviceBuffers deviceBuffers;

struct CudaCallParameters {
  dim3 blocksForParticleBased;
  dim3 threadsForParticleBased;

  dim3 blocksForContactBased;
  dim3 threadsForContactBased;

  dim3 blocksForGridBased;
  dim3 threadsForGridBased;
};

CudaCallParameters cudaCallParameters;

struct CudaCallFullRangeParameters {
  dim3 blocksForAllParticlesPossibleBased;
  dim3 threadsForAllParticlesPossibleBased;
};

CudaCallFullRangeParameters cudaCallFullRangeParameters;


#define MAX_NEIGHBOURS 64
#define KERNEL_WIDTH 3

#define FOR_EACH_PARTICLE cudaCallParameters.blocksForParticleBased,cudaCallParameters.threadsForParticleBased
#define FOR_EACH_CONTACT cudaCallParameters.blocksForContactBased,cudaCallParameters.threadsForContactBased
#define FOR_EACH_CELL cudaCallParameters.blocksForGridBased,cudaCallParameters.threadsForGridBased
#define FOR_ALL_POSSIBLE_PARTICLES cudaCallFullRangeParameters.blocksForAllParticlesPossibleBased,cudaCallFullRangeParameters.threadsForAllParticlesPossibleBased

#define M_PI 3.14159265359
#define M_E 2.71828182845

// #define GET_INDEX const unsigned int index = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
#define GET_INDEX const unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

#define GET_TEXTUREWIDTH const unsigned int textureWidth = params.textureWidth;

#define GET_INDEX_X_Y GET_INDEX \
                      GET_TEXTUREWIDTH \
                      const unsigned int x = (index % textureWidth) * sizeof(float4); \
                      const unsigned int y = index / textureWidth;        

#include "Communication.h"
Communication communication;

#endif // GLOBALS_H