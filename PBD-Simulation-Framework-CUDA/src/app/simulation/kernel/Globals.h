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

struct SimulationParameters{
  unsigned int numberOfParticles;
  unsigned int textureWidth;
  unsigned int maxNeighboursPerParticle;
  unsigned int maxContactConstraints;
  unsigned int maxGrid;
  unsigned int maxParticles;
  float particleRadius;
  float particleDiameter;
  float deltaT;
  unsigned int kernelWidth;
	float restDensity;
};

__constant__ SimulationParameters params;

SimulationParameters simulationParameters;

struct CudaCallParameters {
  dim3 blocksForParticleBased;
  dim3 threadsForParticleBased;

  dim3 blocksForContactBased;
  dim3 threadsForContactBased;

  dim3 blocksForGridBased;
  dim3 threadsForGridBased;
};

CudaCallParameters cudaCallParameters;

#define FOR_EACH_PARTICLE cudaCallParameters.blocksForParticleBased,cudaCallParameters.threadsForParticleBased
#define FOR_EACH_CONTACT cudaCallParameters.blocksForContactBased,cudaCallParameters.threadsForContactBased
#define FOR_EACH_CELL cudaCallParameters.blocksForGridBased,cudaCallParameters.threadsForGridBased

#define M_PI 3.14159265359

#define GET_INDEX const unsigned int index = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
#define GET_TEXTUREWIDTH const unsigned int textureWidth = params.textureWidth;
#define GET_INDEX_X_Y GET_INDEX \
                      GET_TEXTUREWIDTH \
                      const unsigned int x = (index % textureWidth) * sizeof(float4); \
                      const unsigned int y = index / textureWidth;                 

#endif // GLOBALS_H