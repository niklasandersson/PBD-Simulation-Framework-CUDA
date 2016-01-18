#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <stdlib.h> 
#include <time.h> 

#include "Kernels.h"
#include "Globals.h"
#include "SortReorder.h"
#include "Collision.h"

// --------------------------------------------------------------------------

void initializeFrame() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  const unsigned int maxGrid = *glShared.get_unsigned_int_value("maxGrid");
  const unsigned int maxParticles = *glShared.get_unsigned_int_value("maxParticles");

  simulationParameters.numberOfParticles = numberOfParticles;
  simulationParameters.textureWidth = textureWidth;
  simulationParameters.maxNeighboursPerParticle = 20;
  simulationParameters.maxContactConstraints = simulationParameters.maxNeighboursPerParticle * simulationParameters.numberOfParticles;
  simulationParameters.maxGrid = maxGrid;
  simulationParameters.maxParticles = maxParticles;
  simulationParameters.deltaT = 0.01f;
  simulationParameters.particleRadius = 0.5f;
  simulationParameters.particleDiameter = 2.0f * simulationParameters.particleRadius;
  simulationParameters.kernelWidth = 3;

  simulationParameters.bounds.x.min = 25.0f;
  simulationParameters.bounds.x.max = 40.0f - 1.5f;
  simulationParameters.bounds.y.min = 1.5f;
  simulationParameters.bounds.y.max = 64.0f - 1.5f;
  simulationParameters.bounds.z.min = 25.0f;
  simulationParameters.bounds.z.max = 40.0f - 1.5f;

  simulationParameters.randomStart = rand() % simulationParameters.maxNeighboursPerParticle;

  CUDA(cudaMemcpyToSymbol(params, &simulationParameters, sizeof(SimulationParameters)));

  unsigned int threadsPerBlock = 128;
  cudaCallParameters.blocksForParticleBased = dim3((simulationParameters.numberOfParticles)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsForParticleBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksForContactBased = dim3((simulationParameters.maxContactConstraints)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsForContactBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksForGridBased = dim3((simulationParameters.maxGrid)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsForGridBased = dim3(threadsPerBlock, 1, 1);
}

// --------------------------------------------------------------------------

void initializeShared() {
  initializeSharedTexture(positions4, "positions4");
  initializeSharedTexture(predictedPositions4, "predictedPositions4");
  initializeSharedTexture(velocities4, "velocities4");
  initializeSharedTexture(colors4, "colors4");

  initializeSharedTexture(positions4Copy, "positions4Copy");
  initializeSharedTexture(predictedPositions4Copy, "predictedPositions4Copy");
  initializeSharedTexture(velocities4Copy, "velocities4Copy");
  initializeSharedTexture(colors4Copy, "colors4Copy");

  initializeSharedBuffer(d_densities, "d_densities");
}

// --------------------------------------------------------------------------

void cudaInitializeKernels() {
   srand(time(NULL));

  initializeFrame();
  initializeShared();
  initializeSortReorder();
  initializeCollision();
}

// --------------------------------------------------------------------------


#endif // INITIALIZE_H