#ifndef INITIALIZE_H
#define INITIALIZE_H

#include <stdlib.h> 
#include <time.h> 

#include "Kernels.h"
#include "Globals.h"
#include "SortReorder.h"
#include "Collision.h"
#include "Density.h"
#include "Communication.h"

// --------------------------------------------------------------------------

void initializeFrame() {
  Events::addParticle.execute_calls();

  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  const unsigned int maxGrid = *glShared.get_unsigned_int_value("maxGrid");
  const unsigned int maxParticles = *glShared.get_unsigned_int_value("maxParticles");

  simulationParameters.numberOfParticles = numberOfParticles;
  simulationParameters.textureWidth = textureWidth;
  simulationParameters.maxNeighboursPerParticle = MAX_NEIGHBOURS;
  simulationParameters.maxContactConstraints = simulationParameters.maxNeighboursPerParticle * simulationParameters.numberOfParticles;
  simulationParameters.maxGrid = maxGrid;
  simulationParameters.maxParticles = maxParticles;
  simulationParameters.maxPossibleContactConstraints = simulationParameters.maxNeighboursPerParticle * simulationParameters.maxParticles;
  simulationParameters.deltaT = 0.01f;
  simulationParameters.particleRadius = 0.5f;
  simulationParameters.particleDiameter = 2.0f * simulationParameters.particleRadius;
  simulationParameters.kernelWidth = KERNEL_WIDTH;
  simulationParameters.kernelWidthDensity = simulationParameters.particleDiameter * 3.0f;
	simulationParameters.restDensity = 1.0f;

  simulationParameters.bounds.x.min = 25.0f;
  simulationParameters.bounds.x.max = 45.0f - 1.5f;
  simulationParameters.bounds.y.min = 1.5f;
  simulationParameters.bounds.y.max = 64.0f - 1.5f;
  simulationParameters.bounds.z.min = 25.0f;
  simulationParameters.bounds.z.max = 45.0f - 1.5f;

  simulationParameters.randomStart = rand() % simulationParameters.maxNeighboursPerParticle;

  CUDA(cudaMemcpyToSymbol(params, &simulationParameters, sizeof(SimulationParameters)));

  unsigned int threadsPerBlock = 128;
  
  cudaCallParameters.blocksForParticleBased = dim3(std::ceil(simulationParameters.numberOfParticles/(float)threadsPerBlock), 1, 1);
  cudaCallParameters.threadsForParticleBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksForContactBased = dim3(std::ceil(simulationParameters.maxContactConstraints/(float)threadsPerBlock), 1, 1);
  cudaCallParameters.threadsForContactBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksForGridBased = dim3(std::ceil(simulationParameters.maxGrid/(float)threadsPerBlock), 1, 1);
  cudaCallParameters.threadsForGridBased = dim3(threadsPerBlock, 1, 1);

  cudaCallFullRangeParameters.blocksForAllParticlesPossibleBased = dim3(std::ceil(simulationParameters.maxParticles/(float)threadsPerBlock), 1, 1);
  cudaCallFullRangeParameters.threadsForAllParticlesPossibleBased = dim3(threadsPerBlock, 1, 1);  
}

// --------------------------------------------------------------------------

void initializeShared() {
  initializeSharedBuffer(deviceBuffers.d_densities, "d_densities");
  initializeSharedBuffer(deviceBuffers.d_positions, "d_positions");
  initializeSharedBuffer(deviceBuffers.d_predictedPositions, "d_predictedPositions");
  initializeSharedBuffer(deviceBuffers.d_velocities, "d_velocities");
  initializeSharedBuffer(deviceBuffers.d_colors, "d_colors");

  initializeSharedBuffer(deviceBuffers.d_densitiesCopy, "d_densitiesCopy");
  initializeSharedBuffer(deviceBuffers.d_positionsCopy, "d_positionsCopy");
  initializeSharedBuffer(deviceBuffers.d_predictedPositionsCopy, "d_predictedPositionsCopy");
  initializeSharedBuffer(deviceBuffers.d_velocitiesCopy, "d_velocitiesCopy");
  initializeSharedBuffer(deviceBuffers.d_colorsCopy, "d_colorsCopy");
  initializeSharedBuffer(deviceBuffers.d_collisionDeltas, "d_collisionDeltas");
  
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
  communication.initialize();

  initializeFrame();
  initializeShared();
  initializeSortReorder();
  initializeCollision();
	initializeDensity();
}

// --------------------------------------------------------------------------


#endif // INITIALIZE_H