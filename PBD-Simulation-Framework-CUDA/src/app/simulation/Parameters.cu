#include "Parameters.h"


Parameters::Parameters() {
  initializeSharedBuffer(deviceBuffers.d_densities, "d_densities");
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

  update();
}


Parameters::~Parameters() {

}


void Parameters::update() {
  auto glShared = GL_Shared::getInstance();
  const unsigned int numberOfParticles = *glShared.get_unsigned_int_value("numberOfParticles");
  const unsigned int textureWidth = glShared.get_texture("positions4")->width_;
  const unsigned int maxGrid = *glShared.get_unsigned_int_value("maxGrid");
  const unsigned int maxParticles = *glShared.get_unsigned_int_value("maxParticles");

  deviceParameters.numberOfParticles = numberOfParticles;
  deviceParameters.textureWidth = textureWidth;
  deviceParameters.maxContactsPerParticle = 32;
  deviceParameters.maxContactConstraints = deviceParameters.maxContactsPerParticle * deviceParameters.numberOfParticles;
  deviceParameters.maxGrid = maxGrid;
  deviceParameters.maxParticles = maxParticles;
  deviceParameters.deltaT = 0.01f;
  deviceParameters.particleRadius = 0.5f;
  deviceParameters.particleDiameter = 2.0f * deviceParameters.particleRadius;

  unsigned int threadsPerBlock = 128;
  cudaCallParameters.blocksParticleBased = dim3((deviceParameters.numberOfParticles)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsParticleBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksContactBased = dim3((deviceParameters.maxContactConstraints)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsContactBased = dim3(threadsPerBlock, 1, 1);

  cudaCallParameters.blocksGridBased = dim3((deviceParameters.maxGrid)/threadsPerBlock, 1, 1);
  cudaCallParameters.threadsGridBased = dim3(threadsPerBlock, 1, 1);
}