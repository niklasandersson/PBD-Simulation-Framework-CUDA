#include "Density.h"
float const EPSILON = 0.000000001f;
// GLÖM INTE cudaFree
// ---------------------------------
void initilizeDensity(Parameters* parameters) {
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_lambdas, parameters->deviceParameters.maxParticles * sizeof(float)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_deltaPositions, parameters->deviceParameters.maxParticles * sizeof(float4)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_omegas, parameters->deviceParameters.maxParticles * sizeof(float3)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_externalForces, parameters->deviceParameters.maxParticles * sizeof(float4)));
	cudaMemset(parameters->deviceBuffers.d_externalForces, 0.0f, parameters->deviceParameters.maxParticles * sizeof(float4));
}

__device__ float poly6(float4 pi,
  float4 pj,
  float kernelWidth)
{
  pi.w = 0.0f;
  pj.w = 0.0f;

  float distance = length(pi - pj);

	if (distance < 0 || distance > kernelWidth)
	{
		float numeratorTerm = pow(kernelWidth * kernelWidth - distance * distance, 3);
		return (315.0f * numeratorTerm * numeratorTerm) / (64.0f * M_PI * pow(kernelWidth, 9));
	}
	else
		return 0.0f;

  
}

__device__ float4 spiky(float4 pi,
  float4 pj,
  float kernelWidth) {

  pi.w = 0.0f;
  pj.w = 0.0f;
  float4 r = pi - pj;
  float distance = length(r);
	
	
	float numeratorTerm = pow(kernelWidth - distance, 3);
	float denominatorTerm = M_PI * pow(kernelWidth, 6) * (distance + 0.0000001f);
	//printf("numeratorTerm = %f ", numeratorTerm);
	//printf("denominatorTerm = %f ", denominatorTerm);

	return 45.0f * numeratorTerm / (denominatorTerm * r);

}

void cudaApplyDeltaPositions(Parameters* parameters)
{
		applyDeltaPositions<<< PARTICLE_BASED >>>(parameters->deviceParameters.numberOfParticles,
		parameters->deviceBuffers.d_predictedPositions,
		parameters->deviceBuffers.d_deltaPositions);
}

__global__ void applyDeltaPositions(const unsigned int numberOfParticles,
	float4* predictedPositions,
	float4* d_deltaPositions)
{
	GET_INDEX
	if (index < numberOfParticles) {
		//printf("d_deltaPositions.x = %f, d_deltaPositions.y = %f, d_deltaPositions.z = %f \n", d_deltaPositions[index].x, d_deltaPositions[index].y, d_deltaPositions[index].z);
		//predictedPositions[index] = predictedPositions[index] + d_deltaPositions[index];
	}
}



void cudaCallComputeViscosity(Parameters* parameters) {

	computeViscosity << < PARTICLE_BASED >> >(parameters->deviceParameters.numberOfParticles,
			parameters->deviceBuffers.d_predictedPositions,
			parameters->deviceBuffers.d_neighbours,
			parameters->deviceBuffers.d_neighbourCounters,
			parameters->deviceParameters.maxNeighboursPerParticle,
			parameters->deviceParameters.kernelWidth,
			parameters->deviceBuffers.d_velocities);
}


__global__ void computeViscosity(const unsigned int numberOfParticles,
	float4* predictedPositions,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	unsigned int maxNumberOfNeighbors,
	float kernelWidth,
	float4* velocities) {
	GET_INDEX

	float4 pi = predictedPositions[index];
	float4 vi = velocities[index];
	unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
	float4 vSum = make_float4(0.0, 0.0, 0.0, 0.0);
	float c = 0.001;

	for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
		unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		float4 pj = predictedPositions[neighborIndex];
		float4 vij = velocities[neighborIndex] - vi;
		vSum += vij*poly6(pi, pj, kernelWidth);
	}

	float4 vNew = vi + c*vSum;
	velocities[index] = vNew;
}


void cudaCallComputeOmega(Parameters* parameters) {

	computeOmega << < PARTICLE_BASED >> >(parameters->deviceParameters.numberOfParticles,
		parameters->deviceBuffers.d_predictedPositions,
		parameters->deviceBuffers.d_neighbours,
		parameters->deviceBuffers.d_neighbourCounters,
		parameters->deviceParameters.maxNeighboursPerParticle,
		parameters->deviceParameters.kernelWidth,
		parameters->deviceBuffers.d_velocities,
		parameters->deviceBuffers.d_omegas);

}

__global__ void computeOmega(const unsigned int numberOfParticles,
	float4* predictedPositions,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	unsigned int maxNumberOfNeighbors,
	float kernelWidth,
	float4* velocities,
	float3* omegas
	) {
	GET_INDEX

	float4 pi = predictedPositions[index];
	float4 vi = velocities[index];
	unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
	float3 omega = make_float3(0.0f, 0.0f, 0.0f);

	for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
		unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		float4 pj = predictedPositions[neighborIndex];
		float4 vij = velocities[neighborIndex] - vi;
		float3 vij3 = make_float3(vij.x, vij.y, vij.z);
		float4 spike = spiky(pi, pj, kernelWidth);

		float3 spike3 = make_float3(spike.x, spike.y, spike.z);
		omega += cross(vij3, spike3);
	}
	omegas[index] = omega;

}

void cudaCallComputeVorticity(Parameters* parameters) {

	computeVorticity << < PARTICLE_BASED >> >(parameters->deviceParameters.numberOfParticles,
		parameters->deviceBuffers.d_predictedPositions,
		parameters->deviceBuffers.d_neighbours,
		parameters->deviceBuffers.d_neighbourCounters,
		parameters->deviceParameters.maxNeighboursPerParticle,
		parameters->deviceParameters.kernelWidth,
		parameters->deviceBuffers.d_velocities,
		parameters->deviceBuffers.d_omegas,
		parameters->deviceBuffers.d_externalForces);

}

__global__ void computeVorticity(const unsigned int numberOfParticles,
	float4* predictedPositions,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	unsigned int maxNumberOfNeighbors,
	float kernelWidth,
	float4* velocities,
	float3* omegas,
	float4* externalForces
	) {
	GET_INDEX

	float4 pi = predictedPositions[index];
	float4 vi = velocities[index];
	float3 omegai = omegas[index];
	unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

	float3 gradient = make_float3(0.0f, 0.0f, 0.0f);
	const float EPSILON = 0.000001f;
	for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
		unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		//printf("neighborIndex = %i \n", neighborIndex);
		float4 pj = predictedPositions[neighborIndex];
		//printf("pj = %f, %f, %f, \n", pj.x, pj.y, pj.z);
		float3 omegaj = omegas[neighborIndex];
		printf("omegaj = %f, %f, %f, \n", omegaj.x, omegaj.y, omegaj.z);
		float4 vij = velocities[neighborIndex] - vi;
		//printf("vij= %f, %f, %f, \n", vij.x, vij.y, vij.z);
		float omegaLength = length(omegaj - omegai);
		float4 pij = pj - pi + EPSILON;

		gradient.x += omegaLength / pij.x;
		gradient.y += omegaLength / pij.y;
		gradient.z += omegaLength / pij.z;
		//printf("gradient = %f, %f, %f, \n", gradient.x, gradient.y, gradient.z);
	}

	float3 N = (1.0f / (length(gradient) + 0.00001f)) * gradient;
	float epsilon = 1.0f;
	float3 vorticity = epsilon * cross(N, omegas[index]);
	//if (vorticity.x > 10 || vorticity.y > 10 || vorticity.z > 10 || vorticity.x < -10 || vorticity.y < -10 || vorticity.z < -10)
		
	externalForces[index] = make_float4(vorticity.x, vorticity.y, vorticity.z, 0.0f);
	//printf("vorticity.x = %f, vorticity.y = %f, vorticity.z = %f \n", externalForces[index].x, externalForces[index].y, externalForces[index].z);
}

void cudaCallComputeDeltaPositions(Parameters* parameters) {

	computeDeltaPositions << < PARTICLE_BASED >> >(parameters->deviceParameters.numberOfParticles,
		parameters->deviceBuffers.d_predictedPositions,
		parameters->deviceBuffers.d_neighbours,
		parameters->deviceBuffers.d_neighbourCounters,
		parameters->deviceParameters.maxNeighboursPerParticle,
		parameters->deviceParameters.restDensity,
		parameters->deviceParameters.kernelWidth,
		parameters->deviceBuffers.d_lambdas,
		parameters->deviceBuffers.d_deltaPositions);
}


__global__ void computeDeltaPositions(const unsigned int numberOfParticles,
	float4* predictedPositions,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	unsigned int maxNumberOfNeighbors,
	float restDensity,
	float kernelWidth,
	float* lambdas,
	float4* deltaPositions
	) {
	GET_INDEX

		float4 pi = predictedPositions[index];
		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		float lambdai = lambdas[index];
		float4 deltaPosition = make_float4( 0.0f, 0.0f, 0.0f, 0.0f );
		float sCorr = 0.0f;
		float k = 1.0f;
		float n = 1.0f;

		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float4 pj = predictedPositions[neighborIndex];
			float lambdaj = lambdas[neighborIndex];
			float absQ = 0.1f*kernelWidth;
			float4 deltaQ = make_float4(1.0f, 1.0f, 1.0f, 0.0f) * absQ + pi;
			//sCorr = -k * pow(poly6(pi, pj, kernelWidth), n) / poly6(deltaQ, make_float4(0.0f, 0.0f, 0.0f, 0.0f), kernelWidth);

			deltaPosition += (lambdai + lambdaj) * spiky(pi, pj, kernelWidth);
		}

		deltaPositions[index] = deltaPosition / restDensity;
}

// ------------------------------------------

void cudaCallComputeLambda(Parameters* parameters) {

	computeLambda << < PARTICLE_BASED >> >(parameters->deviceParameters.numberOfParticles,
		parameters->deviceBuffers.d_predictedPositions,
		parameters->deviceBuffers.d_neighbours,
		parameters->deviceBuffers.d_neighbourCounters,
		parameters->deviceParameters.maxNeighboursPerParticle,
		parameters->deviceParameters.restDensity,
		parameters->deviceParameters.kernelWidth,
		parameters->deviceBuffers.d_lambdas);

}

__global__ void computeLambda(const unsigned int numberOfParticles,
  float4* predictedPositions,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors,
  float restDensity,
  float kernelWidth,
	float* lambdas
  ) {
  GET_INDEX

  float4 pi = predictedPositions[index];
  float ci = computeConstraintValue(index, pi, predictedPositions,
  restDensity, kernelWidth, neighbors, numberOfNeighbors, maxNumberOfNeighbors);

  float gradientValue = 0.0f;
	const float EPSILON = 0.00000001f;

  unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

  for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
    float4 pj = predictedPositions[neighborIndex];
    float4 gradient = -1.0f * spiky(pi, pj, kernelWidth) / restDensity;
		//printf("gradient.x = %f , gradient.y = %f , gradient.z = %f  \n", gradient.x, gradient.y, gradient.z);
    float gradientLength = length(make_float3(gradient.x, gradient.y, gradient.z));
		//printf("gradLength = %f \n", gradientLength);
		//printf("gradient.x = %f , gradient.y = %f , gradient.z = %f, gradLength = %f \n", gradient.x, gradient.y, gradient.z, gradientLength);
    gradientValue += gradientLength * gradientLength;
  }

  float4 gradientAtSelf = computeGradientAtSelf(index, pi, predictedPositions,
    restDensity, kernelWidth, neighbors, numberOfNeighbors, maxNumberOfNeighbors);

	float gradientAtSelfLength = length(make_float3(gradientAtSelf.x, gradientAtSelf.y, gradientAtSelf.z));
  gradientValue += gradientAtSelfLength * gradientAtSelfLength;

	//if (gradientValue == 0.0f)
		//printf("gradientValue = %f \n", gradientValue);
	//printf("ci = %f \n", ci);
	lambdas[index] = -1.0f * ci / (gradientValue + EPSILON);
}

__device__ float computeConstraintValue(const unsigned int index,
  float4 pi,
  float4* predictedPositions,
  float restDensity,
  float kernelWidth,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors) {

  float density = 0.0f;
  unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

  for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
    float4 pj = predictedPositions[neighborIndex];
    density += poly6(pi, pj, kernelWidth);
  }

  return (density / restDensity) - 1.0f;
}

__device__ float4 computeGradientAtSelf(const unsigned int index,
  float4 pi,
  float4* predictedPositions,
  float restDensity,
  float kernelWidth,
  unsigned int* neighbors,
  unsigned int* numberOfNeighbors,
  unsigned int maxNumberOfNeighbors) {

  float4 gradient = make_float4(0, 0, 0, 0);
  unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

  for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
    float4 pj = predictedPositions[neighborIndex];
    gradient += spiky(pi, pj, kernelWidth);
  }

  return gradient / restDensity;
}