#ifndef DENSITY_H
#define DENSITY_H

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "Kernels.h"
#include "Globals.h"

void initializeDensity() {
	
	CUDA(cudaMalloc((void**)&d_lambdas, simulationParameters.maxParticles * sizeof(float)));
	CUDA(cudaMalloc((void**)&d_deltaPositions, simulationParameters.maxParticles * sizeof(float4)));
	CUDA(cudaMalloc((void**)&d_externalForces, simulationParameters.maxParticles * sizeof(float4)));
	CUDA(cudaMalloc((void**)&d_omegas, simulationParameters.maxParticles * sizeof(float3)));
	cudaMemset(d_externalForces, 0.0f, simulationParameters.maxParticles* sizeof(float4));
	simulationParameters.restDensity = 1000.0f;
}

__global__ void clearAllTheCrap() {
  GET_INDEX_X_Y

    float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  surf2Dwrite(result, velocities4, x, y);
  surf2Dwrite(result, predictedPositions4, x, y);
  surf2Dwrite(result, positions4, x, y);
}

void callClearAllTheCrap() {
  clearAllTheCrap << <FOR_EACH_PARTICLE >> >();
}


__device__ __forceinline__ float poly6(float4 pi, float4 pj)
{
	float kernelWidth = (float) params.kernelWidth;
	pi.w = 0.0f;
	pj.w = 0.0f;

	float distance = length(make_float3(pi - pj));
	float numeratorTerm = kernelWidth * kernelWidth - distance * distance;
	
  return (315.0f * numeratorTerm * numeratorTerm * numeratorTerm) / (64.0f * M_PI * 19683);
}

__device__ __forceinline__ float4 spiky(float4 pi, float4 pj) {

	unsigned int kernelWidth = params.kernelWidth;
	float4 r = pi - pj;
	float distance = length(make_float3(r.x, r.y, r.z));

	float numeratorTerm = kernelWidth - distance;
  float denominatorTerm = M_PI * powf(kernelWidth, 6) * (distance + 0.00001f);

  return 45.0f * numeratorTerm * numeratorTerm / denominatorTerm;
}
// ---------------------------------------------------------------------------------------

__device__ float computeConstraintValue(float4 pi,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors) {
	GET_INDEX_X_Y

	float density = 0.0f;
	unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
	float restDensity = params.restDensity;
	unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;

	for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
		unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		float4 pj;
		float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
		float neighborY = (neighborIndex / textureWidth);
		
		surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
		density += poly6(pi, pj);
	}

	return (density / restDensity) - 1.0f;
}

__device__ float4 computeGradientAtSelf(float4 pi,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors
	) {
	GET_INDEX_X_Y
	unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
	float restDensity = params.restDensity;

	float4 gradient = make_float4(0, 0, 0, 0);
	unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

	for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
		unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		float4 pj;
		float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
		float neighborY = (neighborIndex / textureWidth);

		surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
		gradient += spiky(pi, pj);
	}

	return gradient / restDensity;
}

__device__ float4 part_computeGradientAtSelf(float4 pi, float4 pj) {

}


__global__ void computeLambda(unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	float* lambdas
	) {
	GET_INDEX_X_Y
		unsigned int numberOfParticles = params.numberOfParticles;
	if (index < numberOfParticles)
	{
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float restDensity = params.restDensity;
    float density = 0.0f;

		float gradientValue = 0.0f;
		const float EPSILON = 0.00000001f;
    float4 gradientAtSelf = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float4 pj;
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			//float4 gradient = -1.0f * spiky(pi, pj) / restDensity;
      float4 gradient = spiky(pi, pj) / restDensity;
			float gradientLength = length(make_float3(gradient.x, gradient.y, gradient.z));
			gradientValue += gradientLength * gradientLength;
      gradientAtSelf += gradient;

      density += poly6(pi, pj);
		}

		float gradientAtSelfLength = length(make_float3(gradientAtSelf.x, gradientAtSelf.y, gradientAtSelf.z));
		gradientValue += gradientAtSelfLength * gradientAtSelfLength;
    //gradientValue += gradientValue;

    float ci = (density / restDensity) - 1.0f;

		lambdas[index] = -1.0f * ci / (gradientValue + EPSILON);
	}
}

void cudaCallComputeLambda() {
	computeLambda << <FOR_EACH_PARTICLE >> >(d_neighbours, d_neighbourCounters, d_lambdas);
}

__global__ void computeDeltaPositions(unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	float* lambdas,
	float4* deltaPositions
	) {
	GET_INDEX_X_Y
		unsigned int numberOfParticles = params.numberOfParticles;
	if (index < numberOfParticles)
	{
		float restDensity = params.restDensity;
		unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		unsigned int kernelWidth = params.kernelWidth;

		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		float lambdai = lambdas[index];
		float4 deltaPosition = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
		float sCorr = 0.0f;
		float k = 1.0f;
		float n = 1.0f;

		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float4 pj;
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			if (isnan(pj.x) || isnan(pj.y) || isnan(pj.z))
				printf("IN computeDeltaPositions: pj = %f , %f , %f ...... computeDeltaPositions()  \n", pj.x, pj.y, pj.z);
			float lambdaj = lambdas[neighborIndex];
			float absQ = 0.1f*kernelWidth;
			float4 deltaQ = make_float4(1.0f, 1.0f, 1.0f, 0.0f) * absQ + pi;
			//sCorr = -k * pow(poly6(pi, pj, kernelWidth), n) / poly6(deltaQ, make_float4(0.0f, 0.0f, 0.0f, 0.0f), kernelWidth);

			deltaPosition += (lambdai + lambdaj) * spiky(pi, pj);
		}

		deltaPositions[index] = deltaPosition / restDensity;
		//if (isnan(deltaPositions[index].x) || isnan(deltaPositions[index].y) || isnan(deltaPositions[index].z))
			//printf("IN computeDeltaPositions: deltaPositions[index] = %f , %f , %f ...... at = computeDeltaPositions()  \n", deltaPositions[index].x, deltaPositions[index].y, deltaPositions[index].z);
	}
}


void cudaCallComputeDeltaPositions() {
	computeDeltaPositions << <FOR_EACH_PARTICLE >> >(d_neighbours, d_neighbourCounters, d_lambdas, d_deltaPositions);
}

__global__ void applyDeltaPositions(float4* d_deltaPositions)
{
	GET_INDEX_X_Y
		unsigned int numberOfParticles = params.numberOfParticles;
		float4 predictedPositions;
	surf2Dread(&predictedPositions, predictedPositions4, x, y);
	
	if (index < numberOfParticles) {
		if (isnan(d_deltaPositions[index].x) || isnan(d_deltaPositions[index].y) || isnan(d_deltaPositions[index].z))
			printf("IN APPLYDELTAPOS: d_deltaPositions.x = %f, d_deltaPositions.y = %f, d_deltaPositions.z = %f \n", d_deltaPositions[index].x, d_deltaPositions[index].y, d_deltaPositions[index].z);

		if (isnan(predictedPositions.x) || isnan(predictedPositions.y) || isnan(predictedPositions.z))
			printf("IN APPLYDELTAPOS: predictedPositions.x = %f, predictedPositions.y = %f, predictedPositions.z = %f \n", predictedPositions.x, predictedPositions.y, predictedPositions.z);

		float4 result = predictedPositions + d_deltaPositions[index];
		//printf("result.x = %f, result.y = %f, result.z = %f \n", result.x, result.y, result.z);

		surf2Dwrite(result, predictedPositions4, x, y);
	}
}

void cudaCallApplyDeltaPositions()
{
	applyDeltaPositions << < FOR_EACH_PARTICLE >> >(d_deltaPositions);
}

__global__ void computeVorticity(unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	float3* omegas,
	float4* externalForces
	) {
	GET_INDEX_X_Y
		unsigned int numberOfParticles = params.numberOfParticles;
	if (index < numberOfParticles)
	{

		unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		float4 vi;
		surf2Dread(&vi, velocities4, x, y);
		float3 omegai = omegas[index];
		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

		float3 gradient = make_float3(0.0f, 0.0f, 0.0f);
		const float EPSILON = 0.00000001f;
		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			//printf("IN COMPUTEVORTICITY: neighborIndex = %i \n", neighborIndex);
			float4 pj;
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			//printf("IN COMPUTEVORTICITY: pj = %f, %f, %f, \n", pj.x, pj.y, pj.z);
			float3 omegaj = omegas[neighborIndex];
			//printf("IN COMPUTEVORTICITY: omegaj = %f, %f, %f, \n", omegaj.x, omegaj.y, omegaj.z);
			float4 vj;
			surf2Dread(&vj, velocities4, neighborX, neighborY);
			float4 vij = vj - vi;
			//printf("IN COMPUTEVORTICITY: vij= %f, %f, %f, \n", vij.x, vij.y, vij.z);
			float omegaLength = length(omegaj - omegai);
			float4 pij = pj - pi + EPSILON;

			gradient.x += omegaLength / pij.x;
			gradient.y += omegaLength / pij.y;
			gradient.z += omegaLength / pij.z;
			//printf("IN COMPUTEVORTICITY: gradient = %f, %f, %f, \n", gradient.x, gradient.y, gradient.z);
		}

		float3 N = (1.0f / (length(gradient) + EPSILON)) * (gradient + EPSILON);
		float epsilon = 1.0f;
		float3 vorticity = epsilon * cross(N, omegas[index]);
		//if (vorticity.x > 10 || vorticity.y > 10 || vorticity.z > 10 || vorticity.x < -10 || vorticity.y < -10 || vorticity.z < -10)

		externalForces[index] = make_float4(vorticity.x, vorticity.y, vorticity.z, 0.0f);
		//printf("vorticity.x = %f, vorticity.y = %f, vorticity.z = %f \n", externalForces[index].x, externalForces[index].y, externalForces[index].z);
	}
}

void cudaCallComputeVorticity() {
	computeVorticity << <FOR_EACH_PARTICLE >> >(d_neighbours, d_neighbourCounters, d_omegas, d_externalForces);
}


__global__ void computeOmega(unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	float3* omegas
	) {
	GET_INDEX_X_Y
		unsigned int numberOfParticles = params.numberOfParticles;
	if (index < numberOfParticles)
	{
		unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		float4 vi;
		surf2Dread(&vi, velocities4, x, y);
		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		float3 omega = make_float3(0.0f, 0.0f, 0.0f);

		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {

			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float4 pj;
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);

			float4 vj;
			surf2Dread(&vj, velocities4, neighborX, neighborY);

			float4 vij = vj - vi;
			float3 vij3 = make_float3(vij.x, vij.y, vij.z);

			float4 spike = spiky(pi, pj);
			float3 spike3 = make_float3(spike.x, spike.y, spike.z);
			/*
			if (isnan(spike3.x))
			{
			printf("IN computeOmega: isnan at index %i X = %f \n ", index, spike3.x);
			}
			if (isnan(spike3.y))
			{
			printf("IN computeOmega: isnan at index %i Y = %f \n ", index, spike3.y);
			}
			if (isnan(spike3.z))
			{
			printf("IN computeOmega: isnan at index %i Z = %f \n ", index, spike3.z);
			}*/

			omega += cross(vij3, spike3);
			/*
			if (isnan(omega.x))
			{
			printf("IN computeOmega: isnan at neighbor %i X = %f \n ", i, omega.x);
			}
			if (isnan(omega.y))
			{
			printf("IN computeOmega: isnan at neighbor %i Y = %f \n ", i, omega.y);
			}
			if (isnan(omega.z))
			{
			printf("IN computeOmega: isnan at neighbor %i Z = %f \n ", i, omega.z);
			}*/
		}

		/*
		if (isnan(omega.z))
		printf("IN computeOmega: isnan at X index = %i \n ", index);
		else if (omega.y == omega.y)
		printf(" IN computeOmega: isnan at Y index = %i \n ", index);
		else if (omega.z == omega.z)
		printf("IN computeOmega: isnan at Z index = %i \n ", index);
		*/

		omegas[index] = omega;

	}
}

void cudaCallComputeOmegas() {
	computeOmega << <FOR_EACH_PARTICLE >> >(d_neighbours, d_neighbourCounters, d_omegas);
}

__global__ void computeViscosity(unsigned int* neighbors,
	unsigned int* numberOfNeighbors) {
	GET_INDEX_X_Y
		unsigned int numberOfParticles = params.numberOfParticles;
	if (index < numberOfParticles)
	{

		unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);		
		float4 vi;
		surf2Dread(&vi, velocities4, x, y);
		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		float4 vSum = make_float4(0.0, 0.0, 0.0, 0.0);
		float c = 0.001;

		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			float4 pj;
			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			float4 vj;
			surf2Dread(&vj, velocities4, neighborX, neighborY);
			float4 vij = vj - vi;
			vSum += vij* poly6(pi, pj);
		}

		float4 vNew = vi + c*vSum;
		surf2Dwrite(vNew, velocities4, x, y);
	}
}

void cudaComputeViscosity() {
	computeViscosity << <FOR_EACH_PARTICLE >> >(d_neighbours, d_neighbourCounters);
}
#endif