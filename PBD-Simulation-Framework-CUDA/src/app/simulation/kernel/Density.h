#ifndef DENSITY_H
#define DENSITY_H

#include "cuda/Cuda.h"
#include "cuda/Cuda_Helper_Math.h"

#include "Kernels.h"
#include "Globals.h"

void initializeDensity() {
	CUDA(cudaMalloc((void**)&d_lambdas, params.maxParticles * sizeof(float)));
	CUDA(cudaMalloc((void**)&d_deltaPositions, params.maxParticles * sizeof(float4)));
	/*CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_omegas, parameters->deviceParameters.maxParticles * sizeof(float3)));
	CUDA(cudaMalloc((void**)&parameters->deviceBuffers.d_externalForces, parameters->deviceParameters.maxParticles * sizeof(float4)));
	cudaMemset(parameters->deviceBuffers.d_externalForces, 0.0f, parameters->deviceParameters.maxParticles * sizeof(float4));*/
	simulationParameters.restDensity = 1000.0f;
}


__device__ float poly6(float4 pi, float4 pj)
{
	unsigned int kernelWidth = params.kernelWidth;
	pi.w = 0.0f;
	pj.w = 0.0f;

	float distance = length(pi - pj);
	
	if (distance < 0 || distance > kernelWidth)
	{
		float numeratorTerm = powf(kernelWidth * kernelWidth - distance * distance, 3);
		return (315.0f * numeratorTerm * numeratorTerm) / (64.0f * M_PI * powf(kernelWidth, 9));
	}
	else
		return 0.0f;
}

__device__ float4 spiky(float4 pi, float4 pj) {

	unsigned int kernelWidth = params.kernelWidth;

	pi.w = 0.0f;
	pj.w = 0.0f;
	float4 r = pi - pj;
	float distance = length(make_float3(r.x, r.y, r.z));

	float numeratorTerm = powf(kernelWidth - distance, 3);
	float denominatorTerm = M_PI * powf(kernelWidth, 6) * (distance + 0.0000001f);

	return 45.0f * numeratorTerm / (denominatorTerm * r + make_float4(0.000001f, 0.000001f, 0.000001f, 0.0f));
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


__global__ void computeLambda(const unsigned int numberOfParticles,	
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	float* lambdas
	) {
	GET_INDEX_X_Y
	if (index < numberOfParticles)
	{

		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float restDensity = params.restDensity;
		float ci = computeConstraintValue(pi, neighbors, numberOfNeighbors);

		float gradientValue = 0.0f;
		const float EPSILON = 0.00000001f;

		unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

		for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float4 pj;
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			float4 gradient = -1.0f * spiky(pi, pj) / restDensity;
			//printf("gradient.x = %f , gradient.y = %f , gradient.z = %f  \n", gradient.x, gradient.y, gradient.z);
			float gradientLength = length(make_float3(gradient.x, gradient.y, gradient.z));
			//printf("gradLength = %f \n", gradientLength);
			//printf("gradient.x = %f , gradient.y = %f , gradient.z = %f, gradLength = %f \n", gradient.x, gradient.y, gradient.z, gradientLength);
			gradientValue += gradientLength * gradientLength;
		}

		float4 gradientAtSelf = computeGradientAtSelf(pi, neighbors, numberOfNeighbors);

		float gradientAtSelfLength = length(make_float3(gradientAtSelf.x, gradientAtSelf.y, gradientAtSelf.z));
		gradientValue += gradientAtSelfLength * gradientAtSelfLength;

		//if (gradientValue == 0.0f)
		//printf("gradientValue = %f \n", gradientValue);
		//printf("ci = %f \n", ci);
		lambdas[index] = -1.0f * ci / (gradientValue + EPSILON);
		if (isnan(lambdas[index]))
			printf("lambdas[index] = %f , at = computeLambda()  \n", lambdas[index]);
	}
}

void cudaCallComputeLambda() {
	computeLambda << <FOR_EACH_CONTACT >> >(params.numberOfParticles, d_neighbours, d_neighbourCounters, d_lambdas);
}

__global__ void computeDeltaPositions(const unsigned int numberOfParticles,
	unsigned int* neighbors,
	unsigned int* numberOfNeighbors,
	float* lambdas,
	float4* deltaPositions
	) {
	GET_INDEX_X_Y
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
			float lambdaj = lambdas[neighborIndex];
			float absQ = 0.1f*kernelWidth;
			float4 deltaQ = make_float4(1.0f, 1.0f, 1.0f, 0.0f) * absQ + pi;
			//sCorr = -k * pow(poly6(pi, pj, kernelWidth), n) / poly6(deltaQ, make_float4(0.0f, 0.0f, 0.0f, 0.0f), kernelWidth);

			deltaPosition += (lambdai + lambdaj) * spiky(pi, pj);
		}

		deltaPositions[index] = deltaPosition / restDensity;
		if (isnan(deltaPositions[index].x) || isnan(deltaPositions[index].y) || isnan(deltaPositions[index].z))
			printf("deltaPositions[index] = %f , %f , %f ...... at = computeDeltaPositions()  \n", deltaPositions[index].x, deltaPositions[index].y, deltaPositions[index].z);
	}
}


void cudaCallComputeDeltaPositions() {
	computeDeltaPositions<< <FOR_EACH_CONTACT >> >(params.numberOfParticles, d_neighbours, d_neighbourCounters, d_lambdas, d_deltaPositions);
}

#endif