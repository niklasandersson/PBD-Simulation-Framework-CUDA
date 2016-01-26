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
	CUDA(cudaMemset(d_deltaPositions, 0.0f, simulationParameters.maxParticles * sizeof(float4)));
	CUDA(cudaMemset(d_omegas, 0.0f, simulationParameters.maxParticles* sizeof(float3)));
	CUDA(cudaMemset(d_externalForces, 0.0f, simulationParameters.maxParticles* sizeof(float4)));
	CUDA(cudaMemset(d_lambdas, 0.0f, simulationParameters.maxParticles* sizeof(float)));
}

__global__ void clearAllTheCrap() {
  GET_INDEX_X_Y
  
  float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  surf2Dwrite(result, velocities4, x, y);
  surf2Dwrite(result, predictedPositions4, x, y);
  surf2Dwrite(result, positions4, x, y);
}

void callClearAllTheCrap() {
  clearAllTheCrap<<<FOR_EACH_PARTICLE>>>();
}


__device__ float poly6Viscosity(float4 pi, float4 pj)
{
	float kernelWidth = params.kernelWidthViscosity;
	pi.w = 0.0f;
	pj.w = 0.0f;

	float distance = length(make_float3(pi - pj));
	
	if (distance > 0.001f && distance < (kernelWidth - 0.001f) )
	{
		float numeratorTerm = powf(kernelWidth * kernelWidth - distance * distance, 3);
		return (315.0f * numeratorTerm * numeratorTerm) / (0.001f + 64.0f * M_PI * powf(kernelWidth, 9));
	}
	else
		return 0.0f;
}


__device__ float poly6New(float4 pi, float4 pj) {
	const float kernelWidth = (float) params.kernelWidthDensity;
	const float distance = length(make_float3(pi - pj));
  if( distance <= kernelWidth ) {
	  const float numeratorTerm = pow(kernelWidth * kernelWidth - distance * distance, 3);
    const float rtr = (315.0f * numeratorTerm) / (64.0f * M_PI * pow(kernelWidth, 9));
	  return rtr;
  } else {
    return 0.0f;
  }
}


__device__ float poly6ViscosityNew(float4 pi, float4 pj) {
	const float kernelWidth = (float) params.kernelWidthDensity;
	const float distance = length(make_float3(pi - pj));
  if( distance <= kernelWidth ) {
	  const float numeratorTerm = kernelWidth - distance;
    const float rtr = (45.0f * numeratorTerm) / (M_PI * pow(kernelWidth, 6));
	  return rtr;
  } else {
    return 0.0f;
  }
}

__device__ float4 spikyNew(float4 pi, float4 pj) {
	const float kernelWidth = (float) params.kernelWidthDensity;
	float4 r = pi - pj;
  r.w = 0.0f;
  const float distance = length(make_float3(r));
  if( distance <= kernelWidth) {
	  float numeratorTerm = pow(kernelWidth - distance, 2); 
	  float denominatorTerm = M_PI * pow(kernelWidth, 6);
	  return 45.0f * (numeratorTerm / denominatorTerm) * r;
  } else {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}


__device__ float poly6(float4 pi, float4 pj) {
  //return poly6New(pi, pj);
  return poly6Viscosity(pi, pj);
	const float kernelWidth = (float) params.kernelWidthDensity;
	const float distance = length(make_float3(pi - pj));
  if( distance > 0.0001f && ((kernelWidth - distance) > 0.0001f) ) {
	  const float numeratorTerm = pow(kernelWidth * kernelWidth - distance * distance, 3);
    const float rtr = (315.0f * numeratorTerm) / (64.0f * M_PI * pow(kernelWidth, 9));
    if( rtr <= 0 ) {
      printf("Error poly6 <= 0: %f\n", rtr);
    }
	  return rtr;
  } else {
    return 0.0f;
  }
}


__device__ float4 spiky(float4 pi, float4 pj) {
  //return spikyNew(pi, pj);
	const float kernelWidth = (float) params.kernelWidthDensity;
  /*float4 r = pi - pj;
  float rLength = length(make_float3(r));
	float hr_term = kernelWidth - rLength; 
	float gradient_magnitude = 45.0f / (M_PI * pow(kernelWidth, 6)) * hr_term * hr_term;
	float div = rLength + 0.001f;
	return gradient_magnitude * (1.0f / div) * r;
  */
	float4 r = pi - pj;
  r.w = 0.0f;
  const float distance = length(make_float3(r));
  if( distance > 0.0001f && ((kernelWidth - distance) > 0.0001f)) {
	  float numeratorTerm = pow(kernelWidth - distance, 2); // 3
	  float denominatorTerm = M_PI * pow(kernelWidth, 6);
	  return 15.0f * (numeratorTerm / denominatorTerm) * normalize(r); // should not be normalize here
  } else {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

// ---------------------------------------------------------------------------------------

__device__ float computeConstraintValue(float4 pi,
	                                      unsigned int* neighbors,
	                                      unsigned int* numberOfNeighbors) {
	GET_INDEX_X_Y

	const unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
	const float restDensity = params.restDensity;
	const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
	float density = 0.0f;

  unsigned int neighborIndex;
  float4 pj;
  unsigned int neighborX;
  unsigned int neighborY;
	for (unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
		neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		neighborX = (neighborIndex % textureWidth) * sizeof(float4);
		neighborY = (neighborIndex / textureWidth);
		
		surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
		density += poly6(pi, pj);
	}

	return (density / restDensity) - 1.0f;
}

__device__ float4 computeGradientAtSelf(float4 pi,
	                                      unsigned int* neighbors,
	                                      unsigned int* numberOfNeighbors) {
	GET_INDEX_X_Y

	const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
	float restDensity = params.restDensity;

	const unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
	float4 gradient = make_float4(0, 0, 0, 0);
  
  unsigned  int neighborIndex;
  float4 pj;
  unsigned int neighborX;
  unsigned int neighborY;

	for(unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
    neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
		neighborX = (neighborIndex % textureWidth) * sizeof(float4);
		neighborY = (neighborIndex / textureWidth);

		surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
		gradient += spiky(pi, pj);
	}

	return gradient / restDensity;
}

__global__ void computeLambda(unsigned int* neighbours,
	                            unsigned int* neighbourCounters,
                            	float* lambdas) {
  GET_INDEX_X_Y
  const unsigned int numberOfParticles = params.numberOfParticles;

  if( index < numberOfParticles ) {
    const unsigned int numberOfNeighbours = neighbourCounters[index];
    const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
    const float restDensity = params.restDensity;
    neighbours += index * maxNeighboursPerParticle; 

    float4 pi;
    surf2Dread(&pi, predictedPositions4, x, y);
    
    float accumulateA = 0.0f;
    float accumulateB = 0.0f;
    float4 accumulateC = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(unsigned int i=0; i<numberOfNeighbours; i++) {
      const unsigned int index2 = neighbours[i];
      const unsigned int x2 = (index2 % textureWidth) * sizeof(float4); 
      const unsigned int y2 = index2 / textureWidth;

      float4 pj;
      surf2Dread(&pj, predictedPositions4, x2, y2);
  
      accumulateA += poly6(pi, pj);
      
      float4 spik = spiky(pi, pj);

      accumulateB += dot(spik, spik);
      
      accumulateC += spik;
    }

    float A = (accumulateA / restDensity) - 1.0f;
    
    const float B = accumulateB / restDensity;

    const float C = dot(accumulateC, accumulateC) / restDensity;

    lambdas[index] = -A / (B + C + 0.0001f);
  }
  
}



__global__ void computeLambdaOld(unsigned int* neighbours,
	                            unsigned int* numberOfNeighbours,
                            	float* lambdas) {
	GET_INDEX_X_Y
  const unsigned int numberOfParticles = params.numberOfParticles;
	
  if( index < numberOfParticles ) {
		const float restDensity = params.restDensity;
		const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		const unsigned int currentNumberOfNeighbors = numberOfNeighbours[index];
		
    float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);

    float ciSum = 0.0f;
    float densitySum = 0.0f;
		float gradientValueSum = 0.0f;
    float4 gradientAtSelfSum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(unsigned int i=0; i<currentNumberOfNeighbors; i++) {
			const unsigned int neighbourIndex = neighbours[index * maxNumberOfNeighbors + i];
			const unsigned int neighbourX = (neighbourIndex % textureWidth) * sizeof(float4);
			const unsigned int neighbourY = (neighbourIndex / textureWidth);
      
      float4 pj;			
      surf2Dread(&pj, predictedPositions4, neighbourX, neighbourY);
			
      const float4 spik = spiky(pi, pj);
      const float4 gradient = -1.0f * spik / restDensity;
			const float gradientLength = length(make_float3(gradient));
			
      gradientValueSum += gradientLength * gradientLength;
		  densitySum += poly6(pi, pj);
      gradientAtSelfSum += spik;
    }

    gradientAtSelfSum /= restDensity;
    ciSum = (densitySum / restDensity) - 1.0f;

		const float gradientAtSelfLength = length(make_float3(gradientAtSelfSum));
		gradientValueSum = gradientAtSelfLength * gradientAtSelfLength;

		lambdas[index] = -1.0f * ciSum / (gradientValueSum + 0.0001f);
	}

}

void cudaCallComputeLambda() {
	computeLambda<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_neighbourCounters, d_lambdas);
}
__global__ void computeDeltaPositions(unsigned int* neighbours,
	                                    unsigned int* neighbourCounters,
	                                    float* lambdas,
	                                    float4* deltas) {
  GET_INDEX_X_Y
  const unsigned int numberOfParticles = params.numberOfParticles;

  if( index < numberOfParticles ) {
    const unsigned int numberOfNeighbours = neighbourCounters[index];
    const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
    const float restDensity = params.restDensity;
    neighbours += index * maxNeighboursPerParticle; 

    const float kSCorr = params.kSCorr;
    const int nSCorr = params.nSCorr;
    const float qSCorr = params.qSCorr;

    float4 pi;
    surf2Dread(&pi, predictedPositions4, x, y);
    
    const float li = lambdas[index];
    
    float4 accumulateDelta = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for(unsigned int i=0; i<numberOfNeighbours; i++) {
      const unsigned int index2 = neighbours[i];
      const unsigned int x2 = (index2 % textureWidth) * sizeof(float4); 
      const unsigned int y2 = index2 / textureWidth;
      
      float4 pj;
      surf2Dread(&pj, predictedPositions4, x2, y2);
    
      const float lj = lambdas[index2];
      
      const float sCorr = -kSCorr * pow(poly6(pi, pj) / poly6(make_float4(qSCorr, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f)), nSCorr);
      
      accumulateDelta += (li + lj + sCorr) * spiky(pi, pj);
    }
    
    deltas[index] = accumulateDelta / restDensity;
  }
}
  
__global__ void computeDeltaPositionsOld(unsigned int* neighbors,
	                                    unsigned int* numberOfNeighbors,
	                                    float* lambdas,
	                                    float4* deltaPositions) {
  GET_INDEX_X_Y
	const unsigned int numberOfParticles = params.numberOfParticles;
	
  if( index < numberOfParticles ) {
		const float restDensity = params.restDensity;
		const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		const float kernelWidth = params.kernelWidthDensity;

		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		const unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		const float lambdai = lambdas[index];
		float4 deltaPosition = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    const float absQ = 0.1f * kernelWidth;
		const float sCorr = 0.0f;
		const float k = 1.0f;
    const float n = 1.0f;
    const float4 deltaQ = make_float4(1.0f, 1.0f, 1.0f, 0.0f) * absQ + pi;

    unsigned int neighborIndex;
    float4 pj;
    unsigned int neighborX;
    unsigned int neighborY;
    float lambdaj;
		
    for(unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
			neighborIndex = neighbors[index * maxNumberOfNeighbors + i];
			neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			neighborY = (neighborIndex / textureWidth);
			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
		  lambdaj = lambdas[neighborIndex];
      if (isnan(pj.x) || isnan(pj.y) || isnan(pj.z))
				printf("IN computeDeltaPositions: pj = %f , %f , %f ...... computeDeltaPositions()  \n", pj.x, pj.y, pj.z);
			
      //sCorr = -k * pow(poly6(pi, pj, kernelWidth), n) / poly6(deltaQ, make_float4(0.0f, 0.0f, 0.0f, 0.0f), kernelWidth);
			deltaPosition += (lambdai + lambdaj) * spiky(pi, pj);
		}

		deltaPositions[index] = deltaPosition / restDensity;
	}
}


void cudaCallComputeDeltaPositions() {
	computeDeltaPositions<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_neighbourCounters, d_lambdas, d_deltaPositions);
}

__global__ void applyDeltaPositions(float4* deltas) {
	GET_INDEX_X_Y
	const unsigned int numberOfParticles = params.numberOfParticles;
	
	if( index < numberOfParticles ) {
    float4 predictedPosition;
	  surf2Dread(&predictedPosition, predictedPositions4, x, y);

		const float4 newPredictedPosition = predictedPosition - deltas[index];

		surf2Dwrite(newPredictedPosition, predictedPositions4, x, y);
	}
}

__global__ void applyDeltaPositionsOld(float4* d_deltaPositions) {
	GET_INDEX_X_Y
	const unsigned int numberOfParticles = params.numberOfParticles;
	
  float4 predictedPositions;
	surf2Dread(&predictedPositions, predictedPositions4, x, y);
	
	if( index < numberOfParticles ) {
		if (isnan(d_deltaPositions[index].x) || isnan(d_deltaPositions[index].y) || isnan(d_deltaPositions[index].z))
			printf("IN APPLYDELTAPOS: d_deltaPositions.x = %f, d_deltaPositions.y = %f, d_deltaPositions.z = %f \n", d_deltaPositions[index].x, d_deltaPositions[index].y, d_deltaPositions[index].z);

		if (isnan(predictedPositions.x) || isnan(predictedPositions.y) || isnan(predictedPositions.z))
			printf("IN APPLYDELTAPOS: predictedPositions.x = %f, predictedPositions.y = %f, predictedPositions.z = %f \n", predictedPositions.x, predictedPositions.y, predictedPositions.z);

		const float4 result = predictedPositions + d_deltaPositions[index];
		//printf("result.x = %f, result.y = %f, result.z = %f \n", result.x, result.y, result.z);

		surf2Dwrite(result, predictedPositions4, x, y);
	}
}

void cudaCallApplyDeltaPositions() {
	applyDeltaPositions<<<FOR_EACH_PARTICLE>>>(d_deltaPositions);
}

__global__ void computeVorticity(unsigned int* neighbors,
	                               unsigned int* numberOfNeighbors,
	                               float3* omegas,
	                               float4* externalForces) {
	GET_INDEX_X_Y
  const unsigned int numberOfParticles = params.numberOfParticles;
	if( index < numberOfParticles ) {
		const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		float4 vi;
		surf2Dread(&vi, velocities4, x, y);
		float3 omegai = omegas[index];
		const unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];

		float3 gradient = make_float3(0.0f, 0.0f, 0.0f);
		const float EPSILON = 0.00000001f;
		for(unsigned int i = 0; i < currentNumberOfNeighbors; i++) {
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
	computeVorticity<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_neighbourCounters, d_omegas, d_externalForces);
}


__global__ void computeOmega(unsigned int* neighbors,
	                           unsigned int* numberOfNeighbors,
	                           float3* omegas) {
	GET_INDEX_X_Y
	const unsigned int numberOfParticles = params.numberOfParticles;
	
  if( index < numberOfParticles ) {
		const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);
		float4 vi;
		surf2Dread(&vi, velocities4, x, y);
		const unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		float3 omega = make_float3(0.0f, 0.0f, 0.0f);

		for(unsigned int i=0; i<currentNumberOfNeighbors; i++) {
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
	const unsigned int numberOfParticles = params.numberOfParticles;
	if( index < numberOfParticles ) {
		const unsigned int maxNumberOfNeighbors = params.maxNeighboursPerParticle;
		float4 pi;
		surf2Dread(&pi, predictedPositions4, x, y);		
		float4 vi;
		surf2Dread(&vi, velocities4, x, y);
		const unsigned int currentNumberOfNeighbors = numberOfNeighbors[index];
		float4 vSum = make_float4(0.0, 0.0, 0.0, 0.0);
		const float c = params.cViscosity; // 0.0005f

		for(unsigned int i=0; i<currentNumberOfNeighbors; i++) {
			unsigned int neighborIndex = neighbors[i + index * maxNumberOfNeighbors];
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			float4 pj;
			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			float4 vj;
			surf2Dread(&vj, velocities4, neighborX, neighborY);
			float4 vij = vj - vi;
			vSum += vij * poly6Viscosity(pi, pj);
		}

		float4 vNew = vi + c*vSum;
		surf2Dwrite(vNew, velocities4, x, y);
	}
}

void cudaComputeViscosity() {
	computeViscosity<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_neighbourCounters);
}

#endif