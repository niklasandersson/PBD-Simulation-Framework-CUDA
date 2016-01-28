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


__device__ float poly6Viscosity(float4 pi, float4 pj) {
	float kernelWidth = params.kernelWidthViscosity;
	pi.w = 0.0f;
	pj.w = 0.0f;
	float distance = length(make_float3(pi - pj));
	if (distance > 0.001f && distance < (kernelWidth - 0.001f) ) {
		float numeratorTerm = powf(kernelWidth * kernelWidth - distance * distance, 3);
		return (315.0f * numeratorTerm * numeratorTerm) / (0.001f + 64.0f * M_PI * powf(kernelWidth, 9));
	} else {
		return 0.0f;
  }
}


__device__ float4 spiky(float4 pi, float4 pj) {
	const float kernelWidth = (float) params.kernelWidthDensity;
	float4 r = pi - pj;
  r.w = 0.0f;
  const float distance = length(make_float3(r));
  if( distance > 0.0001f && ((kernelWidth - distance) > 0.0001f)) {
	  float numeratorTerm = pow(kernelWidth - distance, 2); 
	  float denominatorTerm = M_PI * pow(kernelWidth, 6);
	  return 15.0f * (numeratorTerm / denominatorTerm) * normalize(r);
  } else {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}


// ---------------------------------------------------------------------------------------


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
  
      accumulateA += poly6Viscosity(pi, pj);
      
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
      
      const float sCorr = -kSCorr * pow(poly6Viscosity(pi, pj) / poly6Viscosity(make_float4(qSCorr, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f)), nSCorr);
      
      accumulateDelta += (li + lj + sCorr) * spiky(pi, pj);
    }
    
    deltas[index] = accumulateDelta / restDensity;
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
			float4 pj;
			float neighborX = (neighborIndex % textureWidth) * sizeof(float4);
			float neighborY = (neighborIndex / textureWidth);

			surf2Dread(&pj, predictedPositions4, neighborX, neighborY);
			float3 omegaj = omegas[neighborIndex];
			float4 vj;
			surf2Dread(&vj, velocities4, neighborX, neighborY);
			float4 vij = vj - vi;
			float omegaLength = length(omegaj - omegai);
			float4 pij = pj - pi + EPSILON;

			gradient.x += omegaLength / pij.x;
			gradient.y += omegaLength / pij.y;
			gradient.z += omegaLength / pij.z;
		}

		float3 N = (1.0f / (length(gradient) + EPSILON)) * (gradient + EPSILON);
		float epsilon = 1.0f;
		float3 vorticity = epsilon * cross(N, omegas[index]);

		externalForces[index] = make_float4(vorticity.x, vorticity.y, vorticity.z, 0.0f);
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
			
      omega += cross(vij3, spike3);
		}

		omegas[index] = omega;
	}
}


void cudaCallComputeOmegas() {
	computeOmega<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_neighbourCounters, d_omegas);
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
		const float c = params.cViscosity; 

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


#endif // DENSITY_H