#ifndef COLLISION_H
#define COLLISION_H

#include "Kernels.h"
#include "Globals.h"

// --------------------------------------------------------------------------

__global__ void findContacts(unsigned int* neighbours,
                             unsigned int* contactCounters,
                             unsigned int* cellStarts,
                             unsigned int* cellEndings) {
  GET_INDEX_X_Y

  const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int kernelWidth = params.kernelWidth;
  const float particleDiameter = params.particleDiameter;

  neighbours += maxNeighboursPerParticle * index;

  if( index < numberOfParticles ) {
    unsigned int counter = 0;

    float4 predictedPosition1;
    surf2Dread(&predictedPosition1, predictedPositions4, x, y);

    float4 tempPos;
    unsigned int hash;
    unsigned int start;
    unsigned int end;
    unsigned int index2;
    
    if( counter == maxNeighboursPerParticle ) {
      done: 
      contactCounters[index] = counter;
      return;
    }
   
    #pragma unroll
    for(int i=-1; i<=1; i++) {
      #pragma unroll
      for(int j=-1; j<=1; j++) {
        #pragma unroll
        for(int k=-1; k<=1; k++) {

    //for(int j=-1; j<=1; j++) {
    //  for(int k=-1; k<=1; k++) {
    //    for(int i=-1; i<=1; i++) {

          tempPos = predictedPosition1;
          tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
          hash = mortonCode(tempPos);
      
          if( hash < maxGrid ) {
            start = cellStarts[hash]; 
            if( start < numberOfParticles ) {
              end = cellEndings[hash];
              for(index2=start; index2<end; index2++) {
                if( index != index2 && index2 < numberOfParticles ) {
                  neighbours[counter++] = index2;
                  if( counter == maxNeighboursPerParticle ) {
                    goto done;
                  }
                }
              }
            }
          }

        }
      }
    }

    contactCounters[index] = counter;
  }
  
}


void cudaCallFindContacts() {
  findContacts<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_contactCounters, d_cellStarts, d_cellEndings);
}

// --------------------------------------------------------------------------


__device__ __forceinline__ void addNeighbours(float4& tempPos,
                                              float4& predictedPosition1,
                                              int& i, int& j, int& k,
                                              const float& particleDiameter,
                                              unsigned int& hash,
                                              unsigned int& start,
                                              unsigned int& c,
                                              unsigned int* cellStarts,
                                              unsigned int& counter,
                                              unsigned int* contactCounters,
                                              unsigned int* neighbours,
                                              unsigned int* originalNeighbours,
                                              const unsigned int& maxNeighboursPerParticle,
                                              const unsigned int& maxGrid) {
  tempPos = predictedPosition1;
  tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
  hash = getHash(tempPos);
  if( hash < maxGrid ) {
    start = cellStarts[hash]; 
    if( start != UINT_MAX ) {
      c = contactCounters[start];
      for(int i=0; i<c && counter<maxNeighboursPerParticle; i++) {
        neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
      }
    }
  }
}

#define DONE_CHECK if( counter == maxNeighboursPerParticle ) goto done;


__global__ void findNeighbours(unsigned int* neighbours,
                               unsigned int* contactCounters,
                               unsigned int* neighbourCounters,
                               unsigned int* cellStarts,
                               unsigned int* cellEndings) {
  GET_INDEX_X_Y

  const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int kernelWidth = params.kernelWidth;
  const float particleDiameter = params.particleDiameter;

  unsigned int* originalNeighbours = neighbours;
  neighbours += maxNeighboursPerParticle * index;
  unsigned int counter = 0;

  if( index < numberOfParticles ) {
    float4 predictedPosition1;
    surf2Dread(&predictedPosition1, predictedPositions4, x, y);

    float4 tempPos;
    unsigned int hash;
    unsigned int start;
    unsigned int end;
    unsigned int index2;
    
    counter = contactCounters[index];

    if( counter == maxNeighboursPerParticle ) {
      done: 
      neighbourCounters[index] = counter;
      return;
    }


    int i;
    int j;
    int k;
    unsigned int c;

    /*
    i=-3; j=3; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=3; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=3; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=-3; j=0; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=-3; j=-3; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=-3; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=-3; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=0; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=0; k=3;
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK

    i=-3; j=3; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=3; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=3; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=-3; j=0; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=-3; j=-3; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=-3; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=-3; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=0; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=0; k=-3;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
  
    i=-3; j=3; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=3; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=3; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=-3; j=0; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=-3; j=-3; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=0; j=-3; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=-3; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
    i=3; j=0; k=0;    
    addNeighbours(tempPos, predictedPosition1, i, j, k, particleDiameter, hash, start, c, cellStarts, counter, contactCounters, neighbours, originalNeighbours, maxNeighboursPerParticle, maxGrid);
    DONE_CHECK
      */









      /*


    #pragma unroll
    for(int i=0; i<=3; i++) {
      #pragma unroll
      for(int j=0; j<=3; j++) {
        #pragma unroll
        for(int k=0; k<=3; k++) {
          if( !((i == 0) || (j == 0) || (k == 0)) ) continue;
         
          tempPos = predictedPosition1;
          tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
          hash = getHash(tempPos);
          if( hash < maxGrid ) {
            start = cellStarts[hash]; 
            if( start != UINT_MAX ) {
              end = cellEndings[hash];
              for(index2=start; index2<end; index2++) {
                if( index != index2 && index2 < numberOfParticles ) {
                  neighbours[counter++] = index2;
                  if( counter == maxNeighboursPerParticle ) {
                    goto done;
                  }
                }
              }
            }
          }

        }
      }
    }



    i=-2; j=2; k=2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=2; j=2; k=2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=-2; j=-2; k=2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=2; j=-2; k=2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=-2; j=2; k=-2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=2; j=2; k=-2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=-2; j=-2; k=-2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    i=2; j=-2; k=-2;
    tempPos = predictedPosition1;
    tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
    hash = getHash(tempPos);
    if( hash < maxGrid ) {
      start = cellStarts[hash]; 
      if( start != UINT_MAX ) {
        c = contactCounters[start];
        for(int i=0; i<c; i++) {
          neighbours[counter++] = originalNeighbours[start * maxNeighboursPerParticle + i];
          if( counter == maxNeighboursPerParticle ) {
            goto done;
          }
        }
      }
    }


    */

    /*
   -2, 2, 2
   2, 2, 2
   -2, -2, 2
   2, -2, 2

   -2, 2, -2
   2, 2, -2
   -2, -2, -2
   2, -2, -2
   */
    


    int abi;
    int abj;
    for(int shell=2; shell<=kernelWidth; shell++) {
      for(int i=-shell; i<=shell; i++) {
        abi = abs(i);
        for(int j=-shell; j<=shell; j++) {
          abj= abs(j);
          for(int k=-shell; k<=shell; k++) {
            if( (abs(i)!=shell) && (abs(j)!=shell) && (abs(k)!=shell) ) continue;

            tempPos = predictedPosition1;
            tempPos.x += i*particleDiameter; tempPos.y += j*particleDiameter; tempPos.z += k*particleDiameter;
            hash = getHash(tempPos);
            if( hash < maxGrid ) {
              start = cellStarts[hash]; 
              if( start != UINT_MAX ) {
                end = cellEndings[hash];
                for(index2=start; index2<end; index2++) {
                  if( index != index2 && index2 < numberOfParticles ) {
                    neighbours[counter++] = index2;
                    if( counter == maxNeighboursPerParticle ) {
                      goto done;
                    }
                  }
                }
              }
            }

          }
        }
      }
    }
    

    neighbourCounters[index] = counter;
  }
  
}


void cudaCallFindNeighbours() {
  findNeighbours<<<FOR_EACH_PARTICLE>>>(d_neighbours, d_contactCounters, d_neighbourCounters, d_cellStarts, d_cellEndings);
}

// --------------------------------------------------------------------------

__global__ void solveCollisions(unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* neighbours,
                                unsigned int* contactCounters,
                                unsigned int* neighbourCounters) {
  GET_INDEX_X_Y

  const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int kernelWidth = params.kernelWidth;
  const float particleDiameter = params.particleDiameter;
  const int randomStart = params.randomStart;

  if( index < numberOfParticles ) {
    neighbours += maxNeighboursPerParticle * index;

    float4 predictedPosition1;
    surf2Dread(&predictedPosition1, predictedPositions4, x, y);
    float mass1 = predictedPosition1.w;
    predictedPosition1.w = 0.0f;

    float4 position1;
    surf2Dread(&position1, positions4, x, y);

    float4 predictedPosition2;
    float3 pos1ToPos2;
    float3 addTo1;
    unsigned int index2;
    float halfOverlap;
    unsigned int x2;
    unsigned int y2;

    const unsigned int numberOfContacts = contactCounters[index];
    //const unsigned int numberOfContacts = neighbourCounters[index];
    /*
    int start1 = randomStart % (int)numberOfContacts;
    int end1 = numberOfContacts;

    int start2 = 0;
    int end2 = start1;
    */

    for(int i=0; i<numberOfContacts; i++) {
    //for(int i=numberOfContacts-1; i>0; i--) {
   // for(int i=end1-1; i>start1; i--) {
      index2 = neighbours[i];
      
      x2 = (index2 % textureWidth) * sizeof(float4);
      y2 = index2 / textureWidth;     
      surf2Dread(&predictedPosition2, predictedPositions4, x2, y2);
      float mass2 = predictedPosition2.w;
      predictedPosition2.w = 0.0f;

      float3 dir = make_float3(predictedPosition2 - predictedPosition1);
      float len = length(dir);

      halfOverlap = (particleDiameter - len) / 2.0f;

      if( halfOverlap > 0 ) {

        if( len <= 0.00001f ) {
          pos1ToPos2 = make_float3(0.0f, 0.0f, 0.0f);
        } else {
          pos1ToPos2 = dir / len;
        }

        //float inverseMass = ( 1.0f / (mass1 + mass2) );
        halfOverlap += 0.001f;
        addTo1 =  -1.0 * pos1ToPos2 * halfOverlap;
        //addTo2 =  1.0 * pos1ToPos2 * halfOverlap;

        predictedPosition1.x += addTo1.x;
        predictedPosition1.y += addTo1.y;
        predictedPosition1.z += addTo1.z;
        predictedPosition1.w = mass1;

        position1.x += addTo1.x;
        position1.y += addTo1.y;
        position1.z += addTo1.z;
        
        surf2Dwrite(predictedPosition1, predictedPositions4, x, y);

      }
    }
    /*
    for(int i=start2; i<end2; i++) {
      index2 = neighbours[i];
      
      x2 = (index2 % textureWidth) * sizeof(float4);
      y2 = index2 / textureWidth;     
      surf2Dread(&predictedPosition2, predictedPositions4, x2, y2);
      float mass2 = predictedPosition2.w;
      predictedPosition2.w = 0.0f;

      halfOverlap = (particleDiameter - length(predictedPosition2 - predictedPosition1)) / 2.0f;

      if( halfOverlap > 0 ) {
        pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1);
        //float mass = ( 1.0f / (mass1 + mass2) );
        halfOverlap += 0.001f;
        addTo1 =  -1.0 * pos1ToPos2 * halfOverlap;

        predictedPosition1 += addTo1;
        predictedPosition1.w = mass1;

        position1 += addTo1;
        
        surf2Dwrite(predictedPosition1, predictedPositions4, x, y);

      }
    }*/

    surf2Dwrite(position1, positions4, x, y);
  }

}


__global__ void solveCollisions2(unsigned int* cellStarts,
                                unsigned int* cellEndings,
                                unsigned int* neighbours,
                                unsigned int* contactCounters,
                                unsigned int* neighbourCounters,
                                float4* positions,
                                float4* predictedPositions,
                                float4* collisionDeltas,
                                float4* predictedPositionsCopy) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int textureWidth = params.textureWidth;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);


  const unsigned int maxNeighboursPerParticle = params.maxNeighboursPerParticle;
  const unsigned int index = idx / maxNeighboursPerParticle;
  const unsigned int x = (index % textureWidth) * sizeof(float4);
  const unsigned int y = index / textureWidth;

 
  const unsigned int maxGrid = params.maxGrid;
  const unsigned int maxContactConstraints = params.maxContactConstraints;
  const unsigned int kernelWidth = params.kernelWidth;
  const float particleDiameter = params.particleDiameter;
  const int randomStart = params.randomStart;

  if( index < numberOfParticles ) {

    //neighbours += maxNeighboursPerParticle * index;

    float4 predictedPosition1 = predictedPositions[index];
    float mass1 = predictedPosition1.w;
    predictedPosition1.w = 0.0f;

    float4 position1 = positions[index];
    float4 position2;

    float4 predictedPosition2;
    float3 pos1ToPos2;
    float3 addTo1;
    //float3 addTo2;
    unsigned int index2;
    float halfOverlap;
    unsigned int x2;
    unsigned int y2;

    const unsigned int numberOfContacts = contactCounters[index];

    if( idx < (index * maxNeighboursPerParticle + numberOfContacts) ) {

      index2 = neighbours[idx];
      
      x2 = (index2 % textureWidth) * sizeof(float4);
      y2 = index2 / textureWidth;     
      predictedPosition2 = predictedPositions[index2];
      float mass2 = predictedPosition2.w;
      predictedPosition2.w = 0.0f;

      float3 dir = make_float3(predictedPosition2 - predictedPosition1);
      float len = length(dir);

      halfOverlap = (particleDiameter - len) / 2.0f;

      if( halfOverlap > 0 ) {

        if( len <= 0.00001f ) {
          pos1ToPos2 = make_float3(0.0f, 0.0f, 0.0f);
        } else {
          pos1ToPos2 = dir / len;
        }

        //float inverseMass = ( 1.0f / (mass1 + mass2) );
        halfOverlap += 0.001f;
        addTo1 =  -1.0 * pos1ToPos2 * halfOverlap;
        //addTo2 =  1.0 * pos1ToPos2 * halfOverlap;

        
        atomicAdd(&(predictedPositions[index].x), addTo1.x);
        atomicAdd(&(positions[index].x), addTo1.x);

        atomicAdd(&(predictedPositions[index].y), addTo1.y);
        atomicAdd(&(positions[index].y), addTo1.y);

        atomicAdd(&(predictedPositions[index].z), addTo1.z);
        atomicAdd(&(positions[index].z), addTo1.z);



        /*
        atomicAdd(&(predictedPositions[index2].x), addTo2.x);
        atomicAdd(&(positions[index2].x), addTo2.x);

        atomicAdd(&(predictedPositions[index2].y), addTo2.y);
        atomicAdd(&(positions[index2].y), addTo2.y);

        atomicAdd(&(predictedPositions[index2].z), addTo2.z);
        atomicAdd(&(positions[index2].z), addTo2.z);
        */

        /*
        predictedPosition1 += addTo1;
        predictedPosition1.w = mass1;

        predictedPosition2 += addTo2;
        predictedPosition2.w = mass2;

        surf2Dread(&position2, positions4, x2, y2);
        position2 += addTo2;
        
        surf2Dwrite(position1, positions4, x, y);
        surf2Dwrite(predictedPosition1, predictedPositions4, x, y);

        surf2Dwrite(position2, positions4, x2, y2);
        surf2Dwrite(predictedPosition2, predictedPositions4, x2, y2);
        */

      }


    }

  }

}


__global__ void copyToBuffers(float4* positions,
                              float4* predictedPositions,
                              float4* collisionDeltas,
                              float4* predictedPositionsCopy) {
  GET_INDEX_X_Y
  const unsigned int numberOfParticles = params.numberOfParticles;

  if( index < numberOfParticles ) {
    float4 data;
    surf2Dread(&data, positions4, x, y);
    positions[index] = data;

    surf2Dread(&data, predictedPositions4, x, y);
    predictedPositions[index] = data;
    predictedPositionsCopy[index] = data;
  }
}

__global__ void copyToSurfaces(float4* positions,
                               float4* predictedPositions,
                               float4* collisionDeltas,
                               float4* predictedPositionsCopy) {
  GET_INDEX_X_Y

  const unsigned int numberOfParticles = params.numberOfParticles;

  if( index < numberOfParticles ) {
    float4 data;
    data = positions[index];
    surf2Dwrite(data, positions4, x, y);
    
    data = predictedPositions[index];
    surf2Dwrite(data, predictedPositions4, x, y);
  }
} 

void cudaCallSolveCollisions() {
  //solveCollisions<<<FOR_EACH_PARTICLE>>>(d_cellStarts, d_cellEndings, d_neighbours, d_contactCounters, d_neighbourCounters);
  copyToBuffers<<<FOR_EACH_PARTICLE>>>(deviceBuffers.d_positions, deviceBuffers.d_predictedPositions, deviceBuffers.d_collisionDeltas, deviceBuffers.d_predictedPositionsCopy);
  solveCollisions2<<<FOR_EACH_CONTACT>>>(d_cellStarts, d_cellEndings, d_neighbours, d_contactCounters, d_neighbourCounters, deviceBuffers.d_positions, deviceBuffers.d_predictedPositions, deviceBuffers.d_collisionDeltas, deviceBuffers.d_predictedPositionsCopy);
  copyToSurfaces<<<FOR_EACH_PARTICLE>>>(deviceBuffers.d_positions, deviceBuffers.d_predictedPositions, deviceBuffers.d_collisionDeltas, deviceBuffers.d_predictedPositionsCopy);
}

// --------------------------------------------------------------------------

__global__ void resetContacts(unsigned int* contacts) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contacts[idx] = UINT_MAX;
  }
}

void cudaCallResetContacts() {
  resetContacts<<<FOR_EACH_CONTACT>>>(d_neighbours);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintSuccess(int* contactConstraintSucces) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    contactConstraintSucces[idx] = -1;
  } 
}

void cudaCallResetContactConstraintSuccess() {
  resetContactConstraintSuccess<<<FOR_EACH_CONTACT>>>(d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void resetContactConstraintParticleUsed(int* contactConstraintParticleUsed) {
  const unsigned int numberOfParticles = params.numberOfParticles;
  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < numberOfParticles ) {
    contactConstraintParticleUsed[idx] = -1;
  }
}

void cudaCallResetContactConstraintParticleUsed() {
  resetContactConstraintParticleUsed<<<FOR_EACH_PARTICLE>>>(d_contactConstraintParticleUsed);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatches(unsigned int* contacts,
                                                int* particleUsed,
                                                int* constraintSucces) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;
  const unsigned int maxContactsPerParticle = params.maxNeighboursPerParticle;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    const int success = constraintSucces[idx];
    if( success < 0 ) { // If not success
      const unsigned int particle1 = idx / maxContactsPerParticle;
      const unsigned int particle2 = contacts[idx];
      if( particle2 != UINT_MAX ) {
        const unsigned int localId = threadIdx.x;
        const unsigned int localWorkSize = blockDim.x;
        for(unsigned int i=0; i<localWorkSize; i++) {
          if( (i == localId) && (particleUsed[particle1] < 0) && (particleUsed[particle2] < 0) ) {
            if( particleUsed[particle1] == -1 ) {
              particleUsed[particle1] = idx;
            }
            if( particleUsed[particle2] == -1 ) {
              particleUsed[particle2] = idx;
            }
          }
          __syncthreads();
        }
      }

    }
  }
}

void cudaCallSetupCollisionConstraintBatches() {
  setupCollisionConstraintBatches<<<FOR_EACH_CONTACT>>>(d_neighbours, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------

__global__ void setupCollisionConstraintBatchesCheck(unsigned int* contacts,
                                                     int* particleUsed,
                                                     int* constraintSucces) {
  const unsigned int maxContactConstraints = params.maxContactConstraints;
  const unsigned int maxContactsPerParticle = params.maxNeighboursPerParticle;
  const unsigned int textureWidth = params.textureWidth;
  const float particleDiameter = params.particleDiameter;

  const unsigned int idx = threadIdx.x + (((gridDim.x * blockIdx.y) + blockIdx.x) * blockDim.x);
  if( idx < maxContactConstraints ) {
    const unsigned int particle1 = idx / maxContactsPerParticle;
    const unsigned int particle2 = contacts[idx];
    if( particle2 != UINT_MAX ) {
       if( (particleUsed[particle1] == idx) && (particleUsed[particle2] == idx) ) {
        constraintSucces[idx] = 1;

        // Solve constraint for particle1 and particle2
        const unsigned int x1 = (particle1 % textureWidth) * sizeof(float4);
        const unsigned int y1 = particle1 / textureWidth;
        const unsigned int x2 = (particle2 % textureWidth) * sizeof(float4);
        const unsigned int y2 = particle2 / textureWidth;

        float4 predictedPosition1;
        surf2Dread(&predictedPosition1, predictedPositions4, x1, y1);
        predictedPosition1.w = 0.0f;  

        float4 predictedPosition2;
        surf2Dread(&predictedPosition2, predictedPositions4, x2, y2);
        predictedPosition2.w = 0.0f;

        const float distance = length(predictedPosition2 - predictedPosition1);
        const float overlap = particleDiameter - distance;

        if( overlap > 0 ) {
          const float4 pos1ToPos2 = normalize(predictedPosition2 - predictedPosition1); 
          const float halfOverlap = overlap / 2.0f;

          const float4 addTo1 = -1.0 * pos1ToPos2 * halfOverlap;
          const float4 addTo2 = pos1ToPos2 * halfOverlap;

          predictedPosition1 += addTo1;
          predictedPosition2 += addTo2;

          surf2Dwrite(predictedPosition1, predictedPositions4, x1, y1);
          surf2Dwrite(predictedPosition2, predictedPositions4, x2, y2);

          float4 position1;
          surf2Dread(&position1, positions4, x1, y1);
       
          float4 position2;
          surf2Dread(&position2, positions4, x2, y2);

          position1 += addTo1;
          position2 += addTo2;

          surf2Dwrite(position1, positions4, x1, y1);
          surf2Dwrite(position2, positions4, x2, y2);
        }
        
      } 
    }
  }
}

void cudaCallSetupCollisionConstraintBatchesCheck() {
  setupCollisionConstraintBatchesCheck<<<FOR_EACH_CONTACT>>>(d_neighbours, d_contactConstraintParticleUsed, d_contactConstraintSucces);
}

// --------------------------------------------------------------------------


void collisionHandling() {
  unsigned int stabilizationIterations = 1;

  cudaCallFindContacts();
  static int counter = 0;
  if( counter == 0 ) {
    cudaCallFindNeighbours();
    counter = 10;
  }

  for(unsigned int i=0; i<stabilizationIterations; i++) {
     //cudaCallResetContacts();
    
    cudaCallSolveCollisions();
    //cudaCallFindContacts();
    /*
    cudaCallResetContactConstraintSuccess();
    const unsigned int maxBatches = simulationParameters.maxNeighboursPerParticle;
    for(unsigned int b=0; b<maxBatches; b++) {
      cudaCallResetContactConstraintParticleUsed();
      cudaCallSetupCollisionConstraintBatches();
      cudaCallSetupCollisionConstraintBatchesCheck();
    }*/
  }

  counter--;
  
}

// --------------------------------------------------------------------------

void initializeCollision() {
  CUDA(cudaMalloc((void**)&d_neighbours, simulationParameters.maxContactConstraints * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactCounters, simulationParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_neighbourCounters, simulationParameters.maxParticles * sizeof(unsigned int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintSucces, simulationParameters.maxContactConstraints * sizeof(int)));
  CUDA(cudaMalloc((void**)&d_contactConstraintParticleUsed, simulationParameters.maxParticles * sizeof(int)));
}

// --------------------------------------------------------------------------



#endif // COLLISION_H