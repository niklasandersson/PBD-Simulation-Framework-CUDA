#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <iostream>
#include <vector>

#include "glm/glm.hpp"

#include "event/Events.h"
#include "event/Delegate.h"

#include "Globals.h"


__global__ void addParticle(glm::vec3 pos, glm::vec3 dir) {
  float4 position = make_float4(pos.x, pos.y, pos.z, 0.0f);
  float4 velocity = 50.0f * make_float4(dir.x, dir.y, dir.z, 0.0f);
  float4 color = make_float4(1.0f, 0.0f, 0.0f, 1.0f);

  //printf("ADD PARTICLE: %f, %f, %f\n", pos.x, pos.y, pos.z);

  const unsigned int textureWidth = params.textureWidth;
  const unsigned int index = params.numberOfParticles;
  const unsigned int x = (index % textureWidth) * sizeof(float4); 
  const unsigned int y = index / textureWidth;        

  surf2Dwrite(position, positions4, x, y);
  surf2Dwrite(velocity, velocities4, x, y);
  surf2Dwrite(color, colors4, x, y);
}

__global__ void addParticles(const unsigned int numberOfParticlesToAdd,
                             float4* positions,
                             float4* velocities,
                             float4* colors) {
  GET_INDEX_X_Y

  if( index < numberOfParticlesToAdd ) {
    float4 position = positions[index];
    float4 velocity = velocities[index];
    float4 color = colors[index];
    
    const unsigned int numberOfParticles = params.numberOfParticles;
    const unsigned int indexToUse = index + numberOfParticles; 
    const unsigned int xToUse = (indexToUse % textureWidth) * sizeof(float4); 
    const unsigned int yToUse = indexToUse / textureWidth;
 
    surf2Dwrite(position, positions4, xToUse, yToUse);
    surf2Dwrite(velocity, velocities4, xToUse, yToUse);
    surf2Dwrite(color, colors4, xToUse, yToUse);
  }
}

struct Communication {

  Communication() 
  : clicked_(Delegate<void(const double, const double, const int, const int, const int)>::from<Communication, &Communication::clickCallback>(this)),
    addParticle_(Delegate<void(glm::vec3 pos, glm::vec3 dir)>::from<Communication, &Communication::addParticleCallback>(this)), 
    addParticles_(Delegate<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)>::from<Communication, &Communication::addParticlesCallback>(this)),
    clearParticles_(Delegate<void()>::from<Communication, &Communication::clearParticlesCallback>(this)) 
  {
    
  }

  void initialize() {
    //Events::click.subscribe(clicked_);
    Events::addParticle.subscribe(addParticle_);
    Events::addParticles.subscribe(addParticles_);
    Events::clearParticles.subscribe(clearParticles_);
  }

  void clickCallback(const double position_x, const double position_y, const int button, const int action, const int mods) {
    if (button == 0 && action == 1) {
      //std::cout << "CLICK" << std::endl;
    }
  }

  void addParticleCallback(glm::vec3 pos, glm::vec3 dir) {
    //std::cout << "Add Particle: " << pos.x << ", " << pos.y << ", " << pos.z << " | " << dir.x << ", " << dir.y << ", " << dir.z << std::endl;

    auto glShared = GL_Shared::getInstance();
    auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");

    //glm::vec4 pos4(pos.x, pos.y, pos.z, 0.0f);
    //glm::vec4 dir4(dir.x, dir.y, dir.z, 0.0f);
    
    addParticle<<<1, 1>>>(pos, dir);
    glShared.set_unsigned_int_value("numberOfParticles", *numberOfParticles + 1);
  }

  void addParticlesCallback(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col) {
    auto glShared = GL_Shared::getInstance();
    auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
    
    CUDA(cudaMemcpy(&deviceBuffers.d_positionsCopy[0], &pos[0][0], numberOfParticlesToAdd * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(&deviceBuffers.d_velocitiesCopy[0], &vel[0][0], numberOfParticlesToAdd * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(&deviceBuffers.d_colorsCopy[0], &col[0][0], numberOfParticlesToAdd * sizeof(float4), cudaMemcpyHostToDevice));

    const unsigned int numberOfParticlesPerBlock = 128;
    dim3 blocks(std::ceil(numberOfParticlesToAdd / (float)numberOfParticlesPerBlock), 1, 1);
    dim3 threads(numberOfParticlesPerBlock, 1, 1);

    addParticles<<<blocks, threads>>>(numberOfParticlesToAdd, deviceBuffers.d_positionsCopy, deviceBuffers.d_velocitiesCopy, deviceBuffers.d_colorsCopy);
    
    glShared.set_unsigned_int_value("numberOfParticles", *numberOfParticles + numberOfParticlesToAdd);
  }
  
  void clearParticlesCallback() {
    auto glShared = GL_Shared::getInstance();
    glShared.set_unsigned_int_value("numberOfParticles", 0);
  }

  Delegate<void(const double, const double, const int, const int, const int)> clicked_;
  Delegate<void(glm::vec3 pos, glm::vec3 dir)> addParticle_;
  Delegate<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)> addParticles_;
  Delegate<void()> clearParticles_;

};

#endif // COMMUNICATION_H