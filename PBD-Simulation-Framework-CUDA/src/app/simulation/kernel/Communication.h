#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <iostream>

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

struct Communication {

  Communication() 
  : clicked_(Delegate<void(const double, const double, const int, const int, const int)>::from<Communication, &Communication::clickCallback>(this)),
    addParticle_(Delegate<void(glm::vec3 pos, glm::vec3 dir)>::from<Communication, &Communication::addParticleCallback>(this)) 
  {
    
  }

  void initialize() {
    //Events::click.subscribe(clicked_);
    Events::addParticle.subscribe(addParticle_);
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

  Delegate<void(const double, const double, const int, const int, const int)> clicked_;
  Delegate<void(glm::vec3 pos, glm::vec3 dir)> addParticle_;
};

#endif // COMMUNICATION_H