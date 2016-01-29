#ifndef COMMUNICATION_H
#define COMMUNICATION_H

#include <iostream>
#include <vector>

#include "glm/glm.hpp"

#include "event/Events.h"
#include "event/Delegate.h"

#include "parser/Config.h"

#include "console/Console.h"

#include "Globals.h"


__global__ void addParticle(glm::vec3 pos, glm::vec3 dir) {
  const unsigned int index = params.numberOfParticles;
  const unsigned int maxParticles = params.maxParticles;
  if( index < maxParticles ) {
    float4 position = make_float4(pos.x, pos.y, pos.z, 0.0f);
    float4 velocity = 50.0f * make_float4(dir.x, dir.y, dir.z, 0.0f);
    float4 color = make_float4(1.0f, 0.0f, 0.0f, 1.0f);

    const unsigned int textureWidth = params.textureWidth;
    const unsigned int x = (index % textureWidth) * sizeof(float4); 
    const unsigned int y = index / textureWidth;        

    surf2Dwrite(position, positions4, x, y);
    surf2Dwrite(velocity, velocities4, x, y);
    surf2Dwrite(color, colors4, x, y);
  }
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
    
    const unsigned int maxParticles = params.maxParticles;
    const unsigned int numberOfParticles = params.numberOfParticles;
    const unsigned int indexToUse = index + numberOfParticles; 
    if( indexToUse < maxParticles ) {
      const unsigned int xToUse = (indexToUse % textureWidth) * sizeof(float4); 
      const unsigned int yToUse = indexToUse / textureWidth;
 
      surf2Dwrite(position, positions4, xToUse, yToUse);
      surf2Dwrite(velocity, velocities4, xToUse, yToUse);
      surf2Dwrite(color, colors4, xToUse, yToUse);
    }
  }
}


struct Communication {

  Communication() 
  : addParticle_(Delegate<void(glm::vec3 pos, glm::vec3 dir)>::from<Communication, &Communication::addParticleCallback>(this)), 
    addParticles_(Delegate<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)>::from<Communication, &Communication::addParticlesCallback>(this)),
    clearParticles_(Delegate<void()>::from<Communication, &Communication::clearParticlesCallback>(this)), 
    reload_(Delegate<void()>::from<Communication, &Communication::reloadCallback>(this)),
    deferedConsoleExecution_(Delegate<void(const std::string command)>::from<Communication, &Communication::deferedConsoleExecutionCallback>(this)) 
  {}

  void initialize() {
    Events::addParticle.subscribe(addParticle_);
    Events::addParticles.subscribe(addParticles_);
    Events::clearParticles.subscribe(clearParticles_);
    Events::reload.subscribe(reload_);
    Events::deferedConsoleExecution.subscribe(deferedConsoleExecution_);

    auto console = Console::getInstance();
     
    console->add("gravity", this, &Communication::setGravity);
    console->add("active", this, &Communication::setActive);

    Config& config = Config::getInstance();

    console->add("help", [&](const char* argv) {
      std::istringstream is{argv};
      std::string command;
      if( !(is >> command) || command == "help" ) {
        std::cout << "Enter 'ls' to see the available commands." << std::endl;
        std::cout << "Enter 'help <command>' to see the specifics for that command." << std::endl;
      } else {
        if( command == "quit" || command == "q" ) {
          std::cout << "Enter 'quit' or 'q' to quit the program." << std::endl;
        
        } else if( command == "gravity" ) {
          std::cout << "Enter 'gravity <new gravity value>' to change the gravity." << std::endl;
          std::cout << std::endl;
          std::cout << "<new gravity value> is of type float, in the range of [-1000.0, 1000.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'gravity 100.0'" << std::endl;
          std::cout << std::endl;

        } else if( command == "active" ) {
          std::cout << "Enter 'active <to be enabled/disabled> <enable/disable>' to enable or disable a part of the algorithm." << std::endl;
          std::cout << std::endl;
          std::cout << "<to be enabled/disabled> is of type string, options are: ";
          std::vector<std::string> defines{config.getDefines("Application.Simulation.Active")};
          for(unsigned int i=0; i<defines.size(); i++) {
            std::cout << "'" << defines[i] << "'";
            if( i != defines.size()-1 ) std::cout << ", ";
          }
          std::cout << std::endl << std::endl;
          std::cout << "<enable/disable> is of type bool, options are: '0', '1'" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'active updatePositions 0'" << std::endl;
          std::cout << std::endl;

        } else if( command == "n" ) {
          std::cout << "Enter 'n' to see the number of particles currently displayed." << std::endl;  
        
        } else if( command == "r" ) {
          std::cout << "Enter 'r' to reload the config file. Be careful, this command is only recommended for advanced users." << std::endl;  
        
        } else if( command == "w" ) {
          std::cout << "Enter 'w' to write the current configuration to the config file. Be careful, this command is only recommended for advanced users." << std::endl;  
        
        } else if( command == "s" ) {
          std::cout << "Enter 's' to spawn a new box of particles." << std::endl;  
        
        } else if( command == "c" ) {
          std::cout << "Enter 'c' to clear all particles." << std::endl; 
          std::cout << "Tip: Enter 's' to spawn a new box of particles." << std::endl;  
        
        } else if( command == "fps" ) {
          std::cout << "Enter 'fps <enable/disable>' to turn on or off the display of the fps." << std::endl;  
          std::cout << std::endl;
          std::cout << "<enable/disable> is of type bool, options are: '0', '1'" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'fps 1'" << std::endl;
          std::cout << std::endl;
        
        } else if( command == "ls" ) {
          std::cout << "Enter 'ls' to see the available commands." << std::endl; 

        } else if( command == "getCamera" ) {
          std::cout << "Enter 'getCamera' to see the position and the direction of the camera." << std::endl; 
        
        } else if( command == "setCamera" ) {
          std::cout << "Enter 'setCamera' to save the position and the direction of the camera." << std::endl;
          std::cout << "Tip: Inorder to save the current camera attributes to disk the command 'w' also needs to be executed." << std::endl;
        
        } else {
          std::cout << "The command '" << command << "' is not recognized." << std::endl;
          std::cout << "Enter 'ls' to see the available commands." << std::endl;
        }
      }
    });
    
    console->execute("help");
  }

  void setGravity(const float gravity) {
    if( gravity >= -1000.0f && gravity <= 1000.0f ) {
      Config::getInstance().setValue(gravity, "Application.Simulation.Forces.gravity");
    }
  }

  void setActive(const std::string active, const bool value) {
    Config::getInstance().setValue(value, "Application", "Simulation", "Active", active);
  }

  void addParticleCallback(glm::vec3 pos, glm::vec3 dir) {
    auto glShared = GL_Shared::getInstance();
    auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
    addParticle<<<1, 1>>>(pos, dir);
    glShared.set_unsigned_int_value("numberOfParticles", *numberOfParticles + 1);
  }

  void addParticlesCallback(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col) {
    auto glShared = GL_Shared::getInstance();
    auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
    
    CUDA(cudaMemcpy(&deviceBuffers.d_positionsCopy[0], &pos[0][0], numberOfParticlesToAdd * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(&deviceBuffers.d_velocitiesCopy[0], &vel[0][0], numberOfParticlesToAdd * sizeof(float4), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(&deviceBuffers.d_colorsCopy[0], &col[0][0], numberOfParticlesToAdd * sizeof(float4), cudaMemcpyHostToDevice));

    Config& config = Config::getInstance();
    const unsigned int threadsPerBlock = config.getValue<unsigned int>("Application.Cuda.threadsPerBlock");
    dim3 blocks(std::ceil(numberOfParticlesToAdd / (float)threadsPerBlock), 1, 1);
    dim3 threads(threadsPerBlock, 1, 1);

    addParticles<<<blocks, threads>>>(numberOfParticlesToAdd, deviceBuffers.d_positionsCopy, deviceBuffers.d_velocitiesCopy, deviceBuffers.d_colorsCopy);
    glShared.set_unsigned_int_value("numberOfParticles", *numberOfParticles + numberOfParticlesToAdd);
  }
  
  void clearParticlesCallback() {
    auto glShared = GL_Shared::getInstance();
    glShared.set_unsigned_int_value("numberOfParticles", 0);
  }

  void reloadCallback() {
    Config& config = Config::getInstance();
    config.reload();
  }

  void deferedConsoleExecutionCallback(const std::string command) {  
    Console::getInstance()->execute(command.c_str());
  }

  Delegate<void(glm::vec3 pos, glm::vec3 dir)> addParticle_;
  Delegate<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)> addParticles_;
  Delegate<void()> clearParticles_;
  Delegate<void()> reload_;
  Delegate<void(const std::string command)> deferedConsoleExecution_;

};


#endif // COMMUNICATION_H