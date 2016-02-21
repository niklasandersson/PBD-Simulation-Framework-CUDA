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


// --------------------------------------------------------------------------


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


// --------------------------------------------------------------------------


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


// --------------------------------------------------------------------------


struct Communication {

  Communication() 
  : addParticle_(Delegate<void(glm::vec3 pos, glm::vec3 dir)>::from<Communication, &Communication::addParticleCallback>(this)), 
    addParticles_(Delegate<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)>::from<Communication, &Communication::addParticlesCallback>(this)),
    clearParticles_(Delegate<void()>::from<Communication, &Communication::clearParticlesCallback>(this)), 
    reload_(Delegate<void()>::from<Communication, &Communication::reloadCallback>(this)),
    load_(Delegate<void(const std::string file)>::from<Communication, &Communication::loadCallback>(this))
  {}

  void initialize() {
    Events::addParticle.subscribe(addParticle_);
    Events::addParticles.subscribe(addParticles_);
    Events::clearParticles.subscribe(clearParticles_);
    Events::reload.subscribe(reload_);
    Events::load.subscribe(load_);

    auto console = Console::getInstance();
     
    console->add("gravity", this, &Communication::setGravity);
    console->add("active", this, &Communication::setActive);
    console->add("kernelWidthNeighbours", this, &Communication::setKernelWidthNeighbours);
    console->add("cViscosity", this, &Communication::setCViscosity);
		console->add("eVorticity", this, &Communication::setEVorticity);
    console->add("restDensity", this, &Communication::setRestDensity);
    console->add("kernelWidthSpiky", this, &Communication::setKernelWidthSpiky);
    console->add("kernelWidthPoly", this, &Communication::setKernelWidthPoly);
    console->add("stabilizationIterations", this, &Communication::setStabilizationIterations);
    console->add("solverIterations", this, &Communication::setSolverIterations);
    console->add("deltaT", this, &Communication::setDeltaT);
    console->add("collisionType", this, &Communication::setCollisionType);
    console->add("maxCollisionBatches", this, &Communication::setMaxCollisionBatches);
    console->add("collisionStiffness", this, &Communication::setCollisionStiffness);
    console->add("kSCorr", this, &Communication::setKSCorr);
    console->add("nSCorr", this, &Communication::setNSCorr);
    console->add("qSCorr", this, &Communication::setQSCorr);
    console->add("enclosurePositionDamping", this, &Communication::setEnclosurePositionDamping);
    console->add("enclosureVelocityDamping", this, &Communication::setEnclosureVelocityDamping);
    console->add("enclosureX", this, &Communication::setEnclosureX);
    console->add("enclosureY", this, &Communication::setEnclosureY);
    console->add("enclosureZ", this, &Communication::setEnclosureZ);
    console->add("l", this, &Communication::loadCallback);

    Config& config = Config::getInstance();

    console->add("at", [&](const char* argv) {
      std::cout << "Currently at config file: " << config.at() << std::endl;
    });

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
          std::cout << "Current gravity: " << config.getValue<float>("Application.Simulation.Forces.gravity") << std::endl;
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
          std::cout << "Enter 'r' to reload the config file." << std::endl;  
        
        } else if( command == "w" ) {
          std::cout << "Enter 'w <optional file path>' to write the current configuration to disk." << std::endl;
          std::cout << std::endl;
          std::cout << "<optional file path> is of type string" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'w config2.txt'" << std::endl;
          std::cout << std::endl;
          std::cout << "Without an <optional file path> the config will be written to the one currently loaded." << std::endl;
          std::cout << std::endl;
          std::cout << "Be careful, this command is only recommended for advanced users." << std::endl;  
          std::cout << std::endl;

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
        
        } else if( command == "kernelWidthNeighbours" ) {
          std::cout << "Enter 'kernelWidthNeighbours <kernel width>' to set the kernel width which determines the amount of cells in all directions where possible neighbour particles can be found." << std::endl;
          std::cout << std::endl;
          std::cout << "<kernel width> is of type unsigned int, in the range of [1, 50]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'kernelWidthNeighbours 2'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current kernelWidthNeighbours: " << config.getValue<unsigned int>("Application.Simulation.Collision.kernelWidthNeighbours") << std::endl;
          std::cout << std::endl;

        } else if( command == "cViscosity" ) {
          std::cout << "Enter 'cViscosity <c value>' to set the c value for the viscosity computation." << std::endl;
          std::cout << std::endl;
          std::cout << "<c value> is of type float, in the range of [0.0, 100.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Tip: Try a c value of 0.0005 with only collision enabled." << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'cViscosity 0.0005'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current cViscosity: " << config.getValue<float>("Application.Simulation.Viscosity.cViscosity") << std::endl;
          std::cout << std::endl;

				} else if (command == "eVorticity") {
					std::cout << "Enter 'eVorticity <e value>' to set the e value for the vorticity computation." << std::endl;
					std::cout << std::endl;
					std::cout << "<e value> is of type float, in the range of [-10000.0, 10000.0]" << std::endl;
					std::cout << std::endl;
					std::cout << "Example: 'eVorticity 1.0'" << std::endl;
					std::cout << std::endl;
					std::cout << "Current eVorticity: " << config.getValue<float>("Application.Simulation.Vorticity.eVorticity") << std::endl;
					std::cout << std::endl;

        } else if( command == "restDensity" ) {
          std::cout << "Enter 'restDensity <density value>' to set the rest density value for the density computation." << std::endl;
          std::cout << std::endl;
          std::cout << "<density value> is of type float, in the range of ]0.0, 100000.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'restDensity 900'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current restDensity: " << config.getValue<float>("Application.Simulation.Density.restDensity") << std::endl;
          std::cout << std::endl;
        
        } else if( command == "kernelWidthSpiky" ) {
          std::cout << "Enter 'kernelWidthSpiky <kernel width>' to set the kernel width of the spiky kernel." << std::endl;
          std::cout << std::endl;
          std::cout << "<kernel width> is of type float, in the range of ]0.0, 1000.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'kernelWidthSpiky 1.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current kernelWidthSpiky: " << config.getValue<float>("Application.Simulation.Density.kernelWidthSpiky") << std::endl;
          std::cout << std::endl;

        } else if( command == "kernelWidthPoly" ) {
          std::cout << "Enter 'kernelWidthPoly <kernel width>' to set the kernel width of the poly kernel." << std::endl;
          std::cout << std::endl;
          std::cout << "<kernel width> is of type float, in the range of ]0.0, 1000.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'kernelWidthPoly 5.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current kernelWidthPoly: " << config.getValue<float>("Application.Simulation.Viscosity.kernelWidthPoly") << std::endl;
          std::cout << std::endl;

        } else if( command == "stabilizationIterations" ) {
          std::cout << "Enter 'stabilizationIterations <iterations>' to set the number of stabilization iterations." << std::endl;
          std::cout << std::endl;
          std::cout << "<iterations> is of type unsigned int, in the range of [0, 100]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'stabilizationIterations 2'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current stabilizationIterations: " << config.getValue<unsigned int>("Application.Simulation.Iterations.stabilizationIterations") << std::endl;
          std::cout << std::endl;

        } else if( command == "solverIterations" ) {
          std::cout << "Enter 'solverIterations <iterations>' to set the number of solver iterations." << std::endl;
          std::cout << std::endl;
          std::cout << "<iterations> is of type unsigned int, in the range of [0, 100]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'solverIterations 2'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current solverIterations: " << config.getValue<unsigned int>("Application.Simulation.Iterations.solverIterations") << std::endl;
          std::cout << std::endl;

        } else if( command == "deltaT" ) {
          std::cout << "Enter 'deltaT <value>' to set the value of delta time." << std::endl;
          std::cout << std::endl;
          std::cout << "<value> is of type float, in the range of ]0.0, 10.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'deltaT 0.001'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current deltaT: " << config.getValue<float>("Application.Simulation.Forces.deltaT") << std::endl;
          std::cout << std::endl;

        } else if( command == "collisionType" ) {
          std::cout << "Enter 'collisionType <type>' to set the type of collision handling used." << std::endl;
          std::cout << std::endl;
          std::cout << "<type> is of type unsigned int, in the range of {0, 1}" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'collisionType 1'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current collisionType: " << config.getValue<unsigned int>("Application.Simulation.Collision.collisionType") << std::endl;
          std::cout << std::endl;

        } else if( command == "maxCollisionBatches" ) {
          std::cout << "Enter 'maxCollisionBatches <number of batches>' to set the number of collision batches that are executed each stabilizationIteration when the collisionType is 1." << std::endl;
          std::cout << std::endl;
          std::cout << "<number of batches> is of type unsigned int, in the range of [1, 512]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'maxCollisionBatches 32'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current maxCollisionBatches: " << config.getValue<unsigned int>("Application.Simulation.Collision.maxCollisionBatches") << std::endl;
          std::cout << std::endl;

        } else if( command == "collisionStiffness" ) {
          std::cout << "Enter 'collisionStiffness <stiffness>' to set a parameter that controls the collision stiffness between two colliding particles." << std::endl;
          std::cout << std::endl;
          std::cout << "<stiffness> is of type float, in the range of [0.0, 1.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'collisionStiffness 0.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current collisionStiffness: " << config.getValue<float>("Application.Simulation.Collision.stiffness") << std::endl;
          std::cout << std::endl;

        } else if( command == "kSCorr" ) {
          std::cout << "Enter 'kSCorr <value>' to set value of k used for computation of SCorr." << std::endl;
          std::cout << std::endl;
          std::cout << "<value> is of type float, in the range of [-100.0, 100.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'kSCorr 0.5'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current kSCorr: " << config.getValue<float>("Application.Simulation.Density.kSCorr") << std::endl;
          std::cout << std::endl;

        } else if( command == "nSCorr" ) {
          std::cout << "Enter 'nSCorr <value>' to set value of n used for computation of SCorr." << std::endl;
          std::cout << std::endl;
          std::cout << "<value> is of type int, in the range of [-100, 100]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'nSCorr 6'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current nSCorr: " << config.getValue<int>("Application.Simulation.Density.nSCorr") << std::endl;
          std::cout << std::endl;

        } else if( command == "qSCorr" ) {
          std::cout << "Enter 'qSCorr <value>' to set value of q used for computation of SCorr." << std::endl;
          std::cout << std::endl;
          std::cout << "<value> is of type float, in the range of [-100.0, 100.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'qSCorr 0.4'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current qSCorr: " << config.getValue<float>("Application.Simulation.Density.qSCorr") << std::endl;
          std::cout << std::endl;

        } else if( command == "enclosurePositionDamping" ) {
          std::cout << "Enter 'enclosurePositionDamping <value>' to set the position damping of the enclosure." << std::endl;
          std::cout << std::endl;
          std::cout << "<value> is of type float, in the range of [0.0, 1.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'enclosurePositionDamping 0.2'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current enclosurePositionDamping: " << config.getValue<float>("Application.Simulation.Enclosure.enclosurePositionDamping") << std::endl;
          std::cout << std::endl;

        } else if( command == "enclosureVelocityDamping" ) {
          std::cout << "Enter 'enclosureVelocityDamping <value>' to set the velocity damping of the enclosure." << std::endl;
          std::cout << std::endl;
          std::cout << "<value> is of type float, in the range of [0.0, 1.0]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'enclosureVelocityDamping 1.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current enclosureVelocityDamping: " << config.getValue<float>("Application.Simulation.Enclosure.enclosureVelocityDamping") << std::endl;
          std::cout << std::endl;

        } else if( command == "enclosureX" ) {
          std::cout << "Enter 'enclosureX <min> <max>' to set the range of the enclosure in the x axis." << std::endl;
          std::cout << std::endl;
          std::cout << "<min> and <max> is of type float, each in the range of [1.5, 126.5]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'enclosureX 25.0 45.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current enclosureX min: " << config.getValue<float>("Application.Simulation.Enclosure.X.min") << std::endl;
          std::cout << "Current enclosureX max: " << config.getValue<float>("Application.Simulation.Enclosure.X.max") << std::endl;
          std::cout << std::endl;

        } else if( command == "enclosureY" ) {
          std::cout << "Enter 'enclosureY <min> <max>' to set the range of the enclosure in the y axis." << std::endl;
          std::cout << std::endl;
          std::cout << "<min> and <max> is of type float, each in the range of [1.5, 126.5]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'enclosureY 25.0 45.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current enclosureY min: " << config.getValue<float>("Application.Simulation.Enclosure.Y.min") << std::endl;
          std::cout << "Current enclosureY max: " << config.getValue<float>("Application.Simulation.Enclosure.Y.max") << std::endl;
          std::cout << std::endl;

        } else if( command == "enclosureZ" ) {
          std::cout << "Enter 'enclosureZ <min> <max>' to set the range of the enclosure in the z axis." << std::endl;
          std::cout << std::endl;
          std::cout << "<min> and <max> is of type float, each in the range of [1.5, 126.5]" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'enclosureZ 25.0 45.0'" << std::endl;
          std::cout << std::endl;
          std::cout << "Current enclosureZ min: " << config.getValue<float>("Application.Simulation.Enclosure.Z.min") << std::endl;
          std::cout << "Current enclosureZ max: " << config.getValue<float>("Application.Simulation.Enclosure.Z.max") << std::endl;
          std::cout << std::endl;

        } else if( command == "l" ) {
          std::cout << "Enter 'l <config file>' to load a different config file." << std::endl;
          std::cout << std::endl;
          std::cout << "<config file> is of type string" << std::endl;
          std::cout << std::endl;
          std::cout << "Example: 'l config2.txt'" << std::endl;
          std::cout << std::endl;

        } else if( command == "at" ) {
          std::cout << "Enter 'at' to see the currently loaded config file." << std::endl;
        
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
    } else {
      std::cout << "Value out of range [-1000.0, 1000.0]." << std::endl;
    }
  }

  void setActive(const std::string active, const bool value) {
    Config::getInstance().setValue(value, "Application", "Simulation", "Active", active);
  }

  void setKernelWidthNeighbours(const unsigned int kernelWidthNeighbours) {
    if( kernelWidthNeighbours >= 1 && kernelWidthNeighbours <= 50 ) { 
      Config::getInstance().setValue(kernelWidthNeighbours, "Application.Simulation.Collision.kernelWidthNeighbours");
    } else {
      std::cout << "Value out of range [1, 50]." << std::endl;
    }
  }

  void setCViscosity(const float cViscosity) {
    if( cViscosity >= 0.0f && cViscosity <= 100.0f ) {
      Config::getInstance().setValue(cViscosity, "Application.Simulation.Viscosity.cViscosity");
    } else {
      std::cout << "Value out of range [0.0, 100.0]." << std::endl;
    }
  }

	void setEVorticity(const float eVorticity) {
		if (eVorticity >= -10000.0f && eVorticity <= 10000.0f) {
			Config::getInstance().setValue(eVorticity, "Application.Simulation.Vorticity.eVorticity");
		}
		else {
			std::cout << "Value out of range [-10000.0, 10000.0]." << std::endl;
		}
	}

  void setRestDensity(const float restDensity) {
    if( restDensity > 0.0f && restDensity <= 100000.0f ) {
      Config::getInstance().setValue(restDensity, "Application.Simulation.Density.restDensity");
    } else {
      std::cout << "Value out of range ]0.0, 100000.0]." << std::endl;
    }
  }

  void setKernelWidthSpiky(const float kernelWidthSpiky) {
    if( kernelWidthSpiky > 0.0f && kernelWidthSpiky <= 1000.0f ) {
      Config::getInstance().setValue(kernelWidthSpiky, "Application.Simulation.Density.kernelWidthSpiky");
    } else {
      std::cout << "Value out of range ]0.0, 1000.0]." << std::endl;
    }
  }
  
  void setKernelWidthPoly(const float kernelWidthPoly) {
    if( kernelWidthPoly > 0.0f && kernelWidthPoly <= 1000.0f ) {
      Config::getInstance().setValue(kernelWidthPoly, "Application.Simulation.Viscosity.kernelWidthPoly");
    } else {
      std::cout << "Value out of range ]0.0, 1000.0]." << std::endl;
    }
  }
  
  void setSolverIterations(const unsigned int solverIterations) {
    if( solverIterations >= 0 && solverIterations <= 100 ) {
      Config::getInstance().setValue(solverIterations, "Application.Simulation.Iterations.solverIterations");
    } else {
      std::cout << "Value out of range [0, 100]." << std::endl;
    }
  }

  void setStabilizationIterations(const unsigned int stabilizationIterations) {
    if( stabilizationIterations >= 0 && stabilizationIterations <= 100 ) {
      Config::getInstance().setValue(stabilizationIterations, "Application.Simulation.Iterations.stabilizationIterations");
    } else {
      std::cout << "Value out of range [0, 100]." << std::endl;
    }
  }

  void setDeltaT(const float deltaT) {
    if( deltaT > 0.0f && deltaT <= 10.0f ) {
      Config::getInstance().setValue(deltaT, "Application.Simulation.Forces.deltaT");
    } else {
      std::cout << "Value out of range ]0.0, 10.0]." << std::endl;
    }
  }

  void setCollisionType(const unsigned int collisionType) {
    if( collisionType == 0 || collisionType == 1 ) {
      Config::getInstance().setValue(collisionType, "Application.Simulation.Collision.collisionType");
    } else {
      std::cout << "Value out of range {0, 1}." << std::endl;
    }
  }
  
  void setMaxCollisionBatches(const unsigned int maxCollisionBatches) {
    if( maxCollisionBatches >= 1 && maxCollisionBatches <= 512 ) {
      Config::getInstance().setValue(maxCollisionBatches, "Application.Simulation.Collision.maxCollisionBatches");
    } else {
      std::cout << "Value out of range [1, 512]." << std::endl;
    }
  }

  void setCollisionStiffness(const float collisionStiffness) {
    if( collisionStiffness >= 0.0f && collisionStiffness <= 1.0f ) {
      Config::getInstance().setValue(collisionStiffness, "Application.Simulation.Collision.stiffness");
    } else {
      std::cout << "Value out of range [0.0f, 1.0f]." << std::endl;
    }
  }

  void setKSCorr(const float kSCorr) {
    if( kSCorr >= -100.0f && kSCorr <= 100.0f ) {
      Config::getInstance().setValue(kSCorr, "Application.Simulation.Density.kSCorr");
    } else {
      std::cout << "Value out of range [-100.0f, 100.0f]." << std::endl;
    }
  }
  
  void setNSCorr(const int nSCorr) {
    if( nSCorr >= -100 && nSCorr <= 100 ) {
      Config::getInstance().setValue(nSCorr, "Application.Simulation.Density.nSCorr");
    } else {
      std::cout << "Value out of range [-100, 100]." << std::endl;
    }
  }

  void setQSCorr(const float qSCorr) {
    if( qSCorr >= -100.0f && qSCorr <= 100.0f ) {
      Config::getInstance().setValue(qSCorr, "Application.Simulation.Density.qSCorr");
    } else {
      std::cout << "Value out of range [-100.0f, 100.0f]." << std::endl;
    }
  }

  void setEnclosureX(const float min, const float max) {
    if( min < max && min >= 1.5f && max <= 126.5 ) {
      Config::getInstance().setValue(min, "Application.Simulation.Enclosure.X.min");
      Config::getInstance().setValue(max, "Application.Simulation.Enclosure.X.max");
    } else {
      std::cout << "Values out of range, each should be in [1.5, 126.5]." << std::endl;
    }
  }

  void setEnclosureY(const float min, const float max) {
    if( min < max && min >= 1.5f && max <= 126.5 ) {
      Config::getInstance().setValue(min, "Application.Simulation.Enclosure.Y.min");
      Config::getInstance().setValue(max, "Application.Simulation.Enclosure.Y.max");
    } else {
      std::cout << "Values out of range, each should be in [1.5, 126.5]." << std::endl;
    }
  }
  
  void setEnclosureZ(const float min, const float max) {
    if( min < max && min >= 1.5f && max <= 126.5 ) {
      Config::getInstance().setValue(min, "Application.Simulation.Enclosure.Z.min");
      Config::getInstance().setValue(max, "Application.Simulation.Enclosure.Z.max");
    } else {
      std::cout << "Values out of range, each should be in [1.5, 126.5]." << std::endl;
    }
  }

  void setEnclosurePositionDamping(const float enclosurePositionDamping) {
    if( enclosurePositionDamping >= 0.0f && enclosurePositionDamping <= 1.0f ) {
      Config::getInstance().setValue(enclosurePositionDamping, "Application.Simulation.Enclosure.enclosurePositionDamping");
    } else {
      std::cout << "Value out of range [0.0f, 1.0f]." << std::endl;
    }
  }

  void setEnclosureVelocityDamping(const float enclosureVelocityDamping) {
    if( enclosureVelocityDamping >= 0.0f && enclosureVelocityDamping <= 1.0f ) {
      Config::getInstance().setValue(enclosureVelocityDamping, "Application.Simulation.Enclosure.enclosureVelocityDamping");
    } else {
      std::cout << "Value out of range [0.0f, 1.0f]." << std::endl;
    }
  }

  void addParticleCallback(glm::vec3 pos, glm::vec3 dir) {
    auto glShared = GL_Shared::getInstance();
    auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
    auto maxParticles = glShared.get_unsigned_int_value("maxParticles");

    if( (*numberOfParticles + 1) > *maxParticles ) return;
    
    addParticle<<<1, 1>>>(pos, dir);
    glShared.set_unsigned_int_value("numberOfParticles", *numberOfParticles + 1);
  }

  void addParticlesCallback(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col) {
    auto glShared = GL_Shared::getInstance();
    auto numberOfParticles = glShared.get_unsigned_int_value("numberOfParticles");
    auto maxParticles = glShared.get_unsigned_int_value("maxParticles");

    if( (*numberOfParticles + numberOfParticlesToAdd) > *maxParticles ) return;
    
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
  
  void loadCallback(const std::string file) {
    Config& config = Config::getInstance();
    config.load(file);
  }
  
  Delegate<void(glm::vec3 pos, glm::vec3 dir)> addParticle_;
  Delegate<void(const unsigned int numberOfParticlesToAdd, std::vector<glm::vec4>& pos, std::vector<glm::vec4>& vel, std::vector<glm::vec4>& col)> addParticles_;
  Delegate<void()> clearParticles_;
  Delegate<void()> reload_;
  Delegate<void(const std::string file)> load_;

};


// --------------------------------------------------------------------------


#endif // COMMUNICATION_H