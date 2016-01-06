#ifndef ENGINE_H
#define ENGINE_H

#include <iostream>
#include <stdexcept>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class Engine {

public:
  Engine(GLFWwindow* window);

  void run();

protected:

private:
  GLFWwindow* window_;

};

#endif