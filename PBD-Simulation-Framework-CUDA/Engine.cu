#include "Engine.h"


Engine::Engine(GLFWwindow* window)
  : window_(window)
{

}

/*
__global__ void testKernel() {
  printf("thread id: %i", threadIdx.x);
}
*/


void Engine::run() {

  glfwSetInputMode(window_, GLFW_STICKY_KEYS, GL_TRUE);

  glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

  //dim3 block{2, 2, 2};

  //testKernel<<<1, 5>>>();
  //int size = 5;
  //testKernel << <1, size >> >();

  do {
    glClear(GL_COLOR_BUFFER_BIT);

    glfwSwapBuffers(window_);
    glfwPollEvents();

  } while( glfwGetKey(window_, GLFW_KEY_ESCAPE) != GLFW_PRESS 
           && glfwWindowShouldClose(window_) == 0 );

}