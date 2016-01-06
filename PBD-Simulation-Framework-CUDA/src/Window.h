#ifndef WINDOW_H
#define WINDOW_H

#include <iostream>
#include <stdexcept>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

class Window {

public:
  Window(const std::string title, 
         const unsigned int width = 1024, 
         const unsigned int height = 768,
         const unsigned int majorGLVersion = 3,
         const unsigned int minorGLVersion = 3);
  ~Window();

  GLFWwindow* getGLFWWindow() const { return window_; }

protected:
  void initializeGLFW();
  void setGLFWWindowHints();
  void createGLFWWindow();

  void initializeGLEW();

private:
  std::string title_;

  unsigned int width_;
  unsigned int height_;

  unsigned int majorGLVersion_;
  unsigned int minorGLVersion_;

  GLFWwindow* window_;

};

#endif