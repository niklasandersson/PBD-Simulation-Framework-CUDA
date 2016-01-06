#include "Window.h"


Window::Window(const std::string title, 
               const unsigned int width, 
               const unsigned int height,
               const unsigned int majorGLVersion,
               const unsigned int minorGLVersion)
  : title_(title)
  , width_(width)
  , height_(height)
  , majorGLVersion_(majorGLVersion)
  , minorGLVersion_(minorGLVersion)
{  
  initializeGLFW();
  initializeGLEW();
}


Window::~Window() {
  glfwTerminate();
}


void Window::initializeGLFW() {
  if( !glfwInit() ) {
    throw std::domain_error("Failed to initialize GLFW");
  } 
  setGLFWWindowHints();
  createGLFWWindow();
}


void Window::setGLFWWindowHints() {
  glfwWindowHint(GLFW_SAMPLES, 4);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, majorGLVersion_);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, minorGLVersion_);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}


void Window::createGLFWWindow() {
  window_ = glfwCreateWindow(width_, height_, title_.c_str(), nullptr, nullptr);
  if( window_ == nullptr ) {
    glfwTerminate();
    throw std::domain_error("Failed to create GLFW window");
  }
  glfwMakeContextCurrent(window_);
}


void Window::initializeGLEW() {
  if( glewInit() != GLEW_OK ) {
    glfwTerminate();
    throw std::domain_error("Failed to initialize GLEW");
  }
}