#include "Canvas.h"


Canvas::Canvas(const unsigned int window_width, 
               const unsigned int window_height, 
               const std::string& window_title,
               const unsigned int multisampling_and_antialiasing,
               const unsigned int opengl_major_version,
               const unsigned int opengl_minor_version) 
: GLFW_Window(window_width, window_height, window_title)
, multisampling_and_antialiasing_(multisampling_and_antialiasing)
, opengl_major_version_(opengl_major_version)
, opengl_minor_version_(opengl_minor_version)
{
  addConsoleCommands();
}


Canvas::~Canvas() {

}


void Canvas::addConsoleCommands() {
  //Console::getInstance()->add("fps", dynamic_cast<GLFW_Window*>(this), &GLFW_Window::set_print_fps);
  Console::getInstance()->add("fps", [&](const char* argv) {
    std::istringstream is(argv);
    bool printFps = false;
    is >> printFps;
    set_print_fps(printFps);
  });

  //Console::getInstance()->add("freeze", dynamic_cast<GLFW_Window*>(this), &GLFW_Window::toggleFreeze);
  Console::getInstance()->add("freeze", [&](const char* argv) {
    toggleFreeze();
  });

  Console::getInstance()->add("camera", [&](const char* argv) {
    if( glfw_controls_ ) {
      glm::vec3 positon = glfw_controls_->getCameraPosition();
      glm::vec3 direction = glfw_controls_->getCameraDirection();

      std::cout << "Position: {" << positon.x << ", " << positon.y << ", " << positon.z << "}" << std::endl; 
      std::cout << "Direction: {" << direction.x << ", " << direction.y << ", " << direction.z << "}" << std::endl; 
    }
  });
}


void Canvas::initialize() {
  GLFW_Window::initialize();

  loadPrograms();

  glfw_controls_ = new GLFW_Controls{
    glfw_window_,
    window_width_,
    window_height_,
    glm::vec3(0, 3, 12), 
    glm::vec3(0, 0.2, -1),
    45.0f,
    0.1f,
    200.0f,
    5.0f,
    0.002f
  };

  // glfw_controls_->setInputModeStandard();

 
  floor_ = new Floor();
  floor_->setCameraNear(glfw_controls_->getCameraNear());
  floor_->setCameraFar(glfw_controls_->getCameraFar());

  
  particles_ = new Particles();
  particles_->setCameraNear(glfw_controls_->getCameraNear());
  particles_->setCameraFar(glfw_controls_->getCameraFar());

  enclosure_ = new Enclosure();
  enclosure_->setCameraNear(glfw_controls_->getCameraNear());
  enclosure_->setCameraFar(glfw_controls_->getCameraFar());

}


void Canvas::set_glfw_window_hints() {
  glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
  glfwWindowHint(GLFW_SAMPLES, multisampling_and_antialiasing_);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, opengl_major_version_);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, opengl_minor_version_);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
}


void Canvas::loadPrograms() {
  Config& config = Config::getInstance();

  OpenGL_Loader& openglLoader = OpenGL_Loader::getInstance();

  openglLoader.loadPrograms(config.getValue<std::string>("Application.OpenGL.programDefinitions"));

  if( config.getValue<bool>("Application.OpenGL.displayAvailableShaders") ) {
    std::cout << "Available Shaders:" << std::endl;
    openglLoader.printAvailableShaders();
  }

  if( config.getValue<bool>("Application.OpenGL.displayAvailablePrograms") ) {
    std::cout << "Available Programs:" << std::endl;
    openglLoader.printAvailablePrograms();
  }
 
}


void Canvas::cleanup() {
  GLFW_Window::cleanup();
  delete glfw_controls_;
  delete floor_;
  delete particles_;
  delete enclosure_;
}


void Canvas::render() {
  GLFW_Window::render();  

  const float current_time = get_current_time();

  glfw_controls_->handleInput();
  // glfw_controls_->checkInGameToggle(current_time);

  glm::mat4 view_matrix = glfw_controls_->getViewMatrix();
  glm::mat4 projection_matrix = glfw_controls_->getProjectionMatrix();
  glm::vec3 camera_position = glfw_controls_->getCameraPosition();
  glm::vec3 view_direction = glfw_controls_->getCameraDirection();

  floor_->setViewMatrix(view_matrix);
  floor_->setProjectionMatrix(projection_matrix);
  floor_->setCameraPosition(camera_position);
  floor_->setViewDirection(view_direction);
  floor_->setCurrentTime(current_time);
  floor_->render();
 
  particles_->setViewMatrix(view_matrix);
  particles_->setProjectionMatrix(projection_matrix);
  particles_->setCameraPosition(camera_position);
  particles_->setViewDirection(view_direction);
  particles_->setCurrentTime(current_time);
  particles_->render();

  enclosure_->setViewMatrix(view_matrix);
  enclosure_->setProjectionMatrix(projection_matrix);
  enclosure_->setCameraPosition(camera_position);
  enclosure_->setViewDirection(view_direction);
  enclosure_->setCurrentTime(current_time);
  enclosure_->render();
  
  glfwSwapBuffers(glfw_window_);
  glfwPollEvents();

  glFinish();

}
