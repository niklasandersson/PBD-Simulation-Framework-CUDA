#ifndef CANVAS_H
#define CANVAS_H

#include "glfw/GLFW_Window.h"
#include "glfw/GLFW_Controls.h"

#include "opengl/OpenGL_Loader.h"
#include "opengl/GL_Renderable.h"

#include "parser/Config.h"

#include "console/Console.h"

#include "event/Events.h"

#include "Floor.h"
#include "Particles.h"


class Canvas : public GLFW_Window {

public:
  Canvas(const unsigned int window_width = 800, 
         const unsigned int window_height = 600, 
         const std::string& window_title = "Canvas",
         const unsigned int multisampling_and_antialiasing = 4,
         const unsigned int opengl_major_version = 3,
         const unsigned int opengl_minor_version = 3);

  ~Canvas();

  void initialize() override;
  void render() override;
  void cleanup() override;

protected:
  void set_glfw_window_hints() override;

private:
  void addConsoleCommands();
  void loadPrograms();

  const unsigned int multisampling_and_antialiasing_;
  const unsigned int opengl_major_version_;
  const unsigned int opengl_minor_version_;

  GLFW_Controls* glfw_controls_;

  Floor* floor_;
  //Particles* particles_;

};


#endif