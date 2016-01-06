#ifndef VAO_H
#define VAO_H

#include <GL/glew.h>

struct Vao {

  Vao(std::string vao_name)
  : vao_name_(vao_name)
  {

  }

  const std::string vao_name_;

  GLuint vao_;
  bool generated_;

};

#endif // VAO_H