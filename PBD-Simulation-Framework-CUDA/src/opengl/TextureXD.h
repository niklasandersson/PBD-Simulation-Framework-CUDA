#ifndef TEXTUREXD_H
#define TEXTUREXD_H

#include <string>
#include <functional>

#include <GL/glew.h>

class TextureXD {

public:
  enum Dim{ ONE_DIM, TWO_DIM, THREE_DIM };
            
  TextureXD(Dim dim,
            std::string texture_name, 
            GLenum target,
            GLint level,
            GLint internalFormat,
            GLsizei width,
            GLint border,
            GLenum format,
            GLenum type,
            void* data);
  virtual ~TextureXD() = 0;

  void setParameters(std::function<void()> parameters) {
    parameters_ = parameters;
  }

  GLuint texture_;
  bool generated_;
  
  const std::string texture_name_;
  const Dim dim_;

  GLenum target_;
  GLint level_;
  GLint internalFormat_;
  GLsizei width_;
  GLsizei height_;
  GLsizei depth_;
  GLint border_;
  GLenum format_;
  GLenum type_;
  void* data_;

  std::function<void()> parameters_;

protected:

private:


};


#endif // TEXTUREXD_H