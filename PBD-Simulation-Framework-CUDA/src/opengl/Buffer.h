#ifndef BUFFER_H
#define BUFFER_H

#include <string>

#include <GL/glew.h>

struct Buffer {

  Buffer(std::string buffer_name)
  : buffer_name_(buffer_name)
  , bufferDataSet_(false)
  , vertexAttribPointerSet_(false)
  , vertexAttribDivisorSet_(false)
  , generated_(false)
  {

  }

  int getLength() {
    if( bufferDataSet_ && vertexAttribPointerSet_ ) {
      if( type_ == GL_FLOAT || type_ == GL_INT || type_ == GL_UNSIGNED_INT ) {
        return (int) sizeiptr_ / 4;
      } else if( type_ == GL_SHORT ) {
        return (int) sizeiptr_ / 2;
      } else if( type_ == GL_DOUBLE ) {
        return (int) sizeiptr_ / 8;
      } else if( GL_BYTE || GL_UNSIGNED_BYTE) {
        return (int) sizeiptr_;
      } 
    } 
    return 0;
  }

  const std::string buffer_name_;

  GLuint buffer_;
  bool generated_;

  // https://www.opengl.org/sdk/docs/man3/xhtml/glBufferData.xml
  bool bufferDataSet_;
  GLenum target_;
  GLsizeiptr sizeiptr_;
  const GLvoid * data_;
  GLenum usage_;

  // https://www.opengl.org/sdk/docs/man/html/glVertexAttribPointer.xhtml
  bool vertexAttribPointerSet_;
  GLuint index_;
  GLint size_;
  GLenum type_;
  GLboolean normalized_;
  GLsizei stride_;

  // https://www.opengl.org/sdk/docs/man3/xhtml/glVertexAttribDivisor.xml
  bool vertexAttribDivisorSet_;
  GLuint divisor_;

};


#endif // BUFFER_H