#ifndef GL_RENDERABLE_H
#define GL_RENDERABLE_H

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <memory>

#include <GL/glew.h>

#include <glm/glm.hpp>

#include "exception/Error.h"

#include "opengl/OpenGL_Loader.h"
#include "opengl/GL_Shared.h"

#include "format/Bitmap.h"

#include "TextureXD.h"
#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture3D.h"

#include "Buffer.h"

#include "Vao.h"


class GL_Renderable {

public:
  GL_Renderable(const std::string& program_name);

  virtual ~GL_Renderable();

  virtual void render() = 0;

  void setViewMatrix(const glm::mat4& view_matrix);
  void setProjectionMatrix(const glm::mat4& projection_matrix);
  void setCameraPosition(const glm::vec3& camera_position);
  void setViewDirection(const glm::vec3& view_direction);
  void setCameraNear(const float camera_near);
  void setCameraFar(const float camera_far);
  void setCurrentTime(const float current_time);

protected:
  GLuint program_;

  glm::vec3 camera_position_;
  glm::vec3 view_direction_;

  float camera_near_;
  float camera_far_;

  glm::mat4 view_matrix_;
  glm::mat4 projection_matrix_;

  float current_time_;

  void generateResources();

  void add_buffer(const std::string& buffer_name);
  void add_shared_buffer(const std::string& buffer_name);


  void add_vao(const std::string& vao_name);
  void add_shared_vao(const std::string& vao_name);


  void add_texture(const std::string& texture_name, const std::string& file);
  void add_shared_texture(const std::string& texture_name, const std::string& file);


  void add_texture1D(std::string texture_name,
                     int width,
                     void* data = nullptr,
                     GLint internalFormat = GL_RGBA32F,
                     GLenum target = GL_TEXTURE_1D, 
                     GLenum format = GL_RGBA, 
                     GLenum type = GL_FLOAT,
                     GLint level = 0,
                     GLint border = 0);

  void add_shared_texture1D(std::string texture_name,
                            int width,
                            void* data = nullptr,
                            GLint internalFormat = GL_RGBA32F,
                            GLenum target = GL_TEXTURE_1D, 
                            GLenum format = GL_RGBA, 
                            GLenum type = GL_FLOAT,
                            GLint level = 0,
                            GLint border = 0);

  void add_texture2D(std::string texture_name,
                     int width,
                     int height,
                     void* data = nullptr,
                     GLint internalFormat = GL_RGBA32F,
                     GLenum target = GL_TEXTURE_2D, 
                     GLenum format = GL_RGBA, 
                     GLenum type = GL_FLOAT,
                     GLint level = 0,
                     GLint border = 0);

  void add_shared_texture2D(std::string texture_name,
                            int width,
                            int height,
                            void* data = nullptr,
                            GLint internalFormat = GL_RGBA32F,
                            GLenum target = GL_TEXTURE_2D, 
                            GLenum format = GL_RGBA, 
                            GLenum type = GL_FLOAT,
                            GLint level = 0,
                            GLint border = 0);

  void add_texture3D(std::string texture_name,
                     int width,
                     int height,
                     int depth,
                     void* data = nullptr,
                     GLint internalFormat = GL_RGBA32F,
                     GLenum target = GL_TEXTURE_3D, 
                     GLenum format = GL_RGBA, 
                     GLenum type = GL_FLOAT,
                     GLint level = 0,
                     GLint border = 0);

  void add_shared_texture3D(std::string texture_name,
                            int width,
                            int height,
                            int depth,
                            void* data = nullptr,
                            GLint internalFormat = GL_RGBA32F,
                            GLenum target = GL_TEXTURE_3D, 
                            GLenum format = GL_RGBA, 
                            GLenum type = GL_FLOAT,
                            GLint level = 0,
                            GLint border = 0);

  void add_uniform(const std::string& uniform_name);

  GLuint get_buffer(const std::string& buffer_name);

  GLuint get_vao(const std::string& vao_name);

  GLuint get_uniform(const std::string& uniform_name);

  void bindVertexArray(const std::string& vao_name);

  void unBindVertexArray();

  void bindBuffer(const std::string& buffer_name, GLuint type = GL_ARRAY_BUFFER);

  void unBindBuffer(GLuint type = GL_ARRAY_BUFFER);

  void bufferData(GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage = GL_STATIC_DRAW); 

  void vertexAttribPointer(GLuint index, GLint size = 3, GLenum type = GL_FLOAT, GLboolean normalized = GL_FALSE, GLsizei stride = 0, const GLvoid* pointer = nullptr);

  void vertexAttribDivisor(GLuint index, GLuint divisor = 0);

  void enableVertexAttribArray(GLuint index);

  void disableVertexAttribArray(GLuint index);

  void useProgram();

  void activateTextures();


private:
  std::string boundBuffer_;

  std::vector<Bitmap*> bitmaps_;

  std::map<std::string, GLuint> uniforms_;

  std::vector<std::shared_ptr<TextureXD> > textures_;

  std::map<std::string, std::shared_ptr<Buffer> > buffers_;

  std::map<std::string, std::shared_ptr<Vao> > vaos_;

};


#endif // GL_RENDERABLE_H