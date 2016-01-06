#include "GL_Renderable.h"


GL_Renderable::GL_Renderable(const std::string& program_name) 
: boundBuffer_("")
{
  program_ = OpenGL_Loader::getInstance().accessProgram(program_name);
}


GL_Renderable::~GL_Renderable() {
  for(unsigned int i=0; i<bitmaps_.size(); i++) {
    delete bitmaps_[i];
  }
}


void GL_Renderable::setViewMatrix(const glm::mat4& view_matrix) {
  view_matrix_ = view_matrix;
}


void GL_Renderable::setProjectionMatrix(const glm::mat4& projection_matrix) {
  projection_matrix_ = projection_matrix;
}


void GL_Renderable::setCameraPosition(const glm::vec3& camera_position) {
  camera_position_ = camera_position;
}


void GL_Renderable::setViewDirection(const glm::vec3& view_direction) {
  view_direction_ = view_direction;
}


void GL_Renderable::setCameraNear(const float camera_near) {
  camera_near_ = camera_near;
}


void GL_Renderable::setCameraFar(const float camera_far) {
  camera_far_ = camera_far;
}


void GL_Renderable::setCurrentTime(const float current_time) {
  current_time_ = current_time;
}


void GL_Renderable::generateResources() {
  for(auto& pair : vaos_) {
    glGenVertexArrays(1, &pair.second->vao_);
    pair.second->generated_ = true;
  }

  for(auto& pair : buffers_) {
    glGenBuffers(1, &pair.second->buffer_);
    pair.second->generated_ = true;
  }

  for(unsigned int i=0; i<textures_.size(); i++) {
    glGenTextures(1, &textures_[i]->texture_);
    textures_[i]->generated_ = true;

    glBindTexture(textures_[i]->target_, textures_[i]->texture_);
    glTexImage2D(
                 textures_[i]->target_, 
                 textures_[i]->level_, 
                 textures_[i]->internalFormat_, 
                 textures_[i]->width_, 
                 textures_[i]->height_, 
                 textures_[i]->border_, 
                 textures_[i]->format_, 
                 textures_[i]->type_,
                 textures_[i]->data_
                );

    textures_[i]->parameters_();
    glGenerateMipmap(textures_[i]->target_);
    glBindTexture(textures_[i]->target_, 0);
  }
}


void GL_Renderable::add_buffer(const std::string& buffer_name) {
  if( buffers_.find(buffer_name) != buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does already contain the buffer '" << buffer_name << "'") };
  }

  std::shared_ptr<Buffer> buffer(new Buffer{buffer_name});
  buffers_[buffer_name] = buffer;
}


void GL_Renderable::add_shared_buffer(const std::string& buffer_name) {
  if( buffers_.find(buffer_name) != buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does already contain the buffer '" << buffer_name << "'") };
  }

  std::shared_ptr<Buffer> buffer(new Buffer{buffer_name});

  GL_Shared::getInstance().add_buffer(buffer);

  buffers_[buffer_name] = buffer;
}


void GL_Renderable::add_vao(const std::string& vao_name) {
  if( vaos_.find(vao_name) != vaos_.end() ) {
    throw std::invalid_argument{ report_error("The vaos map does already contain the vao '" << vao_name << "'") };
  }

  std::shared_ptr<Vao> vao(new Vao{vao_name});
  vaos_[vao_name] = vao;

}


void GL_Renderable::add_shared_vao(const std::string& vao_name) {
  if( vaos_.find(vao_name) != vaos_.end() ) {
    throw std::invalid_argument{ report_error("The vaos map does already contain the vao '" << vao_name << "'") };
  }

  std::shared_ptr<Vao> vao(new Vao{vao_name});

  GL_Shared::getInstance().add_vao(vao);

  vaos_[vao_name] = vao;

}


void GL_Renderable::add_texture(const std::string& texture_name, const std::string& file) {
  Bitmap* bitmap = new Bitmap{file};
  bitmaps_.push_back(bitmap);

  std::shared_ptr<Texture2D> texture2D(new Texture2D{texture_name,
                                                     GL_TEXTURE_2D,
                                                     0,
                                                     (GLint)bitmap->get_gl_channels(),
                                                     bitmap->get_width(),
                                                     bitmap->get_width(),
                                                     0,
                                                     bitmap->get_gl_channels(),
                                                     bitmap->get_gl_data_type(),
                                                     bitmap->get_data()});

  texture2D->setParameters([](){
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
  });

  add_uniform(texture_name);
  textures_.push_back(texture2D);
}


void GL_Renderable::add_shared_texture(const std::string& texture_name, const std::string& file) {
  Bitmap* bitmap = new Bitmap{file};
  bitmaps_.push_back(bitmap);

  std::shared_ptr<Texture2D> texture2D(new Texture2D{texture_name,
                                                     GL_TEXTURE_2D,
                                                     0,
                                                     (GLint)bitmap->get_gl_channels(),
                                                     bitmap->get_width(),
                                                     bitmap->get_width(),
                                                     0,
                                                     bitmap->get_gl_channels(),
                                                     bitmap->get_gl_data_type(),
                                                     bitmap->get_data()});

  texture2D->setParameters([](){
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); 
  });

  GL_Shared::getInstance().add_texture(texture2D);

  add_uniform(texture_name);
  textures_.push_back(texture2D);
}


void GL_Renderable::add_texture1D(std::string texture_name,
                   int width,
                   void* data,
                   GLint internalFormat,
                   GLenum target, 
                   GLenum format, 
                   GLenum type,
                   GLint level ,
                   GLint border) {
  std::shared_ptr<Texture1D> texture1D(new Texture1D{texture_name,
                                                     target,
                                                     level,
                                                     internalFormat,
                                                     width,
                                                     border,
                                                     format,
                                                     type,
                                                     data});

  add_uniform(texture_name);
  textures_.push_back(texture1D);
}


void GL_Renderable::add_shared_texture1D(std::string texture_name,
                          int width,
                          void* data,
                          GLint internalFormat,
                          GLenum target, 
                          GLenum format, 
                          GLenum type,
                          GLint level,
                          GLint border) {

  std::shared_ptr<Texture1D> texture1D(new Texture1D{texture_name,
                                                     target,
                                                     level,
                                                     internalFormat,
                                                     width,
                                                     border,
                                                     format,
                                                     type,
                                                     data});

  GL_Shared::getInstance().add_texture(texture1D);

  add_uniform(texture_name);
  textures_.push_back(texture1D);
}


void GL_Renderable::add_texture2D(std::string texture_name,
                   int width,
                   int height,
                   void* data,
                   GLint internalFormat,
                   GLenum target, 
                   GLenum format, 
                   GLenum type,
                   GLint level,
                   GLint border) {

  std::shared_ptr<Texture2D> texture2D(new Texture2D{texture_name,
                                                     target,
                                                     level,
                                                     internalFormat,
                                                     width,
                                                     height,
                                                     border,
                                                     format,
                                                     type,
                                                     data});
  add_uniform(texture_name);
  textures_.push_back(texture2D);
}


void GL_Renderable::add_shared_texture2D(std::string texture_name,
                          int width,
                          int height,
                          void* data,
                          GLint internalFormat,
                          GLenum target, 
                          GLenum format, 
                          GLenum type,
                          GLint level,
                          GLint border) {

  std::shared_ptr<Texture2D> texture2D(new Texture2D{texture_name,
                                                     target,
                                                     level,
                                                     internalFormat,
                                                     width,
                                                     height,
                                                     border,
                                                     format,
                                                     type,
                                                     data});

  GL_Shared::getInstance().add_texture(texture2D);

  add_uniform(texture_name);
  textures_.push_back(texture2D);
}


void GL_Renderable::add_texture3D(std::string texture_name,
                   int width,
                   int height,
                   int depth,
                   void* data,
                   GLint internalFormat,
                   GLenum target, 
                   GLenum format, 
                   GLenum type,
                   GLint level,
                   GLint border) {

  std::shared_ptr<Texture3D> texture3D(new Texture3D{texture_name,
                                                     target,
                                                     level,
                                                     internalFormat,
                                                     width,
                                                     height,
                                                     depth,
                                                     border,
                                                     format,
                                                     type,
                                                     data});

  add_uniform(texture_name);
  textures_.push_back(texture3D);
}


void GL_Renderable::add_shared_texture3D(std::string texture_name,
                          int width,
                          int height,
                          int depth,
                          void* data,
                          GLint internalFormat,
                          GLenum target, 
                          GLenum format, 
                          GLenum type,
                          GLint level,
                          GLint border) {
  std::shared_ptr<Texture3D> texture3D(new Texture3D{texture_name,
                                                     target,
                                                     level,
                                                     internalFormat,
                                                     width,
                                                     height,
                                                     depth,
                                                     border,
                                                     format,
                                                     type,
                                                     data});

  GL_Shared::getInstance().add_texture(texture3D);

  add_uniform(texture_name);
  textures_.push_back(texture3D);
}


void GL_Renderable::add_uniform(const std::string& uniform_name) {
  if( uniforms_.find(uniform_name) != uniforms_.end() ) {
    throw std::invalid_argument{ report_error("The uniform map does already contain the uniform '" << uniform_name << "'") };
  }
  uniforms_[uniform_name] = glGetUniformLocation(program_, uniform_name.c_str());
}


GLuint GL_Renderable::get_buffer(const std::string& buffer_name) {
  if( buffers_.find(buffer_name) == buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does not contain the buffer '" << buffer_name << "'") };
  }

  if( !buffers_[buffer_name]->generated_ ) {
    throw std::invalid_argument{ report_error("The buffer '" << buffer_name << "' has not been generated") };
  }

  return buffers_[buffer_name]->buffer_;
}


GLuint GL_Renderable::get_vao(const std::string& vao_name) {
  if( vaos_.find(vao_name) == vaos_.end() ) {
    throw std::invalid_argument{ report_error("The vaos map does not contain the vao '" << vao_name << "'") };
  }

  if( !vaos_[vao_name]->generated_ ) {
    throw std::invalid_argument{ report_error("The vao '" << vao_name << "' has not been generated") };
  }

  return vaos_[vao_name]->vao_;
}


GLuint GL_Renderable::get_uniform(const std::string& uniform_name) {
  auto ptr = uniforms_.find(uniform_name);

  if( ptr == uniforms_.end() ) {
    throw std::invalid_argument{ report_error("The uniform map does not contain the uniform '" << uniform_name << "'") };
  } 

  return ptr->second;
}


void GL_Renderable::bindVertexArray(const std::string& vao_name) {
  glBindVertexArray(get_vao(vao_name));
}


void GL_Renderable::unBindVertexArray() {
  glBindVertexArray(0);
}


void GL_Renderable::bindBuffer(const std::string& buffer_name, GLuint type) {
  if( buffers_.find(buffer_name) == buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does not contain the buffer '" << buffer_name << "'") };
  }

  glBindBuffer(type, buffers_[buffer_name]->buffer_);
  boundBuffer_ = buffer_name;
}


void GL_Renderable::unBindBuffer(GLuint type) {
  glBindBuffer(type, 0);
  boundBuffer_ = "";
}


void GL_Renderable::bufferData(GLenum target, GLsizeiptr size, const GLvoid* data, GLenum usage) {
  glBufferData(target, size, data, usage);

  if( boundBuffer_ == "" || buffers_.find(boundBuffer_) == buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does not contain the bound buffer '" << boundBuffer_ << "'") };
  }
  buffers_[boundBuffer_]->sizeiptr_ = size;
}


void GL_Renderable::vertexAttribPointer(GLuint index, GLint size, GLenum type, GLboolean normalized , GLsizei stride, const GLvoid* pointer) {
  glVertexAttribPointer(index, size, type, normalized, stride, pointer);

  if( boundBuffer_ == "" || buffers_.find(boundBuffer_) == buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does not contain the bound buffer '" << boundBuffer_ << "'") };
  }
  buffers_[boundBuffer_]->bufferDataSet_ = true;
  buffers_[boundBuffer_]->type_ = type;
}


void GL_Renderable::vertexAttribDivisor(GLuint index, GLuint divisor) {
  glVertexAttribDivisor(index, divisor);

  if( boundBuffer_ == "" || buffers_.find(boundBuffer_) == buffers_.end() ) {
    throw std::invalid_argument{ report_error("The buffers map does not contain the bound buffer '" << boundBuffer_ << "'") };
  }
  buffers_[boundBuffer_]->vertexAttribPointerSet_ = true;
  buffers_[boundBuffer_]->divisor_ = divisor;
}


void GL_Renderable::enableVertexAttribArray(GLuint index) {
  glEnableVertexAttribArray(index);
}


void GL_Renderable::disableVertexAttribArray(GLuint index) {
  glDisableVertexAttribArray(index);
}


void GL_Renderable::useProgram() {
  glUseProgram(program_);
}


void GL_Renderable::activateTextures() {
  for(unsigned int i=0; i<textures_.size(); i++) {
    glActiveTexture(GL_TEXTURE0 + i);
    glBindTexture(textures_[i]->target_, textures_[i]->texture_);
    glUniform1i(get_uniform(textures_[i]->texture_name_), i);
  }
}

