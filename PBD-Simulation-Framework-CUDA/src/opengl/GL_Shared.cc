#include "GL_Shared.h"


void GL_Shared::clear() {
  textures_.clear();
  buffers_.clear();
  vaos_.clear();
  intValues_.clear();
  unsignedIntValues_.clear();
  floatValues_.clear();
}


bool GL_Shared::has_vao(const std::string& vao_name) const {
  if( vaos_.find(vao_name) == vaos_.end() ) {
    return false;
  }
  return true;
}


void GL_Shared::add_vao(std::shared_ptr<Vao> vao) {
  if( has_vao(vao->vao_name_) ) {
    throw std::invalid_argument{ report_error("There is already a vao with name '" << vao->vao_name_ << "' added to the shared vaos") };
  }
  vaos_[vao->vao_name_] = vao;
}


std::shared_ptr<Vao> GL_Shared::get_vao(std::string vao_name) {
  if( !has_buffer(vao_name) ) {
    throw std::invalid_argument{ report_error("The shared vao '" << vao_name << "' was not found") };
  }
  return vaos_[vao_name];
}


bool GL_Shared::has_buffer(const std::string& buffer_name) const {
  if( buffers_.find(buffer_name) == buffers_.end() ) {
    return false;
  }
  return true;
}


void GL_Shared::add_buffer(std::shared_ptr<Buffer> buffer) {
  if( has_texture(buffer->buffer_name_) ) {
    throw std::invalid_argument{ report_error("There is already a buffer with name '" << buffer->buffer_name_ << "' added to the shared buffers") };
  }
  buffers_[buffer->buffer_name_] = buffer;
}


std::shared_ptr<Buffer> GL_Shared::get_buffer(std::string buffer_name) {
  if( !has_buffer(buffer_name) ) {
    throw std::invalid_argument{ report_error("The shared buffer '" << buffer_name << "' was not found") };
  }
  return buffers_[buffer_name];
}


bool GL_Shared::has_texture(const std::string& texture_name) const {
  if( textures_.find(texture_name) == textures_.end() ) {
    return false;
  }
  return true;
}


void GL_Shared::add_texture(std::shared_ptr<TextureXD> texture) {
  if( has_texture(texture->texture_name_) ) {
    throw std::invalid_argument{ report_error("There is already a texture with name '" << texture->texture_name_ << "' added to the shared textures") };
  }
  textures_[texture->texture_name_] = texture;
}


std::shared_ptr<TextureXD> GL_Shared::get_texture(std::string texture_name) {
  if( !has_texture(texture_name) ) {
    throw std::invalid_argument{ report_error("The shared texture '" << texture_name << "' was not found") };
  }
  return textures_[texture_name];
}


bool GL_Shared::has_int_value(const std::string& int_name) const {
  if( intValues_.find(int_name) == intValues_.end() ) {
    return false;
  }
  return true;
}


void GL_Shared::add_int_value(const std::string int_name, std::shared_ptr<int> int_value) {
  if( has_int_value(int_name) ) {
    throw std::invalid_argument{ report_error("There is already a int value with name '" << int_name << "' added to the shared int values") };
  }
  intValues_[int_name] = int_value;
}


void GL_Shared::set_int_value(const std::string int_name, const int int_value) {
  if( !has_int_value(int_name) ) {
    throw std::invalid_argument{ report_error("The shared int value '" << int_name << "' was not found") };
  }
  *intValues_[int_name] = int_value;
}


// int GL_Shared::get_int_value(const std::string int_name) const {
//   if( !has_int_value(int_name) ) {
//     throw std::invalid_argument{ report_error("The shared int value '" << int_name << "' was not found") };
//   }
//   return *intValues_[int_name];
// }


std::shared_ptr<int> GL_Shared::get_int_value(const std::string int_name) {
  if( !has_int_value(int_name) ) {
    throw std::invalid_argument{ report_error("The shared int value '" << int_name << "' was not found") };
  }
  return intValues_[int_name];
}


bool GL_Shared::has_unsigned_int_value(const std::string& unsigned_int_name) const {
  if( unsignedIntValues_.find(unsigned_int_name) == unsignedIntValues_.end() ) {
    return false;
  }
  return true;
}


void GL_Shared::add_unsigned_int_value(const std::string unsigned_int_name, std::shared_ptr<unsigned int> unsigned_int_value) {
  if( has_unsigned_int_value(unsigned_int_name) ) {
    throw std::invalid_argument{ report_error("There is already a unsigned int value with name '" << unsigned_int_name << "' added to the shared unsigned int values") };
  }
  unsignedIntValues_[unsigned_int_name] = unsigned_int_value;
}


void GL_Shared::set_unsigned_int_value(const std::string unsigned_int_name, const unsigned int unsigned_int_value) {
  if( !has_unsigned_int_value(unsigned_int_name) ) {
    throw std::invalid_argument{ report_error("The shared unsigned int value '" << unsigned_int_name << "' was not found") };
  }
  *unsignedIntValues_[unsigned_int_name] = unsigned_int_value;
}


// unsigned int GL_Shared::get_unsigned_int_value(const std::string unsigned_int_name) const {
//   if( !has_unsigned_int_value(unsigned_int_name) ) {
//     throw std::invalid_argument{ report_error("The shared unsigned int value '" << unsigned_int_name << "' was not found") };
//   }
//   return *unsignedIntValues_[unsigned_int_name];
// }


std::shared_ptr<unsigned int> GL_Shared::get_unsigned_int_value(const std::string unsigned_int_name) {
  if( !has_unsigned_int_value(unsigned_int_name) ) {
    throw std::invalid_argument{ report_error("The shared unsigned int value '" << unsigned_int_name << "' was not found") };
  }
  return unsignedIntValues_[unsigned_int_name];
}


bool GL_Shared::has_float_value(const std::string& float_value) const {
  if( floatValues_.find(float_value) == floatValues_.end() ) {
    return false;
  }
  return true;
}


void GL_Shared::add_float_value(const std::string float_name, std::shared_ptr<float> float_value) {
  if( has_float_value(float_name) ) {
    throw std::invalid_argument{ report_error("There is already a float value with name '" << float_name << "' added to the shared float values") };
  }
  floatValues_[float_name] = float_value;
}


void GL_Shared::set_float_value(const std::string float_name, const float value) {
  if( !has_float_value(float_name) ) {
    throw std::invalid_argument{ report_error("The shared float value '" << float_name << "' was not found") };
  }
  *floatValues_[float_name] = value;
}


// float GL_Shared::get_float_value(const std::string float_name) const {
//   if( !has_float_value(float_name) ) {
//     throw std::invalid_argument{ report_error("The shared float value '" << float_name << "' was not found") };
//   }
//   return *floatValues_[float_name];
// }


std::shared_ptr<float> GL_Shared::get_float_value(const std::string float_name) {
  if( !has_float_value(float_name) ) {
    throw std::invalid_argument{ report_error("The shared float value '" << float_name << "' was not found") };
  }
  return floatValues_[float_name];
}