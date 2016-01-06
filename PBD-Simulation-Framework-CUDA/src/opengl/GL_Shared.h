#ifndef GL_CL_SHARED_H
#define GL_CL_SHARED_H

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <stdexcept>
#include <memory>

#include "exception/Error.h"

#include "TextureXD.h"
#include "Texture1D.h"
#include "Texture2D.h"
#include "Texture3D.h"

#include "Buffer.h"

#include "Vao.h"


class GL_Shared {

public:
  static GL_Shared& getInstance() {
    static GL_Shared gl_shared{};
    return gl_shared;
  };

  void clear();

  bool has_vao(const std::string& vao_name) const;
  void add_vao(std::shared_ptr<Vao> vao);
  std::shared_ptr<Vao> get_vao(std::string vao_name);

  bool has_buffer(const std::string& buffer_name) const;
  void add_buffer(std::shared_ptr<Buffer> buffer);
  std::shared_ptr<Buffer> get_buffer(std::string buffer_name);

  bool has_texture(const std::string& texture_name) const;
  void add_texture(std::shared_ptr<TextureXD> texture);
  std::shared_ptr<TextureXD> get_texture(std::string texture_name);

  bool has_int_value(const std::string& int_name) const;
  void add_int_value(const std::string int_name, std::shared_ptr<int> int_value);
  void set_int_value(const std::string int_name, const int int_value);
  // int get_int_value(std::string int_name) const;
  std::shared_ptr<int> get_int_value(std::string int_name);

  bool has_unsigned_int_value(const std::string& unsigned_int_name) const;
  void add_unsigned_int_value(const std::string unsigned_int_name, std::shared_ptr<unsigned int> int_value);
  void set_unsigned_int_value(const std::string unsigned_int_name, const unsigned unsigned_int_value);
  // unsigned int get_unsigned_int_value(std::string unsigned_int_name) const;
  std::shared_ptr<unsigned int> get_unsigned_int_value(std::string unsigned_int_name);

  bool has_float_value(const std::string& float_name) const;
  void add_float_value(const std::string float_name, std::shared_ptr<float> float_value);
  void set_float_value(const std::string float_name, const float float_value);
  // float get_float_value(std::string float_name) const;
  std::shared_ptr<float> get_float_value(std::string float_name);

protected:

private:
  GL_Shared() = default;

  std::map<std::string, std::shared_ptr<TextureXD> > textures_;
  std::map<std::string, std::shared_ptr<Buffer> > buffers_;
  std::map<std::string, std::shared_ptr<Vao> > vaos_;
  std::map<std::string, std::shared_ptr<int> > intValues_;
  std::map<std::string, std::shared_ptr<unsigned int> > unsignedIntValues_;
  std::map<std::string, std::shared_ptr<float> > floatValues_;

};


#endif // GL_CL_SHARED_H