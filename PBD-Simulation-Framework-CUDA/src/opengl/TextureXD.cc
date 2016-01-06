#include "TextureXD.h"


TextureXD::TextureXD(Dim dim,
                     std::string texture_name, 
                     GLenum target,
                     GLint level,
                     GLint internalFormat,
                     GLsizei width,
                     GLint border,
                     GLenum format,
                     GLenum type,
                     void* data) 
: texture_name_(texture_name)
, dim_(dim)
, target_(target)
, level_(level)
, internalFormat_(internalFormat)
, width_(width)
, border_(border)
, format_(format)
, type_(type)
, data_(data)
, height_(1)
, depth_(1)
, generated_(false)
{
  parameters_ = [&]() {
    glTexParameteri(target_, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(target_, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(target_, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
  };
}


TextureXD::~TextureXD() {

}