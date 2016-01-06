#ifndef TEXTURE2D_H
#define TEXTURE2D_H


#include "TextureXD.h"


struct Texture2D : public TextureXD {
  Texture2D(std::string texture_name,
            GLenum target,
            GLint level,
            GLint internalFormat,
            GLsizei width,
            GLsizei height,
            GLint border,
            GLenum format,
            GLenum type,
            void* data)
    : TextureXD{TextureXD::Dim::TWO_DIM,
                texture_name, 
                target,
                level,
                internalFormat,
                width,
                border,
                format,
                type,
                data}
  {
    height_ = height;
  }

};


#endif // TEXTURE2D_H