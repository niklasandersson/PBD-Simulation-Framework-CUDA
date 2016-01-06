#ifndef TEXTURE1D_H
#define TEXTURE1D_H


#include "TextureXD.h"


struct Texture1D : public TextureXD {
  Texture1D(std::string texture_name,
            GLenum target,
            GLint level,
            GLint internalFormat,
            GLsizei width,
            GLint border,
            GLenum format,
            GLenum type,
            void* data)
    : TextureXD{TextureXD::Dim::ONE_DIM,
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
  }

};


#endif // TEXTURE1D_H