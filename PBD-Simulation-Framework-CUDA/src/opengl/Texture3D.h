#ifndef TEXTURE3D_H
#define TEXTURE3D_H


#include "TextureXD.h"


struct Texture3D : public TextureXD {
  Texture3D(std::string texture_name,
            GLenum target,
            GLint level,
            GLint internalFormat,
            GLsizei width,
            GLsizei height,
            GLsizei depth,
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
    depth_ = depth;
  }

};


#endif // TEXTURE3D_H