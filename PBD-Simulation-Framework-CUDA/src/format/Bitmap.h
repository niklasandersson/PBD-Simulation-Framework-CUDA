#ifndef BITMAP_H
#define BITMAP_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <stdexcept>

#include <GL/glew.h>

#include "exception/Error.h"


class Bitmap {

public:
  Bitmap(const std::string& file);
  ~Bitmap();

  unsigned char* get_data();

  GLint get_width() const;
  GLint get_height() const;
  GLenum get_gl_channels() const;
  GLenum get_gl_data_type() const;

private:
  unsigned char* data_;
  unsigned int width_;
  unsigned int height_;
  unsigned int compression_;

  enum DIB_Types { 
                  DIB_BOTTOM_TO_TOP, // <--- Origin is at lower left corner
                  DIB_TOP_TO_BOTTOM  // <--- Origin is at upper left corner
                 };

  enum Compression_Types { 
                          BI_RGB,       // <--- Uncompressed format
                          BI_RLE8,
                          BI_RLE4,
                          BI_BITFIELDS,
                          BI_JPEG,
                          BI_PNG
                         };

  enum Bit_Count_Types { 
                        JPEG_OR_PNG_IMPLIED = 0,       
                        MONOCHROME = 1,
                        MAX_16_COLORS = 4,
                        MAX_256_COLORS = 8, // <--- Each byte represents a single pixel
                        MAX_2_POW_16_COLORS = 16, // <--- Each word represents a single pixel
                        MAX_2_POW_24_COLORS = 24, // <--- 3 byte pixel representation
                        MAX_2_POW_32_COLORS = 32 // <--- Each dword represents a single pixel
                       };

  unsigned int device_independent_bitmap_type_;

  #pragma pack(push, 1)
  typedef struct {
    unsigned short bfType;
    unsigned int   bfSize;
    unsigned short bfReserved1;
    unsigned short bfReserved2;
    unsigned int   bfOffBits;
  } Bitmap_File_Header;
  #pragma pack(pop)

  #pragma pack(push, 1)
  typedef struct {
    unsigned int   biSize;
    int            biWidth;
    int            biHeight;
    unsigned short biPlanes;
    unsigned short biBitCount;
    unsigned int   biCompression;
    unsigned int   biSizeImage;
    int            biXPelsPerMeter;
    int            biYPelsPerMeter;
    unsigned int   biClrUsed;
    unsigned int   biClrImportant;
  } Bitmap_Info_Header;
  #pragma pack(pop)

};


#endif // BITMAP_H