#include "Bitmap.h"


Bitmap::Bitmap(const std::string& filename) {

  std::ifstream input_file(filename.c_str());

  if( !input_file ) {
    throw std::invalid_argument{ report_error( "Could not open file '" << filename << "'." ) };
  }

  input_file.seekg(0, std::ios::end);
  std::streampos file_length = input_file.tellg();
  input_file.seekg(0, std::ios::beg);

  std::vector<char> buffer;
  buffer.resize(file_length);
  input_file.read(&buffer[0], file_length);
  
  input_file.close();

  Bitmap_File_Header* file_header;
  Bitmap_Info_Header* info_header;

  if( file_length >= sizeof(Bitmap_File_Header) ) {
    file_header = (Bitmap_File_Header*)(&buffer[0]);
    if( buffer[0] != 'B' || buffer[1] != 'M' ) {
      throw std::invalid_argument{ report_error( "The file type is not Bitmap." ) };
    }
  } else {
    throw std::invalid_argument{ report_error( "Bitmap invalid." ) };
  }
  
  if( file_length >= sizeof(Bitmap_File_Header) + sizeof(Bitmap_Info_Header) ) {
    info_header = (Bitmap_Info_Header*)(&buffer[0] + sizeof(Bitmap_File_Header));
  } else {
    throw std::invalid_argument{ report_error( "Bitmap invalid." ) };
  }

  if( info_header->biHeight > 0 ) {
    device_independent_bitmap_type_ = DIB_BOTTOM_TO_TOP;
  } else {
    device_independent_bitmap_type_ = DIB_TOP_TO_BOTTOM;
    info_header->biHeight *= -1;
    if( info_header->biCompression != BI_RGB || info_header->biCompression != BI_BITFIELDS ) {
      throw std::invalid_argument{ report_error( "Bitmap invalid." ) };
    }
  }

  width_ = info_header->biWidth;
  height_ = info_header->biHeight;
  compression_ = info_header->biCompression;



  if( info_header->biBitCount == MAX_2_POW_24_COLORS && info_header->biCompression == BI_RGB ) {

    unsigned int size = 3 * info_header->biWidth * info_header->biHeight;
    data_ = new unsigned char[size]; 
    for(unsigned int i=0; i<size; i++) {
      data_[i] = 0;
    }

    for(unsigned int i=0; i<size; i+=3) {
      data_[i + 0] = buffer[file_header->bfOffBits + i + 0];
      data_[i + 1] = buffer[file_header->bfOffBits + i + 1];
      data_[i + 2] = buffer[file_header->bfOffBits + i + 2];
    }

    return;
  }



  if( info_header->biBitCount != MAX_2_POW_32_COLORS ) {
    // std::cout << "info_header->biBitCount = " << info_header->biBitCount << std::endl;
    throw std::invalid_argument{ report_error( "Bitmap type not supported." ) };
  }

  if( info_header->biCompression != BI_BITFIELDS ) {
    // std::cout << "info_header->biCompression = " << info_header->biCompression << std::endl;
    throw std::invalid_argument{ report_error( "Bitmap type not supported." ) };
  }

  if( ((int)(file_length) - file_header->bfOffBits) != info_header->biWidth * info_header->biHeight * 4 ) {
    throw std::invalid_argument{ report_error( "Bitmap type not supported." ) };
  }

  if( info_header->biWidth % 2 != 0 || info_header->biHeight % 2 != 0 ) {
    throw std::invalid_argument{ report_error( "Bitmap type not supported." ) };
  }

  unsigned int size = 4 * info_header->biWidth * info_header->biHeight;
  data_ = new unsigned char[size]; 
  for(unsigned int i=0; i<size; i++) {
    data_[i] = 0;
  }

  for(unsigned int i=0; i<size; i+=4) {
    data_[i + 0] = buffer[file_header->bfOffBits + i + 3];
    data_[i + 1] = buffer[file_header->bfOffBits + i + 2];
    data_[i + 2] = buffer[file_header->bfOffBits + i + 1];
    data_[i + 3] = buffer[file_header->bfOffBits + i + 0];
  }


}


Bitmap::~Bitmap() {
  if( data_ ) {
    delete [] data_;
    data_ = nullptr;
  }
}


unsigned char* Bitmap::get_data() {
  return data_;
}


GLint Bitmap::get_width() const {
  return width_;
}


GLint Bitmap::get_height() const {
  return height_;
}


GLenum Bitmap::get_gl_channels() const {

  switch( compression_ ) {
    case BI_RGB : return GL_RGB;
    case BI_BITFIELDS : return GL_RGBA;
  }

}


GLenum Bitmap::get_gl_data_type() const {
  return GL_UNSIGNED_BYTE;
}