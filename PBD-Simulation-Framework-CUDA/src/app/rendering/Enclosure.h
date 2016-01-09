#ifndef ENCLOSURE_H
#define ENCLOSURE_H

#include <array>

#include "opengl/GL_Renderable.h"

#include "console/Console.h"

class Enclosure : public GL_Renderable {

public:
  Enclosure();
  ~Enclosure();

  void render() override;

protected:

private:
  void generateEnclosure();

  std::vector<std::array<float, 3> > vertices_;
  std::vector<unsigned short> indices_;

};

#endif // ENCLOSURE_H