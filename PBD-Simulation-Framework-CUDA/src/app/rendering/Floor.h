#ifndef FLOOR_H
#define FLOOR_H

#include <array>
#include <mutex>

#include "opengl/GL_Renderable.h"

#include "console/Console.h"

class Floor : public GL_Renderable {

public:
  Floor();
  ~Floor() = default;

  void render() override;

protected:

private:
  void generateFloor();

  glm::vec3 light_direction_{1.0f, -1.0f, 0.0f};

  std::vector<std::array<float, 3> > vertices_;
  std::vector<unsigned short> indices_;

};


#endif // FLOOR_H