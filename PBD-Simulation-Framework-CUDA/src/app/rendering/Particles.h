#ifndef PARTICLES_H
#define PARTICLES_H

#include <random>

#include "opengl/GL_Renderable.h"

#include "console/Console.h"

#include "parser/Parser.h"
#include "parser/OBJParser.h"
#include "parser/Config.h"

#include "event/Events.h"
#include "event/Delegate.h"


class Particles : public GL_Renderable {

public:
  Particles();
  ~Particles() = default;

  void render() override;

  void clickCallback(const double position_x,
                     const double position_y,
                     const int button,
                     const int action,
                     const int mods);

protected:

private:
  void generateParticles();
  void addConsoleCommands();
  void registerSharedVariables();

  Delegate<void(const double, const double, const int, const int, const int)> clicked_;
  std::shared_ptr<unsigned int> numberOfParticles_;
  std::shared_ptr<unsigned int> maxParticles_;
  std::shared_ptr<unsigned int> maxGrid_;
  unsigned int initialNumberOfParticles_;

  glm::vec3 light_direction_{1.0f, -1.0f, 0.0f};

  std::vector<float> vertices_;
  std::vector<float> normals_;
  std::vector<unsigned short> indices_;

  std::vector<glm::vec3> positons_;
  std::vector<glm::vec4> positons4_;
  std::vector<glm::vec4> velocities4_;
  std::vector<glm::vec4> colors4_;
  std::vector<glm::vec4> collisionDeltas4_;
  std::vector<float> densities_;

  std::vector<glm::vec3> colors_;

  OBJParser<Parser> objParser_;

};


#endif // PARTICLES_H