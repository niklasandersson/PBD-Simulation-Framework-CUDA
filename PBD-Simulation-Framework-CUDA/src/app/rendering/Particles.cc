#include "Particles.h"

Particles::Particles()
  : GL_Renderable("program_particles_sprite_geom")
  , clicked_(Delegate<void(const double, const double, const int, const int, const int)>::from<Particles, &Particles::clickCallback>(this))
  , numberOfParticles_(new unsigned int{0})
  , maxParticles_(new unsigned int{65536})
  , maxGrid_(new unsigned int{128 * 128 * 128})
{
  Events::click.subscribe(clicked_);

  generateParticles();

  registerSharedVariables();

  add_vao("particles_vao");
  add_buffer("element_buffer");

  const unsigned int textureWidth = 256;

  positons4_.resize(textureWidth * textureWidth);
  colors4_.resize(textureWidth * textureWidth);
  velocities4_.resize(textureWidth * textureWidth);

  add_shared_texture2D("positions4", textureWidth, textureWidth, &positons4_[0][0]);
  add_shared_texture2D("predictedPositions4", textureWidth, textureWidth, &positons4_[0][0]);
  add_shared_texture2D("velocities4", textureWidth, textureWidth, &velocities4_[0][0]);
  add_shared_texture2D("colors4", textureWidth, textureWidth, &colors4_[0][0]);

  add_shared_texture2D("positions4Copy", textureWidth, textureWidth, &positons4_[0][0]);
  add_shared_texture2D("predictedPositions4Copy", textureWidth, textureWidth, &positons4_[0][0]);
  add_shared_texture2D("velocities4Copy", textureWidth, textureWidth, &velocities4_[0][0]);
  add_shared_texture2D("colors4Copy", textureWidth, textureWidth, &colors4_[0][0]);

  add_shared_buffer("d_densities");
  add_shared_buffer("d_positions");
  add_shared_buffer("d_predictedPositions");
  add_shared_buffer("d_velocities");
  add_shared_buffer("d_colors");

  add_shared_buffer("d_densitiesCopy");
  add_shared_buffer("d_positionsCopy");
  add_shared_buffer("d_predictedPositionsCopy");
  add_shared_buffer("d_velocitiesCopy");
  add_shared_buffer("d_colorsCopy");

  generateResources();

  add_uniform("view_matrix");
  add_uniform("inverse_view_matrix");
  add_uniform("projection_matrix");
  add_uniform("rotation_matrix");
  add_uniform("inverse_rotation_matrix");
  add_uniform("light_direction");
  add_uniform("camera_position");
  add_uniform("view_direction");

  bindVertexArray("particles_vao");

  densities_.resize(*maxParticles_);
  bindBuffer("d_densities");
  bufferData(GL_ARRAY_BUFFER, densities_.size() * sizeof(float), &densities_[0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(0);
  vertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(0, 1);

  positons4_.resize(*maxParticles_);
  bindBuffer("d_positions");
  bufferData(GL_ARRAY_BUFFER, positons4_.size() * 4 * sizeof(float), &positons4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(1);
  vertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(1, 1);

  positons4_.resize(*maxParticles_);
  bindBuffer("d_predictedPositions");
  bufferData(GL_ARRAY_BUFFER, positons4_.size() * 4 * sizeof(float), &positons4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(2);
  vertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(2, 1);

  velocities4_.resize(*maxParticles_);
  bindBuffer("d_velocities");
  bufferData(GL_ARRAY_BUFFER, velocities4_.size() * 4 * sizeof(float), &velocities4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(3);
  vertexAttribPointer(3, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(3, 1);

  colors4_.resize(*maxParticles_);
  bindBuffer("d_colors");
  bufferData(GL_ARRAY_BUFFER, colors4_.size() * 4 * sizeof(float), &colors4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(4);
  vertexAttribPointer(4, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(4, 1);

  densities_.resize(*maxParticles_);
  bindBuffer("d_densitiesCopy");
  bufferData(GL_ARRAY_BUFFER, densities_.size() * sizeof(float), &densities_[0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(5);
  vertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(5, 1);

  positons4_.resize(*maxParticles_);
  bindBuffer("d_positionsCopy");
  bufferData(GL_ARRAY_BUFFER, positons4_.size() * 4 * sizeof(float), &positons4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(6);
  vertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(6, 1);

  positons4_.resize(*maxParticles_);
  bindBuffer("d_predictedPositionsCopy");
  bufferData(GL_ARRAY_BUFFER, positons4_.size() * 4 * sizeof(float), &positons4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(7);
  vertexAttribPointer(7, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(7, 1);

  velocities4_.resize(*maxParticles_);
  bindBuffer("d_velocitiesCopy");
  bufferData(GL_ARRAY_BUFFER, velocities4_.size() * 4 * sizeof(float), &velocities4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(8);
  vertexAttribPointer(8, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(8, 1);

  colors4_.resize(*maxParticles_);
  bindBuffer("d_colorsCopy");
  bufferData(GL_ARRAY_BUFFER, colors4_.size() * 4 * sizeof(float), &colors4_[0][0], GL_DYNAMIC_DRAW);
  enableVertexAttribArray(9);
  vertexAttribPointer(9, 4, GL_FLOAT, GL_FALSE, 0, nullptr);
  vertexAttribDivisor(9, 1);

  bindBuffer("element_buffer", GL_ELEMENT_ARRAY_BUFFER);
  bufferData(GL_ELEMENT_ARRAY_BUFFER, 1 * sizeof(unsigned short), nullptr, GL_STATIC_DRAW);
  unBindVertexArray();

  addConsoleCommands();
}


void Particles::addConsoleCommands() {
  auto console = Console::getInstance();
  console->add("n", [&](const char* argv) {
    std::cout << "NumberOfParticles: " << *numberOfParticles_ << std::endl;
  });
  console->add("r", [&](const char* argv) {
    Events::reload();
  });
  console->add("s", [&](const char* argv) {
    Events::addParticles(initialNumberOfParticles_, positons4_, velocities4_, colors4_);
  });
  console->add("c", [&](const char* argv) {
    Events::clearParticles();
  });
}


void Particles::clickCallback(const double position_x, const double position_y, const int button, const int action, const int mods) {
  if( button == 0 && action == 1 ) {
    Events::addParticle(camera_position_, view_direction_);
  } else if( button == 1 && action == 1 ) {
    Events::addParticles(initialNumberOfParticles_, positons4_, velocities4_, colors4_);
  } else if( button == 2 && action == 1 ) {
    Events::clearParticles();
  }
}


void Particles::generateParticles() {
  Config& config = Config::getInstance();

  const float offset = 30;
  const float scale = config.getValue<float>("Application.Sim.particlesScale"); // 0.99f
  const unsigned int width = config.getValue<unsigned int>("Application.Sim.particlesWidth"); // 32
  for (unsigned int i = 0; i<width; i++) {
    for (unsigned int j = 0; j<width; j++) {
      for (unsigned int k = 0; k<width; k++) {
        positons4_.push_back(glm::vec4{ offset + i*scale, 2 + offset + j*scale, offset + k*scale, 0 });
        velocities4_.push_back(glm::vec4{ 0, 0, 0, 0 });
        collisionDeltas4_.push_back(glm::vec4{ 0, 0, 0, 0 });
      }
    }
  }

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0, 1);
  for (unsigned int i = 0; i<256 * 256; i++) {
    colors4_.push_back(glm::vec4{ distribution(generator), distribution(generator), distribution(generator), 1 });
  }

  for(unsigned int i=0; i<*maxParticles_; i++) {
    densities_.push_back(0.0f);
  }

  *numberOfParticles_ = positons4_.size();
  initialNumberOfParticles_ = positons4_.size();
}


void Particles::registerSharedVariables() {
  GL_Shared& glShared = GL_Shared::getInstance();
  glShared.add_unsigned_int_value("numberOfParticles", numberOfParticles_);
  glShared.add_float_value("time", std::shared_ptr<float>{new float{0}});
  glShared.add_unsigned_int_value("maxParticles", maxParticles_);
  glShared.add_unsigned_int_value("maxGrid", maxGrid_);
}


void Particles::render() {
  glm::mat4 L = view_matrix_;
  L[3][0] = 0.0;
  L[3][1] = 0.0;
  L[3][2] = 0.0;

  L[0][3] = 0.0;
  L[1][3] = 0.0;
  L[2][3] = 0.0;

  glm::mat4 rotation_matrix = L;
  glm::mat4 inverse_rotation_matrix = glm::inverse(L);
  glm::mat4 inverse_view_matrix = glm::inverse(view_matrix_);

  GL_Shared::getInstance().set_float_value("time", current_time_);

  glEnable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);

  useProgram();

  activateTextures();

  glUniform3f(get_uniform("light_direction"), light_direction_[0], light_direction_[1], light_direction_[2]);

  glUniform3f(get_uniform("camera_position"), camera_position_[0], camera_position_[1], camera_position_[2]);
  glUniform3f(get_uniform("view_direction"), view_direction_[0], view_direction_[1], view_direction_[2]);

  glUniformMatrix4fv(get_uniform("view_matrix"), 1, GL_FALSE, &view_matrix_[0][0]);
  glUniformMatrix4fv(get_uniform("inverse_view_matrix"), 1, GL_FALSE, &inverse_view_matrix[0][0]);
  glUniformMatrix4fv(get_uniform("projection_matrix"), 1, GL_FALSE, &projection_matrix_[0][0]);
  glUniformMatrix4fv(get_uniform("rotation_matrix"), 1, GL_FALSE, &rotation_matrix[0][0]);
  glUniformMatrix4fv(get_uniform("inverse_rotation_matrix"), 1, GL_FALSE, &inverse_rotation_matrix[0][0]);

  bindVertexArray("particles_vao");
  glDrawElementsInstanced(GL_POINTS, 1, GL_UNSIGNED_SHORT, nullptr, *numberOfParticles_);
  unBindVertexArray();

  Events::click.execute_calls();
}
