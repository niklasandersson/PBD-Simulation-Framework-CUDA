#include "Particles.h"

Particles::Particles()
  : GL_Renderable("program_particles_sprite_geom")
  // : GL_Renderable("program_particles_sprite")
  // : GL_Renderable("program_particles")
  , clicked_(Delegate<void(const double, const double, const int, const int, const int)>::from<Particles, &Particles::clickCallback>(this))
  , numberOfParticles_(new unsigned int{0})
{
  Events::click.subscribe(clicked_);

  generateParticles();

  add_vao("particles_vao");
  // add_buffer("particle_vertices");
  // add_shared_buffer("particle_positions");
  // add_shared_buffer("particle_colors");
  add_buffer("element_buffer");


  positons4_.resize(256 * 256);
  colors4_.resize(256 * 256);
  velocities4_.resize(256 * 256);

  add_shared_texture2D("positions4", 256, 256, &positons4_[0][0]);
  add_shared_texture2D("predictedPositions4", 256, 256, &positons4_[0][0]);
  add_shared_texture2D("velocities4", 256, 256, &velocities4_[0][0]);
  add_shared_texture2D("colors4", 256, 256, &colors4_[0][0]);

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

  // bindBuffer("particle_vertices");
  // bufferData(GL_ARRAY_BUFFER, vertices_.size() * sizeof(float), &vertices_[0], GL_STATIC_DRAW);
  // enableVertexAttribArray(0);
  // vertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  // vertexAttribDivisor(0, 0);

  // bindBuffer("particle_positions");
  // bufferData(GL_ARRAY_BUFFER, positons_.size() * sizeof(glm::vec3), &positons_[0][0], GL_STATIC_DRAW);
  // enableVertexAttribArray(1);
  // vertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  // vertexAttribDivisor(1, 1);

  // bindBuffer("particle_colors");
  // bufferData(GL_ARRAY_BUFFER, colors_.size() * sizeof(glm::vec3), &colors_[0][0], GL_STATIC_DRAW);
  // enableVertexAttribArray(2);
  // vertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
  // vertexAttribDivisor(2, 1);

  bindBuffer("element_buffer", GL_ELEMENT_ARRAY_BUFFER);
  bufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned short), &indices_[0], GL_STATIC_DRAW);

  unBindVertexArray();


}


Particles::~Particles() {

}


void Particles::clickCallback(const double position_x,
  const double position_y,
  const int button,
  const int action,
  const int mods) {

  // std::cout << "CLICK" << std::endl;
  if (button == 0 && action == 1) {
    // if( (*numberOfParticles_) < positons4_.size() ) {
    //   (*numberOfParticles_)++;
    // }
  }

}


void Particles::generateParticles() {

  objParser_.parseFile("assets/sphere.obj"); // radius = 2

  vertices_ = objParser_.getVertices();
  normals_ = objParser_.getNormals();
  indices_ = objParser_.getVertexIndices();


  // std::cout << "VERTICES: " << vertices_.size() << std::endl;
  // std::cout << "INDICES: " << indices_.size() << std::endl;
  // std::cout << "Total vertices: " << indices_.size() * 512*512 << std::endl;
  // std::cout << "Triangles: " << (indices_.size() * 512*512) / 3 << std::endl;

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(0, 1);

  // for(unsigned int i=0; i<3; i++) {
  //   // positons_.push_back(glm::vec3{-3, 2 + 3*i, 0});
  //   // positons_.push_back(glm::vec3{0, 2 + 3*i, 0});
  //   // positons_.push_back(glm::vec3{3, 2 + 3*i, 0});

  //   // colors_.push_back(glm::vec3{distribution(generator), distribution(generator), distribution(generator)});
  //   // colors_.push_back(glm::vec3{distribution(generator), distribution(generator), distribution(generator)});
  //   // colors_.push_back(glm::vec3{distribution(generator), distribution(generator), distribution(generator)});

  //   positons4_.push_back(glm::vec4{5 + -0.5, 2 + 3*i, 5, 0});
  //   positons4_.push_back(glm::vec4{5 + 0, + 3*i, 5, 0});
  //   positons4_.push_back(glm::vec4{5 + 0.5, 2 + 3*i, 5, 0});

  // }


  const float offset = 4;
  const float scale = 1.5f;
  const unsigned int width = 3;
  for (unsigned int i = 0; i<width; i++) {
    for (unsigned int j = 0; j<width; j++) {
      for (unsigned int k = 0; k<width; k++) {
        positons4_.push_back(glm::vec4{ offset + i*scale, 10 + offset + j*scale, offset + k*scale, 0 });
        velocities4_.push_back(glm::vec4{ 0, 0, 0, 0 });
      }
    }
  }

  // positons4_.push_back(glm::vec4{30, 10, 15, 0});
  // positons4_.push_back(glm::vec4{5, 10, 15, 0});
  // velocities4_.push_back(glm::vec4{1, 0, 0, 0});
  // velocities4_.push_back(glm::vec4{-1, 0, 0, 0});



  for (unsigned int i = 0; i<256 * 256; i++) {
    colors4_.push_back(glm::vec4{ distribution(generator), distribution(generator), distribution(generator), 1 });
    // colors_.push_back(glm::vec3{distribution(generator), distribution(generator), distribution(generator)});
  }

  *numberOfParticles_ = positons4_.size();
  // std::cout << "positons4_.size() = " << positons4_.size() << std::endl;

  GL_Shared::getInstance().add_unsigned_int_value("numberOfParticles", numberOfParticles_);
  GL_Shared::getInstance().add_unsigned_int_value("gridSize", std::shared_ptr<unsigned int>{new unsigned int{ 100 * 100 * 100 }});
  GL_Shared::getInstance().add_float_value("time", std::shared_ptr<float>{new float{ 0 }});

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

  indices_ = std::vector<unsigned short>{0};
  glDrawElementsInstanced(GL_POINTS, indices_.size(), GL_UNSIGNED_SHORT, nullptr, *numberOfParticles_);

  unBindVertexArray();

  Events::click.execute_calls();

}
