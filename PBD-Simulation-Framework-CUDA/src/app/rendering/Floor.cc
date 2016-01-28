#include "Floor.h"


Floor::Floor() : GL_Renderable("program_floor") { 
  generateFloor();

  add_vao("floor_vao");
  add_buffer("floor_vertices");
  add_buffer("element_buffer");
  
  generateResources();

  add_uniform("view_matrix");
  add_uniform("projection_matrix");

  add_uniform("light_direction");

  add_uniform("camera_position");
  add_uniform("view_direction");

  bindVertexArray("floor_vao");

  bindBuffer("floor_vertices");
  glBufferData(GL_ARRAY_BUFFER, vertices_.size() * 3 * sizeof(float), &vertices_[0][0], GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  bindBuffer("element_buffer", GL_ELEMENT_ARRAY_BUFFER);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned short), &indices_[0], GL_STATIC_DRAW);

  unBindVertexArray();
}


void Floor::generateFloor() {
  const float width = 64.0f;

  vertices_.push_back({0.0f, 0.0f, 0.0f});
  vertices_.push_back({width, 0.0f, width});
  vertices_.push_back({0.0f, 0.0f, width});
  vertices_.push_back({width, 0.0f, 0.0f});

  vertices_.push_back({0.0f, width, 0.0f});
  vertices_.push_back({width, width, width});
  vertices_.push_back({0.0f, width, width});
  vertices_.push_back({width, width, 0.0f});

  indices_.push_back((unsigned short)0);
  indices_.push_back((unsigned short)2);
  indices_.push_back((unsigned short)1);
  indices_.push_back((unsigned short)0);
  indices_.push_back((unsigned short)1);
  indices_.push_back((unsigned short)3);
}


void Floor::render() {
  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);

  glUseProgram(program_);

  activateTextures();

  glUniform3f(get_uniform("light_direction"), light_direction_[0], light_direction_[1], light_direction_[2]);

  glUniform3f(get_uniform("camera_position"), camera_position_[0], camera_position_[1], camera_position_[2]);
  glUniform3f(get_uniform("view_direction"), view_direction_[0], view_direction_[1], view_direction_[2]);

  glUniformMatrix4fv(get_uniform("view_matrix"), 1, GL_FALSE, &view_matrix_[0][0]);
  glUniformMatrix4fv(get_uniform("projection_matrix"), 1, GL_FALSE, &projection_matrix_[0][0]);

  bindVertexArray("floor_vao");

  glDrawElements(GL_TRIANGLES, indices_.size(), GL_UNSIGNED_SHORT, nullptr);

  unBindVertexArray();
}