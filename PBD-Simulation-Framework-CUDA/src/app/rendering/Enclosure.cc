#include "Enclosure.h"

Enclosure::Enclosure() : GL_Renderable("program_enclosure") {
  
  generateEnclosure();

  add_vao("enclosure_vao");
  add_buffer("enclosure_vertices");
  add_buffer("element_buffer");
  
  generateResources();

  add_uniform("view_matrix");
  add_uniform("projection_matrix");

  bindVertexArray("enclosure_vao");

  bindBuffer("enclosure_vertices");
  glBufferData(GL_ARRAY_BUFFER, vertices_.size() * 3 * sizeof(float), &vertices_[0][0], GL_STATIC_DRAW);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

  bindBuffer("element_buffer", GL_ELEMENT_ARRAY_BUFFER);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices_.size() * sizeof(unsigned short), &indices_[0], GL_STATIC_DRAW);

  unBindVertexArray();
}


Enclosure::~Enclosure() {

}


void Enclosure::generateEnclosure() {

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

  indices_.push_back((unsigned short)5);
  indices_.push_back((unsigned short)6);
  indices_.push_back((unsigned short)4);
  indices_.push_back((unsigned short)7);
  indices_.push_back((unsigned short)5);
  indices_.push_back((unsigned short)4);

  indices_.push_back((unsigned short)0);
  indices_.push_back((unsigned short)7);
  indices_.push_back((unsigned short)4);
  indices_.push_back((unsigned short)0);
  indices_.push_back((unsigned short)3);
  indices_.push_back((unsigned short)7);

  indices_.push_back((unsigned short)3);
  indices_.push_back((unsigned short)5);
  indices_.push_back((unsigned short)7);
  indices_.push_back((unsigned short)3);
  indices_.push_back((unsigned short)1);
  indices_.push_back((unsigned short)5);

  indices_.push_back((unsigned short)4);
  indices_.push_back((unsigned short)6);
  indices_.push_back((unsigned short)2);
  indices_.push_back((unsigned short)4);
  indices_.push_back((unsigned short)2);
  indices_.push_back((unsigned short)0);

  indices_.push_back((unsigned short)2);
  indices_.push_back((unsigned short)6);
  indices_.push_back((unsigned short)1);
  indices_.push_back((unsigned short)6);
  indices_.push_back((unsigned short)5);
  indices_.push_back((unsigned short)1);

}


void Enclosure::render() {

  glDisable(GL_CULL_FACE);
  glEnable(GL_DEPTH_TEST);
  
  glUseProgram(program_);

  glUniformMatrix4fv(get_uniform("view_matrix"), 1, GL_FALSE, &view_matrix_[0][0]);
  glUniformMatrix4fv(get_uniform("projection_matrix"), 1, GL_FALSE, &projection_matrix_[0][0]);

  bindVertexArray("enclosure_vao");

  glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  glDrawElements(GL_TRIANGLES, indices_.size(), GL_UNSIGNED_SHORT, nullptr);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  unBindVertexArray();

}