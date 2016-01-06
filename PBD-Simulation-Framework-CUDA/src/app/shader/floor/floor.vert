#version 330

layout(location = 0) in vec3 vertex;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out vec3 position;

void main() {

  position = vertex;

  gl_Position = projection_matrix * view_matrix * vec4(vertex, 1.0f);

}