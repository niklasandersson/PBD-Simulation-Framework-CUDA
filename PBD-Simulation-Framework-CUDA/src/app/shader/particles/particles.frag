#version 330

in vec3 color;

out vec4 frag_color;

void main() {

  frag_color = vec4(0.0f, 1.0f, 0.0f, 1.0f);

  frag_color.rgb = color;

}
