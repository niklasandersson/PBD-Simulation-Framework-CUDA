#version 330

layout(location = 0) in vec3 rawVertex;
layout(location = 1) in vec3 rawPosition;
layout(location = 2) in vec3 rawColor;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

uniform sampler2D positions4;

out vec3 color;

void main()
{ 
  color = vec3(1,0,1);

  const int widthi = 512; 
  // const int heighti = 512;

  const float one_divided_by_width_minus_onef = 1.0 / 511.0;
  const float one_divided_by_height_minus_onef = 1.0 / 511.0;


  vec2 uv = vec2( (gl_InstanceID % widthi) * one_divided_by_width_minus_onef, 
                  (gl_InstanceID / widthi) * one_divided_by_height_minus_onef );

  vec3 pos = texture2D(positions4, uv).xyz;



  gl_Position = projection_matrix * view_matrix * vec4(pos + rawVertex, 1.0f);
  // gl_Position = projection_matrix * view_matrix * vec4(pos + rawVertex, 1.0f);
  // gl_Position = projection_matrix * view_matrix * vec4(rawPosition + rawVertex, 1.0f);
}