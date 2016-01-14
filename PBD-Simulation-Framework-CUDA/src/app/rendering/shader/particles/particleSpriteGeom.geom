#version 330
// #version 330 core
// #extension GL_EXT_gpu_shader4 : enable 

// layout(points) in;
// layout(triangle_strip, max_vertices=4) out;
layout (points) in;
layout (triangle_strip) out;
layout (max_vertices = 4) out; 

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

in vec3 colorVert[];

out vec3 color;

out vec2 uv;

out vec4 viewPos;
out vec3 L;


void main()
{

  vec4 pos = gl_in[0].gl_Position;
  viewPos = view_matrix * pos; 
  color = colorVert[0];
  float sphere_radius = 0.5;
  
  vec3 lightPos = vec3(0,0,10);
  L = normalize(((vec4((lightPos-pos.xyz), 1))).xyz);
  
  uv = vec2(-1.0,-1.0);
  gl_Position = viewPos;
  gl_Position.xy += vec2(-sphere_radius, -sphere_radius);
  gl_Position = projection_matrix  * gl_Position;
  EmitVertex();

  uv = vec2(1.0,-1.0);
  gl_Position = viewPos;
  gl_Position.xy += vec2(sphere_radius, -sphere_radius);
  gl_Position = projection_matrix  * gl_Position;
  EmitVertex();

  uv = vec2(-1.0,1.0);
  gl_Position = viewPos;
  gl_Position.xy += vec2(-sphere_radius, sphere_radius);
  gl_Position = projection_matrix  * gl_Position;
  EmitVertex();

  uv = vec2(1.0,1.0);
  gl_Position = viewPos;
  gl_Position.xy += vec2(sphere_radius, sphere_radius);
  gl_Position = projection_matrix  * gl_Position;
  EmitVertex();

  EndPrimitive();
}