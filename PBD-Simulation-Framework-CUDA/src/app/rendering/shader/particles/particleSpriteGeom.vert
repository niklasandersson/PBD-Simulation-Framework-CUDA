#version 330
// #version 330 core
// #extension GL_EXT_gpu_shader4 : enable

// layout(location = 0) in vec3 rawVertex;
// layout(location = 1) in vec3 rawPosition;
// layout(location = 2) in vec3 rawColor;

layout(location = 0) in float density;

uniform mat4 view_matrix;
uniform mat4 projection_matrix;

uniform sampler2D positions4;
uniform sampler2D colors4;

out vec3 colorVert;
out vec3 posen;
out vec4 viewPos;

out vec3 L;

void main()
{
  //color = vec3(1,0,1);


  const int widthi = 256; 
  // const int heighti = 512;

  const float one_divided_by_width_minus_onef = 1.0 / 255.0;
  const float one_divided_by_height_minus_onef = 1.0 / 255.0;


  vec2 uv = vec2( (gl_InstanceID % widthi) * one_divided_by_width_minus_onef, 
                  (gl_InstanceID / widthi) * one_divided_by_height_minus_onef );

  vec3 pos = texture2D(positions4, uv).xyz;

  colorVert = texture2D(colors4, uv).xyz;
  colorVert = vec3(density, density, density);
  // colorVert = vec3(1,0,0);
  gl_Position = vec4(pos, 1.0);


  // // vec3 lightPos = vec3(0.577, 0.577, 0.577);
  // vec3 lightPos = vec3(0,0,10);
  // L = normalize(((vec4((lightPos-pos), 1))).xyz);
  // // L = normalize(((view_matrix * vec4((lightPos-pos), 1))).xyz);


  // color = vec3(clamp((pos.x+3)/6.0f, 0.0f, 1.0f), clamp(pos.y/10.0f, 0.0f, 1.0f), clamp((pos.z+2)/5.0f, 0.0f, 1.0f));


  // const float pointRadius = 5.0f; // point size in world space
  // const float pointScale = 120.0f; // scale to calculate size in pixels

  // // Calculate window-space point size
  // vec3 posEye = vec3(view_matrix * vec4(pos, 1.0));
  // float dist = length(posEye);
  // gl_PointSize = pointRadius * (pointScale / dist);

  // // float screenWidth = 800;
  // // float sphere_radius = 1.0f;
  // // vec4 eye_position = view_matrix * vec4(rawVertex, 1.0);
  // // vec4 projCorner = projection_matrix * vec4(sphere_radius, sphere_radius, eye_position.z, eye_position.w);
  // // gl_PointSize = screenWidth * projCorner.x / projCorner.w;


  // viewPos = view_matrix * vec4(pos, 1.0f);
  // gl_Position = projection_matrix * viewPos;
  // // gl_Position = projection_matrix * view_matrix * vec4(pos, 1.0f);

  // posen = pos;
}