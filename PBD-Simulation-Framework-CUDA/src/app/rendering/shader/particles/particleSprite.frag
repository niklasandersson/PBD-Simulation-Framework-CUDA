#version 330


uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat4 rotation_matrix;
uniform mat4 inverse_rotation_matrix;
uniform vec3 view_direction;


in vec3 color;
in vec3 posen;
in vec4 viewPos;
in vec3 L;

out vec4 frag_color;
out float gl_FragDepth;

void main()
{ 

  vec3 N;
  N.xy = gl_PointCoord * vec2(2.0, -2.0) + vec2(-1.0, 1.0);

  float mag = dot(N.xy, N.xy);

  if( mag > 1 ) {
    discard;   
  }

  N.z = sqrt(1 - mag);

  float displaceZ = N.z;

  vec4 mult = inverse_rotation_matrix*vec4(N,1);

  N = normalize(mult.xyz);

  vec3 V = normalize(-view_direction);
  vec3 R = reflect(-L, N);

  float ka = 0.3f;
  float kd = 0.7f;
  float ks = 0.2f;

  vec3 diffuseColor = color;
  vec3 ambientColor = diffuseColor;
  vec3 specularColor = vec3(1,1,1);

  float specularity = 10;

  frag_color.rgb = ka * ambientColor
                 + kd * max(0.0, dot(L, N)) * diffuseColor
                 + ks * pow(clamp(dot(R, V), 0.0f, 1.0f), specularity) * specularColor;

  frag_color.a = 1.0f;

  // frag_color = vec4(N, 1.0f);
  // frag_color = vec4(displaceZ, 0, 0, 1.0f);


  float sphere_radius = 0.5;
  vec4 pos = viewPos;
  pos.z += sphere_radius * displaceZ;
  pos = projection_matrix * pos;
  gl_FragDepth = 0.5*(pos.z / pos.w)+0.5;


}