#version 330

in vec3 position;

uniform sampler2D compass;

out vec4 frag_color;

float filter_width(vec2 uv)
{
  vec2 fw = max(abs(dFdx(uv)), abs(dFdy(uv)));
  return max(fw.x, fw.y);
}


float checker(vec2 uv)
{
  float width = filter_width(uv);

  vec2 p0 = uv - 0.5f * width;
  vec2 p1 = uv + 0.5f * width;
  
  vec2 floor_p0 = floor(p0 / 2.0f);
  vec2 floor_p1 = floor(p1 / 2.0f);

  vec2 r = ( ( floor_p1 + 2.0f * max( (p1/2.0f) - floor_p1 - 0.5f, 0.0f ) ) - 
             ( floor_p0 + 2.0f * max( (p0/2.0f) - floor_p0 - 0.5f, 0.0f ) )   ) / width;

  return r.x * r.y + (1.0f - r.x) * (1.0f - r.y);
}


#define COLOR1 (vec3(255, 127, 0.0f) / 255.0f) // <--- Orange
// #define COLOR1 (vec3(80, 255, 255) / 255.0f) // <--- Light Blue
// #define COLOR1 (vec3(0, 0, 255) / 200.0f) // <--- Blue
#define COLOR2 vec3(0.12f, 0.12f, 0.12f)


void main() {

  vec2 uv = position.xz;

  float checker_value = checker(uv);

  // frag_color = vec4(mix(COLOR1, COLOR2, smoothstep(0.0f, 1.0f, checker_value)), 1.0f);
  frag_color = vec4(mix(mix(vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), smoothstep(-4.0f, 4.0f, position.x)), COLOR2, smoothstep(0.0f, 1.0f, checker_value)), 1.0f);

  if( position.z > 0 ) {
    frag_color.xyz = vec3(mix(frag_color.xyz, mix(COLOR1, COLOR2, smoothstep(0.0f, 1.0f, checker_value)), smoothstep(0.0f, 4.0f, position.z)));
  } 

  // #define ORIGO 0.0f
  // if( (position.x > - 1.0f + ORIGO && position.x < 1.0f + ORIGO) && (position.z > - 1.0f + ORIGO && position.z < 1.0f + ORIGO) ) {
  //   frag_color = vec4(1.0f);
  // }

  /*
  float mag = dot(position.xz, position.xz);

  if( mag < 1 ) {
    // frag_color = vec4(COLOR2,1);
    // frag_color.rgb += vec3(1.0f)*smoothstep(0, gl_FragCoord.z/10.0, 1-mag);
    uv.x += 0.029;
    uv.y = -uv.y;
    uv.y -= 0.005;
    float cmp = 1 - texture2D(compass, (uv + vec2(1, 1)) * 0.5).a;
    frag_color.rgb = mix(frag_color.rgb, texture2D(compass, (uv + vec2(1, 1)) * 0.5).rgb,  smoothstep(0, gl_FragCoord.z/10.0, 1-mag));
  }
  */


  frag_color.w = gl_FragCoord.z; // HAX

}