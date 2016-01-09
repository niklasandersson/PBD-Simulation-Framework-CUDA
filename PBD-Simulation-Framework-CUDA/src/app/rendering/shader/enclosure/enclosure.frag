#version 330

out vec4 frag_color;

void main() { 
	
	float val = clamp(1.0f - pow(gl_FragCoord.z, 92), 0.1f, 1.0f);

	frag_color = vec4(val, val, val, 1.0f);

}