#version 450
layout(push_constant) uniform PushConsts {
    float time;
}
push;

layout(set = 0, binding = 0) uniform texture2D tex;
layout(set = 0, binding = 1) uniform sampler samp;

layout(location = 1) in vec3 frag_color;
layout(location = 2) in vec2 frag_uv;

layout(location = 0) out vec4 color;

void main() {
    // float time1 = -0.9 * abs(sin(push.time * 0.9)) + 0.9;
    // color = vec4(frag_color, 1.0) * vec4(time1, time1, time1, 1.0);
    vec4 tex_color = texture(sampler2D(tex, samp), frag_uv);
    color = tex_color;
}