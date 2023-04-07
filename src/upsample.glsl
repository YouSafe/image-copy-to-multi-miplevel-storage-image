#version 450

layout(push_constant) uniform Pass {
    // mip level of the input image
    int mipLevel;
    // size of one texel in the input image;
    vec2 texelSize;
} pass;

layout(set = 0, binding = 0) uniform sampler2D inputImage;
layout(set = 0, binding = 1, rgba16f) uniform image2D outputImage;

// one invokation for every texel in the output image
// output image has **DOUBLE** the dimensions of the input image
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

vec3 textureSample(vec2 uv, vec2 offset) {
    return textureLod(inputImage, uv + offset * pass.texelSize, float(0)).rgb;
}

void main() {
    ivec2 texel_output = ivec2(gl_GlobalInvocationID.xy);
    vec2 texel_input = vec2(texel_output) / 2.0;

    vec2 uv = texel_input * pass.texelSize;

    // Source: https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
    // Take 9 samples around current texel:
    // a - b - c
    // d - e - f
    // g - h - i
    // === ('e' is the current texel) ===
    vec3 a = textureSample(uv, vec2(-1,1));
    vec3 b = textureSample(uv, vec2(0,1));
    vec3 c = textureSample(uv, vec2(1,1));

    vec3 d = textureSample(uv, vec2(-1,0));
    vec3 e = textureSample(uv, vec2(0,0));
    vec3 f = textureSample(uv, vec2(1,0));

    vec3 g = textureSample(uv, vec2(-1,-1));
    vec3 h = textureSample(uv, vec2(0,-1));
    vec3 i = textureSample(uv, vec2(1,-1));

    // Apply weighted distribution, by using a 3x3 tent filter:
    //  1   | 1 2 1 |
    // -- * | 2 4 2 |
    // 16   | 1 2 1 |
    vec3 upsample = 1.0 / 16.0 * (
        4.0 * e +
        2.0 * (b + d + f + h) +
        1.0 * (a + c + g + i)
    );

    // blend color additively with previous color
    upsample += imageLoad(outputImage, texel_output).rgb;

    imageStore(outputImage, texel_output, vec4(upsample, 1.0));
}