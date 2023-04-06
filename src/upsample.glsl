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

void main() {
    ivec2 texel_output = ivec2(gl_GlobalInvocationID.xy);
    vec2 texel_input = vec2(texel_output) / 2.0;

    vec2 uv = texel_input * pass.texelSize;

    // one sample for now
    vec3 e = textureLod(inputImage, uv, float(pass.mipLevel)).rgb;

    vec3 upsample = e;

    // blend color additively with previous color
    upsample += imageLoad(outputImage, texel_output).rgb;

    imageStore(outputImage, texel_output, vec4(upsample, 1.0));
}