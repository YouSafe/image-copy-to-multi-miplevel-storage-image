#version 450

layout(push_constant) uniform Pass {
    // mip level of the input image
    int mipLevel;
    // size of one texel in the input image
    vec2 texelSize;
} pass;

layout(set = 0, binding = 0) uniform sampler2D inputImage;
layout(set = 0, binding = 1, rgba16f) writeonly uniform image2D outputImage;

// one invokation for every texel in the output image
// output image has **HALF** the dimensions of the input image
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
    ivec2 texel_output = ivec2(gl_GlobalInvocationID.xy);
    ivec2 texel_input = 2 * texel_output;

    // uv coordinates between 0-1
    vec2 uv = vec2(texel_input) * pass.texelSize;

    // one sample for now
    vec3 e = textureLod(inputImage, uv, float(pass.mipLevel)).rgb;

    vec3 downsample = e;

    // this should change the result image to be more green
    downsample = vec3(0.0,1.0,0.0);

    imageStore(outputImage, texel_output, vec4(downsample, 1.0));
}