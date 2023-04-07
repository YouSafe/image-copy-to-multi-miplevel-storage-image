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

vec3 textureSample(vec2 uv, vec2 offset) {
    return textureLod(inputImage, uv + offset * pass.texelSize, float(pass.mipLevel)).rgb;
}

void main() {
    ivec2 texel_output = ivec2(gl_GlobalInvocationID.xy);
    ivec2 texel_input = 2 * texel_output;

    // uv coordinates between 0-1
    vec2 uv = vec2(texel_input) * pass.texelSize;

    // Source: https://learnopengl.com/Guest-Articles/2022/Phys.-Based-Bloom
    // Take 13 samples around current texel:
    // a - b - c
    // - j - k -
    // d - e - f
    // - l - m -
    // g - h - i
    // === ('e' is the current texel) ===

    vec3 a = textureSample(uv, vec2(-2, 2));
    vec3 b = textureSample(uv, vec2( 0, 2));
    vec3 c = textureSample(uv, vec2( 2, 2));

    vec3 d = textureSample(uv, vec2(-2, 0));
    vec3 e = textureSample(uv, vec2( 0, 0));
    vec3 f = textureSample(uv, vec2( 2, 0));

    vec3 g = textureSample(uv, vec2(-2,-2));
    vec3 h = textureSample(uv, vec2( 0,-2));
    vec3 i = textureSample(uv, vec2( 2,-2));

    vec3 j = textureSample(uv, vec2(-1, 1));
    vec3 k = textureSample(uv, vec2( 1, 1));
    vec3 l = textureSample(uv, vec2(-1,-1));
    vec3 m = textureSample(uv, vec2( 1,-1));

    // code assumes that the coodinate system has its origin in the bottom left corner

    // Apply weighted distribution:
    // 0.5 + 0.125 + 0.125 + 0.125 + 0.125 = 1

    // a,b,d,e * 0.125
    // b,c,e,f * 0.125
    // d,e,g,h * 0.125
    // e,f,h,i * 0.125
    // j,k,l,m * 0.5

    // This shows 5 square areas that are being sampled. But some of them overlap,
    // so to have an energy preserving downsample we need to make some adjustments.
    // The weights are the distributed, so that the sum of j,k,l,m (e.g.)
    // contribute 0.5 to the final color output. The code below is written
    // to effectively yield this sum. We get:
    // 0.125*5 + 0.03125*4 + 0.0625*4 = 1

    vec3 downsample = e * 0.125
        + (a+c+g+i) * 0.03125 // ((a+c+g+i) / 4) * 0.125
        + (b+d+f+h) * 0.0625  // (b+d+f+h) / 2) * 0.125
        + (j+k+l+m) * 0.125;  // (j+k+l+m) * 0.125

    imageStore(outputImage, texel_output, vec4(downsample, 1.0));
}