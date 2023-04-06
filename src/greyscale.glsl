#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(binding = 0, rgba16f) readonly uniform image2D inputImage;
layout(binding = 1, rgba16f) writeonly uniform image2D outputImage;

void main() {
    ivec2 texel = ivec2(gl_GlobalInvocationID.xy);
    vec4 color = imageLoad(inputImage, texel);

    // Convert to grayscale
    float gray = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    vec4 grayscaleColor = vec4(vec3(gray), color.a);

    imageStore(outputImage, texel, grayscaleColor);
}