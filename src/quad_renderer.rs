use crate::context::Context;
use crate::custom_storage_image::CustomStorageImage;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::layout::DescriptorSetLayout;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, StorageImage, SwapchainImage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::sync::GpuFuture;

pub struct QuadRenderer {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,

    sampler: Arc<Sampler>,
    descriptor_sets: Vec<Arc<PersistentDescriptorSet>>,
    index_buffer: Subbuffer<[u32]>,
    vertex_buffer: Subbuffer<[QuadVertex]>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl QuadRenderer {
    pub fn new(
        context: &Context,
        input_images: &[Arc<ImageView<CustomStorageImage>>],
        output_images: &[Arc<ImageView<SwapchainImage>>],
        final_output_format: Format,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> QuadRenderer {
        let (vertices, indices) = quad_mesh();

        let vertex_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            vertices,
        )
        .unwrap();

        let index_buffer = Buffer::from_iter(
            &memory_allocator,
            BufferCreateInfo {
                usage: BufferUsage::INDEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                usage: MemoryUsage::Upload,
                ..Default::default()
            },
            indices,
        )
        .unwrap();

        let render_pass = vulkano::single_pass_renderpass!(context.device(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: final_output_format,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        let pipeline = {
            let vertex_shader = vs::load(context.device()).unwrap();
            let fragment_shader = fs::load(context.device()).unwrap();

            GraphicsPipeline::start()
                .vertex_input_state(QuadVertex::per_vertex())
                .vertex_shader(vertex_shader.entry_point("main").unwrap(), ())
                .input_assembly_state(InputAssemblyState::new())
                .fragment_shader(fragment_shader.entry_point("main").unwrap(), ())
                .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(context.device())
                .unwrap()
        };

        let sampler = Sampler::new(
            context.device(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                address_mode: [SamplerAddressMode::Repeat; 3],
                mipmap_mode: SamplerMipmapMode::Nearest,
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        let set_layout = pipeline.layout().set_layouts().get(0).unwrap();

        let framebuffers = create_framebuffers(output_images, render_pass.clone());
        let descriptor_sets = create_descriptor_sets(
            set_layout,
            input_images,
            sampler.clone(),
            descriptor_set_allocator.clone(),
        );

        QuadRenderer {
            render_pass,
            pipeline,
            framebuffers,

            sampler,
            descriptor_sets,
            index_buffer,
            vertex_buffer,

            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn resize(
        &mut self,
        input_images: &[Arc<ImageView<CustomStorageImage>>],
        output_images: &[Arc<ImageView<SwapchainImage>>],
    ) {
        let set_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        self.framebuffers = create_framebuffers(output_images, self.render_pass.clone());
        self.descriptor_sets = create_descriptor_sets(
            set_layout,
            input_images,
            self.sampler.clone(),
            self.descriptor_set_allocator.clone(),
        );
    }

    pub fn render<F>(
        &self,
        context: &Context,
        future: F,
        swapchain_frame_index: u32,
        viewport: &Viewport,
    ) -> CommandBufferExecFuture<F>
    where
        F: GpuFuture + 'static,
    {
        let queue = context.queue();

        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            queue.queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        builder
            .set_viewport(0, [viewport.clone()])
            .begin_render_pass(
                RenderPassBeginInfo {
                    clear_values: vec![Some([0.0, 0.0, 0.0, 1.0].into())],
                    ..RenderPassBeginInfo::framebuffer(
                        self.framebuffers[swapchain_frame_index as usize].clone(),
                    )
                },
                SubpassContents::Inline,
            )
            .unwrap()
            .push_constants(
                self.pipeline.layout().clone(),
                0,
                fs::Data {
                    gamma: 2.2,
                    exposure: 1.0,
                },
            )
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.pipeline.layout().clone(),
                0,
                self.descriptor_sets[swapchain_frame_index as usize].clone(),
            )
            .bind_index_buffer(self.index_buffer.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw_indexed(6, 1, 0, 0, 0)
            .unwrap() // TODO: remove magic number 6
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        future
            .then_execute(context.queue(), command_buffer)
            .unwrap()
    }
}

fn create_descriptor_sets(
    set_layout: &Arc<DescriptorSetLayout>,
    input_images: &[Arc<ImageView<CustomStorageImage>>],
    sampler: Arc<Sampler>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
) -> Vec<Arc<PersistentDescriptorSet>> {
    input_images
        .iter()
        .map(|image| {
            PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                set_layout.clone(),
                [WriteDescriptorSet::image_view_sampler(
                    0,
                    image.clone(),
                    sampler.clone(),
                )],
            )
            .unwrap()
        })
        .collect()
}

fn create_framebuffers(
    output_images: &[Arc<ImageView<SwapchainImage>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    output_images
        .iter()
        .map(|image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![image.clone()],
                    ..FramebufferCreateInfo::default()
                },
            )
            .expect("failed to create framebuffer")
        })
        .collect()
}

#[repr(C)]
#[derive(Vertex, BufferContents)]
pub struct QuadVertex {
    #[format(R32G32_SFLOAT)]
    pub position: [f32; 2],

    #[format(R32G32_SFLOAT)]
    pub uv: [f32; 2],
}

const fn quad_mesh() -> ([QuadVertex; 4], [u32; 6]) {
    let vertices = [
        QuadVertex {
            position: [-1.0, -1.0],
            uv: [0.0, 0.0],
        },
        QuadVertex {
            position: [1.0, -1.0],
            uv: [1.0, 0.0],
        },
        QuadVertex {
            position: [1.0, 1.0],
            uv: [1.0, 1.0],
        },
        QuadVertex {
            position: [-1.0, 1.0],
            uv: [0.0, 1.0],
        },
    ];

    let indices = [0, 1, 2, 2, 3, 0];

    (vertices, indices)
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
            #version 450

            layout(location = 0) in vec2 position;
            layout(location = 1) in vec2 uv;

            layout(location = 0) out vec2 v_uv;

            layout(set = 0, binding = 0) uniform sampler2D image;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                v_uv = uv;
            }
        ",
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
            #version 450

            layout(location = 0) in vec2 v_uv;

            layout(location = 0) out vec4 f_color;

            layout(push_constant) uniform Data {
                float exposure;
                float gamma;
            } data;

            layout(set = 0, binding = 0) uniform sampler2D image;
            
            // Source: https://github.com/Shot511/RapidGL/blob/65d1202a5926acad9816483b141fb24480e81668/src/demos/22_pbr/tmo.frag

            vec3 gammaCorrect(vec3 color) 
            {
                return pow(color, vec3(1.0/data.gamma));
            }

            // sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
            mat3 ACESInputMat =
            {
                {0.59719, 0.07600, 0.02840},
                {0.35458, 0.90834, 0.13383},
                {0.04823, 0.01566, 0.83777}
            };
            
            // ODT_SAT => XYZ => D60_2_D65 => sRGB
            mat3 ACESOutputMat =
            {
                { 1.60475, -0.10208, -0.00327},
                {-0.53108,  1.10813, -0.07276},
                {-0.07367, -0.00605,  1.07602 }
            };
            
            vec3 RRTAndODTFit(vec3 v)
            {
                vec3 a = v * (v + 0.0245786f) - 0.000090537f;
                vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
                return a / b;
            }

            void main() {
                vec4 hdrColor = data.exposure * texture(image, v_uv);
                        
                vec3 color = ACESInputMat * hdrColor.rgb;
                     color = RRTAndODTFit(color);
                     color = ACESOutputMat * color;
            
                f_color = vec4(color, 1.0);
            }
        "
    }
}
