use crate::context::Context;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, RenderPassBeginInfo,
    SubpassContents,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryUsage, StandardMemoryAllocator};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::vertex_input::Vertex;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sync::GpuFuture;

pub struct SceneRenderer {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    output_images: Vec<Arc<ImageView<AttachmentImage>>>,
    output_format: Format,

    vertex_buffer: Subbuffer<[SceneVertex]>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
}

impl SceneRenderer {
    pub fn new(
        context: &Context,
        image_count: u32,
        image_dimensions: [u32; 2],
        output_format: Format,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    ) -> SceneRenderer {
        let device = context.device();

        let vs = vs::load(device.clone()).unwrap();
        let fs = fs::load(device.clone()).unwrap();

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: output_format,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        let pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_state(SceneVertex::per_vertex())
            .input_assembly_state(InputAssemblyState::new())
            .vertex_shader(vs.entry_point("main").unwrap(), ())
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(device.clone())
            .unwrap();

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
            triangle_vertices(),
        )
        .unwrap();

        let output_images = create_output_images(
            image_count,
            image_dimensions,
            output_format,
            memory_allocator.clone(),
        );
        let framebuffers = create_framebuffers(&output_images, render_pass.clone());

        SceneRenderer {
            render_pass,
            pipeline,
            framebuffers,
            output_images,
            output_format,

            vertex_buffer,

            memory_allocator,
            command_buffer_allocator,
        }
    }

    pub fn resize(&mut self, image_count: u32, image_dimensions: [u32; 2]) {
        self.output_images = create_output_images(
            image_count,
            image_dimensions,
            self.output_format,
            self.memory_allocator.clone(),
        );
        self.framebuffers = create_framebuffers(&self.output_images, self.render_pass.clone());
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
            .set_viewport(0, [viewport.clone()])
            .bind_pipeline_graphics(self.pipeline.clone())
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap()
            .end_render_pass()
            .unwrap();

        let command_buffer = builder.build().unwrap();

        future
            .then_execute(context.queue(), command_buffer)
            .unwrap()
    }

    pub fn output_images(&self) -> &Vec<Arc<ImageView<AttachmentImage>>> {
        &self.output_images
    }
}

fn create_output_images(
    count: u32,
    dimension: [u32; 2],
    format: Format,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView<AttachmentImage>>> {
    (0..count)
        .map(|_| {
            ImageView::new_default(
                AttachmentImage::with_usage(
                    &memory_allocator,
                    dimension,
                    format,
                    ImageUsage::SAMPLED
                        | ImageUsage::COLOR_ATTACHMENT
                        | ImageUsage::TRANSFER_SRC
                        | ImageUsage::STORAGE,
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect()
}

fn create_framebuffers(
    output_images: &[Arc<ImageView<AttachmentImage>>],
    render_pass: Arc<RenderPass>,
) -> Vec<Arc<Framebuffer>> {
    output_images
        .iter()
        .map(|image| {
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![image.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}

#[repr(C)]
#[derive(Vertex, BufferContents)]
struct SceneVertex {
    #[format(R32G32_SFLOAT)]
    position: [f32; 2],
}

const fn triangle_vertices() -> [SceneVertex; 3] {
    [
        SceneVertex {
            position: [-0.5, 0.5],
        },
        SceneVertex {
            position: [0.5, 0.5],
        },
        SceneVertex {
            position: [0.0, -0.5],
        },
    ]
}

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
				#version 450

				layout(location = 0) in vec2 position;

				void main() {
					gl_Position = vec4(position, 0.0, 1.0);
				}
			"
    }
}

mod fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        src: "
				#version 450

				layout(location = 0) out vec4 f_color;

				void main() {
					f_color = vec4(1.8, 0.0, 0.0, 1.0);
				}
			"
    }
}
