use crate::context::Context;
use std::sync::Arc;
use vulkano::buffer::{Buffer, BufferContents, BufferCreateInfo, BufferUsage, Subbuffer};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    CommandBufferBeginInfo, CommandBufferExecFuture, CommandBufferLevel, CommandBufferUsage,
    RecordingCommandBuffer, RenderPassBeginInfo, SubpassBeginInfo, SubpassContents,
};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{Image, ImageCreateInfo, ImageType, ImageUsage};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::graphics::color_blend::{ColorBlendAttachmentState, ColorBlendState};
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::multisample::MultisampleState;
use vulkano::pipeline::graphics::rasterization::RasterizationState;
use vulkano::pipeline::graphics::vertex_input::{Vertex, VertexDefinition};
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::graphics::GraphicsPipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    DynamicState, GraphicsPipeline, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::sync::GpuFuture;

pub struct SceneRenderer {
    render_pass: Arc<RenderPass>,
    pipeline: Arc<GraphicsPipeline>,
    framebuffers: Vec<Arc<Framebuffer>>,
    output_images: Vec<Arc<ImageView>>,
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

        let render_pass = vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    format: output_format,
                    samples: 1,
                    load_op: Clear,
                    store_op: Store,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )
        .unwrap();

        let pipeline = {
            let vs = vs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let fs = fs::load(device.clone())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let vertex_input_state = SceneVertex::per_vertex().definition(&vs).unwrap();

            let stages = [vs, fs].map(PipelineShaderStageCreateInfo::new);

            let layout = PipelineLayout::new(
                device.clone(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages(&stages)
                    .into_pipeline_layout_create_info(device.clone())
                    .unwrap(),
            )
            .unwrap();

            let subpass = Subpass::from(render_pass.clone(), 0).unwrap();

            GraphicsPipeline::new(
                device.clone(),
                None,
                GraphicsPipelineCreateInfo {
                    stages: stages.into_iter().collect(),
                    vertex_input_state: Some(vertex_input_state),
                    input_assembly_state: Some(InputAssemblyState::default()),
                    viewport_state: Some(ViewportState::default()),
                    rasterization_state: Some(RasterizationState::default()),
                    multisample_state: Some(MultisampleState::default()),
                    color_blend_state: Some(ColorBlendState::with_attachment_states(
                        subpass.num_color_attachments(),
                        ColorBlendAttachmentState::default(),
                    )),
                    dynamic_state: [DynamicState::Viewport].into_iter().collect(),
                    subpass: Some(subpass.into()),
                    ..GraphicsPipelineCreateInfo::layout(layout)
                },
            )
            .unwrap()
        };

        let vertex_buffer = Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::VERTEX_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
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

        let mut builder = RecordingCommandBuffer::new(
            self.command_buffer_allocator.clone(),
            queue.queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
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
                SubpassBeginInfo {
                    contents: SubpassContents::Inline,
                    ..Default::default()
                },
            )
            .unwrap()
            .set_viewport(0, [viewport.clone()].into_iter().collect())
            .unwrap()
            .bind_pipeline_graphics(self.pipeline.clone())
            .unwrap()
            .bind_vertex_buffers(0, self.vertex_buffer.clone())
            .unwrap();

        unsafe {
            builder
                .draw(self.vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap();
        }

        builder.end_render_pass(Default::default()).unwrap();

        let command_buffer = builder.end().unwrap();

        future
            .then_execute(context.queue(), command_buffer)
            .unwrap()
    }

    pub fn output_images(&self) -> &Vec<Arc<ImageView>> {
        &self.output_images
    }
}

fn create_output_images(
    count: u32,
    dimension: [u32; 2],
    format: Format,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView>> {
    (0..count)
        .map(|_| {
            ImageView::new_default(
                Image::new(
                    memory_allocator.clone(),
                    ImageCreateInfo {
                        image_type: ImageType::Dim2d,
                        format,
                        extent: [dimension[0], dimension[1], 1],
                        usage: ImageUsage::SAMPLED
                            | ImageUsage::COLOR_ATTACHMENT
                            | ImageUsage::TRANSFER_SRC
                            | ImageUsage::STORAGE,
                        ..Default::default()
                    },
                    AllocationCreateInfo::default(),
                )
                .unwrap(),
            )
            .unwrap()
        })
        .collect()
}

fn create_framebuffers(
    output_images: &[Arc<ImageView>],
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
