use crate::context::Context;
use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    CommandBufferBeginInfo, CommandBufferExecFuture, CommandBufferLevel, CommandBufferUsage,
    CopyImageInfo, RecordingCommandBuffer,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{DescriptorSet, WriteDescriptorSet};
use vulkano::image::sampler::{
    Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode,
};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{
    mip_level_extent, Image, ImageCreateInfo, ImageSubresourceRange, ImageType, ImageUsage,
};
use vulkano::memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator};
use vulkano::pipeline::compute::ComputePipelineCreateInfo;
use vulkano::pipeline::layout::PipelineDescriptorSetLayoutCreateInfo;
use vulkano::pipeline::{
    ComputePipeline, Pipeline, PipelineBindPoint, PipelineLayout, PipelineShaderStageCreateInfo,
};
use vulkano::sync::GpuFuture;

pub struct BloomRenderer {
    downsample_pipeline: Arc<ComputePipeline>,
    upsample_pipeline: Arc<ComputePipeline>,

    sampler: Arc<Sampler>,

    input_images: Vec<Arc<ImageView>>,
    output_images: Vec<Arc<ImageView>>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl BloomRenderer {
    pub fn new(
        context: &Context,
        input_images: Vec<Arc<ImageView>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> BloomRenderer {
        let downsample_pipeline = {
            let shader = cs::downsample::load(context.device())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(shader);
            let layout = PipelineLayout::new(
                context.device(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(context.device())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                context.device(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let upsample_pipeline = {
            let shader = cs::upsample::load(context.device())
                .unwrap()
                .entry_point("main")
                .unwrap();

            let stage = PipelineShaderStageCreateInfo::new(shader);
            let layout = PipelineLayout::new(
                context.device(),
                PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
                    .into_pipeline_layout_create_info(context.device())
                    .unwrap(),
            )
            .unwrap();

            ComputePipeline::new(
                context.device(),
                None,
                ComputePipelineCreateInfo::stage_layout(stage, layout),
            )
            .unwrap()
        };

        let sampler = Sampler::new(
            context.device(),
            SamplerCreateInfo {
                mag_filter: Filter::Linear,
                min_filter: Filter::Linear,
                mipmap_mode: SamplerMipmapMode::Nearest,
                address_mode: [SamplerAddressMode::ClampToEdge; 3],
                ..SamplerCreateInfo::default()
            },
        )
        .unwrap();

        let output_images = create_output_images(input_images.clone(), memory_allocator.clone());

        BloomRenderer {
            downsample_pipeline,
            upsample_pipeline,
            sampler,

            input_images,
            output_images,

            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn resize(&mut self, input_images: Vec<Arc<ImageView>>) {
        self.input_images = input_images.clone();
        self.output_images = create_output_images(input_images, self.memory_allocator.clone())
    }

    pub fn compute<F>(
        &self,
        context: &Context,
        future: F,
        swapchain_frame_index: u32,
    ) -> CommandBufferExecFuture<F>
    where
        F: GpuFuture + 'static,
    {
        let mut builder = RecordingCommandBuffer::new(
            self.command_buffer_allocator.clone(),
            context.queue().queue_family_index(),
            CommandBufferLevel::Primary,
            CommandBufferBeginInfo {
                usage: CommandBufferUsage::OneTimeSubmit,
                ..Default::default()
            },
        )
        .unwrap();

        let scene_image = self.input_images[swapchain_frame_index as usize]
            .image()
            .clone();
        let work_image = self.output_images[swapchain_frame_index as usize]
            .image()
            .clone();

        // copy scene image to work image
        builder
            .copy_image(CopyImageInfo::images(
                scene_image.clone(),
                work_image.clone(),
            ))
            .unwrap();

        // downsample passes
        builder
            .bind_pipeline_compute(self.downsample_pipeline.clone())
            .unwrap();

        let downsample_set_layout = self
            .downsample_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        for i in 0..(work_image.mip_levels() - 1) {
            let input_miplevel = i;
            let output_miplevel = i + 1;

            let input_size = mip_level_extent(work_image.extent(), input_miplevel).unwrap();

            let output_size = mip_level_extent(work_image.extent(), output_miplevel).unwrap();

            let output_image_view = ImageView::new(
                work_image.clone(),
                ImageViewCreateInfo {
                    format: work_image.format(),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: (output_miplevel)..(output_miplevel + 1), // mip level of output image
                        ..work_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();

            let input_image_view = ImageView::new(
                work_image.clone(),
                ImageViewCreateInfo {
                    format: work_image.format(),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: (input_miplevel)..(input_miplevel + 1), // mip level of output image
                        ..work_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();

            let downsample_descriptor_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                downsample_set_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        input_image_view.clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, output_image_view.clone()),
                ],
                [],
            )
            .unwrap();

            let downsample_pass = cs::downsample::Pass {
                texelSize: [1.0 / input_size[0] as f32, 1.0 / input_size[1] as f32],
                useThreshold: (input_miplevel == 0) as u32,
                threshold: 1.5,
                knee: 0.1,
            };

            builder
                .push_constants(
                    self.downsample_pipeline.layout().clone(),
                    0,
                    downsample_pass,
                )
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.downsample_pipeline.layout().clone(),
                    0,
                    downsample_descriptor_set.clone(),
                )
                .unwrap();

            unsafe {
                // TODO: check if valid
                builder.dispatch(output_size).unwrap();
            }
        }

        // upsample passes

        builder
            .bind_pipeline_compute(self.upsample_pipeline.clone())
            .unwrap();

        let upsample_set_layout = self
            .upsample_pipeline
            .layout()
            .set_layouts()
            .get(0)
            .unwrap();

        for i in (0..(work_image.mip_levels() - 1)).rev() {
            let input_miplevel = i + 1;
            let output_miplevel = i;

            let input_size = mip_level_extent(work_image.extent(), input_miplevel).unwrap();

            let output_size = mip_level_extent(work_image.extent(), output_miplevel).unwrap();

            let output_image_view = ImageView::new(
                work_image.clone(),
                ImageViewCreateInfo {
                    format: work_image.format(),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: (output_miplevel)..(output_miplevel + 1), // mip level of output image
                        ..work_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();

            let input_image_view = ImageView::new(
                work_image.clone(),
                ImageViewCreateInfo {
                    format: work_image.format(),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: (input_miplevel)..(input_miplevel + 1), // mip level of output image
                        ..work_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();

            let upsample_descriptor_set = DescriptorSet::new(
                self.descriptor_set_allocator.clone(),
                upsample_set_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        input_image_view.clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, output_image_view.clone()),
                ],
                [],
            )
            .unwrap();

            let upsample_pass = cs::upsample::Pass {
                texelSize: [1.0 / input_size[0] as f32, 1.0 / input_size[1] as f32],

                intensity: 1.0,
            };

            builder
                .push_constants(self.upsample_pipeline.layout().clone(), 0, upsample_pass)
                .unwrap()
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.upsample_pipeline.layout().clone(),
                    0,
                    upsample_descriptor_set.clone(),
                )
                .unwrap();

            unsafe {
                // TODO: check if valid
                builder.dispatch(output_size).unwrap();
            }
        }
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
    input_images: Vec<Arc<ImageView>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView>> {
    input_images
        .iter()
        .map(|image| {
            let storage_image = Image::new(
                memory_allocator.clone(),
                ImageCreateInfo {
                    image_type: ImageType::Dim2d,
                    format: image.format(),
                    extent: image.image().extent(),
                    mip_levels: 6,
                    usage: ImageUsage::TRANSFER_DST | ImageUsage::STORAGE | ImageUsage::SAMPLED,
                    ..ImageCreateInfo::default()
                },
                AllocationCreateInfo {
                    memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                        | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                    ..Default::default()
                },
            )
            .unwrap();

            let view = ImageView::new(
                storage_image.clone(),
                ImageViewCreateInfo {
                    format: storage_image.format(),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: 0..1,
                        ..storage_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();
            view
        })
        .collect()
}

mod cs {
    pub mod downsample {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/downsample.glsl"
        }
    }
    pub mod upsample {
        vulkano_shaders::shader! {
            ty: "compute",
            path: "src/upsample.glsl"
        }
    }
}
