use crate::context::Context;
use crate::custom_storage_image::CustomStorageImage;
use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage, CopyImageInfo,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::image::view::{ImageView, ImageViewCreateInfo};
use vulkano::image::{
    AttachmentImage, ImageAccess, ImageCreateFlags, ImageSubresourceRange, ImageUsage,
    ImageViewAbstract,
};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::padded::Padded;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sampler::{Filter, Sampler, SamplerAddressMode, SamplerCreateInfo, SamplerMipmapMode};
use vulkano::sync::GpuFuture;

pub struct BloomRenderer {
    downsample_pipeline: Arc<ComputePipeline>,
    upsample_pipeline: Arc<ComputePipeline>,

    sampler: Arc<Sampler>,

    input_images: Vec<Arc<ImageView<AttachmentImage>>>,
    output_images: Vec<Arc<ImageView<CustomStorageImage>>>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl BloomRenderer {
    pub fn new(
        context: &Context,
        input_images: Vec<Arc<ImageView<AttachmentImage>>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> BloomRenderer {
        let downsample_pipeline = {
            let shader = cs::downsample::load(context.device()).unwrap();

            ComputePipeline::new(
                context.device(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

        let upsample_pipeline = {
            let shader = cs::upsample::load(context.device()).unwrap();

            ComputePipeline::new(
                context.device(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
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

    pub fn resize(&mut self, input_images: Vec<Arc<ImageView<AttachmentImage>>>) {
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
        let mut builder = AutoCommandBufferBuilder::primary(
            &self.command_buffer_allocator,
            context.queue().queue_family_index(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();

        let scene_image = self.input_images[swapchain_frame_index as usize]
            .image()
            .clone();
        let work_image = self.output_images[swapchain_frame_index as usize]
            .image()
            .clone();

        let input_image_view = ImageView::new_default(work_image.clone()).unwrap();

        // copy scene image to work image
        builder
            .copy_image(CopyImageInfo::images(
                scene_image.clone(),
                work_image.clone(),
            ))
            .unwrap();

        // downsample passes
        builder.bind_pipeline_compute(self.downsample_pipeline.clone());
        {
            let downsample_set_layout = self
                .downsample_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap();

            let output_image_view = ImageView::new(
                work_image.clone(),
                ImageViewCreateInfo {
                    format: Some(work_image.format()),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: 1..2,
                        ..work_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();

            let downsample_descriptor_set = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                downsample_set_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        input_image_view.clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, output_image_view.clone()),
                ],
            )
            .unwrap();

            let downsample_pass = cs::downsample::Pass {
                mipLevel: Padded::from(0),
                texelSize: work_image
                    .dimensions()
                    .width_height()
                    .map(|v| 1.0 / v as f32)
                    .into(),
            };

            builder
                .push_constants(
                    self.downsample_pipeline.layout().clone(),
                    0,
                    downsample_pass,
                )
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.downsample_pipeline.layout().clone(),
                    0,
                    downsample_descriptor_set.clone(),
                )
                .dispatch(work_image.dimensions().width_height_depth().map(|v| v / 2))
                .unwrap();
        }

        // upsample passes

        builder.bind_pipeline_compute(self.upsample_pipeline.clone());
        {
            let upsample_set_layout = self
                .upsample_pipeline
                .layout()
                .set_layouts()
                .get(0)
                .unwrap();

            let output_image_view = ImageView::new(
                work_image.clone(),
                ImageViewCreateInfo {
                    format: Some(work_image.format()),
                    subresource_range: ImageSubresourceRange {
                        mip_levels: 0..1,
                        ..work_image.subresource_range()
                    },
                    ..ImageViewCreateInfo::default()
                },
            )
            .unwrap();

            let upsample_descriptor_set = PersistentDescriptorSet::new(
                &self.descriptor_set_allocator,
                upsample_set_layout.clone(),
                [
                    WriteDescriptorSet::image_view_sampler(
                        0,
                        input_image_view.clone(),
                        self.sampler.clone(),
                    ),
                    WriteDescriptorSet::image_view(1, output_image_view.clone()),
                ],
            )
            .unwrap();

            let upsample_pass = cs::upsample::Pass {
                mipLevel: Padded::from(1),
                texelSize: work_image
                    .dimensions()
                    .width_height()
                    .map(|v| 1.0 / v as f32 * 0.5)
                    .into(),
            };

            builder
                .push_constants(self.upsample_pipeline.layout().clone(), 0, upsample_pass)
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.upsample_pipeline.layout().clone(),
                    0,
                    upsample_descriptor_set.clone(),
                )
                .dispatch(work_image.dimensions().width_height_depth())
                .unwrap();
        }

        let command_buffer = builder.build().unwrap();

        future
            .then_execute(context.queue(), command_buffer)
            .unwrap()
    }

    pub fn output_images(&self) -> &Vec<Arc<ImageView<CustomStorageImage>>> {
        &self.output_images
    }
}

fn create_output_images(
    input_images: Vec<Arc<ImageView<AttachmentImage>>>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView<CustomStorageImage>>> {
    input_images
        .iter()
        .map(|image| {
            let image = CustomStorageImage::uninitialized(
                &memory_allocator,
                image.dimensions().width_height(),
                image.image().format(),
                2,
                ImageUsage::TRANSFER_DST | ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ImageCreateFlags::empty(),
            )
            .unwrap();

            ImageView::new_default(image).unwrap()
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