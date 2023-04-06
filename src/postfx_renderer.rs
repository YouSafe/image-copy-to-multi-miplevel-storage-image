use crate::context::Context;
use crate::custom_storage_image::CustomStorageImage;
use std::sync::Arc;
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferExecFuture, CommandBufferUsage,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::Queue;
use vulkano::image::view::ImageView;
use vulkano::image::{
    AttachmentImage, ImageAccess, ImageCreateFlags, ImageUsage, ImageViewAbstract, StorageImage,
};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineBindPoint};
use vulkano::sync::GpuFuture;

pub struct PostFXRenderer {
    pipeline: Arc<ComputePipeline>,

    input_images: Vec<Arc<ImageView<AttachmentImage>>>,
    output_images: Vec<Arc<ImageView<CustomStorageImage>>>,

    memory_allocator: Arc<StandardMemoryAllocator>,
    command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
    descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
}

impl PostFXRenderer {
    pub fn new(
        context: &Context,
        input_images: Vec<Arc<ImageView<AttachmentImage>>>,
        memory_allocator: Arc<StandardMemoryAllocator>,
        command_buffer_allocator: Arc<StandardCommandBufferAllocator>,
        descriptor_set_allocator: Arc<StandardDescriptorSetAllocator>,
    ) -> PostFXRenderer {
        let pipeline = {
            let shader = cs::load(context.device()).unwrap();

            ComputePipeline::new(
                context.device(),
                shader.entry_point("main").unwrap(),
                &(),
                None,
                |_| {},
            )
            .unwrap()
        };

        let output_images = create_output_images(
            input_images.clone(),
            context.queue(),
            memory_allocator.clone(),
        );

        PostFXRenderer {
            pipeline,

            input_images,
            output_images,

            memory_allocator,
            command_buffer_allocator,
            descriptor_set_allocator,
        }
    }

    pub fn resize(
        &mut self,
        input_images: Vec<Arc<ImageView<AttachmentImage>>>,
        queue: Arc<Queue>,
    ) {
        self.input_images = input_images.clone();
        self.output_images =
            create_output_images(input_images, queue, self.memory_allocator.clone())
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

        let input_image = self.input_images[swapchain_frame_index as usize].clone();

        let output_image = self.output_images[swapchain_frame_index as usize].clone();

        let set_layout = self.pipeline.layout().set_layouts().get(0).unwrap();

        let descriptor_set = PersistentDescriptorSet::new(
            &self.descriptor_set_allocator,
            set_layout.clone(),
            [
                WriteDescriptorSet::image_view(0, input_image.clone()),
                WriteDescriptorSet::image_view(1, output_image.clone()),
            ],
        )
        .unwrap();

        builder
            // .copy_image(CopyImageInfo::images(
            //     input_image.image().clone(),
            //     output_image.image().clone(),
            // ))
            // .unwrap()
            .bind_pipeline_compute(self.pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Compute,
                self.pipeline.layout().clone(),
                0,
                descriptor_set.clone(),
            )
            .dispatch(output_image.dimensions().width_height_depth())
            .unwrap();

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
    queue: Arc<Queue>,
    memory_allocator: Arc<StandardMemoryAllocator>,
) -> Vec<Arc<ImageView<CustomStorageImage>>> {
    input_images
        .iter()
        .map(|image| {
            let image = CustomStorageImage::uninitialized(
                &memory_allocator,
                image.dimensions().width_height(),
                image.image().format(),
                6,
                ImageUsage::TRANSFER_DST | ImageUsage::STORAGE | ImageUsage::SAMPLED,
                ImageCreateFlags::empty(),
            )
            .unwrap();

            ImageView::new_default(image).unwrap()
        })
        .collect()
}

mod cs {
    vulkano_shaders::shader! {
        ty: "compute",
        path: "src/greyscale.glsl"
    }
}
