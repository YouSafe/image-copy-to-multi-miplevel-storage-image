mod bloom_renderer;
mod context;
mod quad_renderer;
mod scene_renderer;

use crate::bloom_renderer::BloomRenderer;
use crate::context::Context;
use crate::quad_renderer::QuadRenderer;
use crate::scene_renderer::SceneRenderer;
use std::sync::Arc;
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::swapchain::{ColorSpace, SurfaceInfo};
use vulkano::{
    command_buffer::allocator::StandardCommandBufferAllocator,
    image::ImageUsage,
    memory::allocator::StandardMemoryAllocator,
    pipeline::graphics::viewport::Viewport,
    swapchain::{acquire_next_image, Swapchain, SwapchainCreateInfo, SwapchainPresentInfo},
    sync::{self, GpuFuture},
};
use vulkano::{Validated, VulkanError};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
};

fn main() {
    let event_loop = EventLoop::new().unwrap();
    let window_builder = WindowBuilder::new();

    let window = Arc::new(window_builder.build(&event_loop).unwrap());

    let context = Context::new(window.clone(), &event_loop);

    let surface = context.surface();
    let device = context.device();
    let physical_device = context.physical_device();
    let queue = context.queue();

    // Some little debug info.
    println!(
        "Using device: {} (type: {:?})",
        physical_device.properties().device_name,
        physical_device.properties().device_type,
    );

    let (mut swapchain, images) = {
        let surface_capabilities = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        // Choosing the internal format that the images will have.
        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, SurfaceInfo::default())
                .expect("could not fetch surface formats")
                .iter()
                .min_by_key(|(format, color)| {
                    // Prefer a srgb format
                    match (format, color) {
                        (Format::B8G8R8A8_SRGB, _) => 1,
                        (Format::R8G8B8A8_SRGB, ColorSpace::SrgbNonLinear) => 2,
                        (_, _) => 3,
                    }
                })
                .expect("could not fetch image format")
                .0, // just the format
        )
        .unwrap();

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: surface_capabilities.min_image_count,
                image_format,
                image_extent: window.inner_size().into(),
                image_usage: ImageUsage::COLOR_ATTACHMENT,
                composite_alpha: surface_capabilities
                    .supported_composite_alpha
                    .into_iter()
                    .next()
                    .unwrap(),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let swapchain_image_views: Vec<Arc<ImageView>> = images
        .iter()
        .map(|image| ImageView::new_default(image.clone()).unwrap())
        .collect();

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        Default::default(),
    ));

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        context.device(),
        Default::default(),
    ));

    // Dynamic viewports allow us to recreate just the viewport when the window is resized
    // Otherwise we would have to recreate the whole pipeline.
    let mut viewport = Viewport {
        offset: [0.0, 0.0],
        extent: [0.0, 0.0],
        depth_range: 0.0..=1.0,
    };

    let mut scene_renderer = SceneRenderer::new(
        &context,
        swapchain.image_count(),
        swapchain.image_extent(),
        Format::R16G16B16A16_SFLOAT,
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
    );

    let mut bloom_renderer = BloomRenderer::new(
        &context,
        scene_renderer.output_images().clone(),
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    let mut quad_renderer = QuadRenderer::new(
        &context,
        bloom_renderer.output_images(),
        &swapchain_image_views,
        swapchain.image_format(),
        memory_allocator.clone(),
        command_buffer_allocator.clone(),
        descriptor_set_allocator.clone(),
    );

    let mut recreate_swapchain = false;
    let mut previous_frame_end = Some(sync::now(device.clone()).boxed());

    event_loop.set_control_flow(ControlFlow::Poll);

    event_loop
        .run(move |event, elwt| {
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    elwt.exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_),
                    ..
                } => {
                    recreate_swapchain = true;
                }
                Event::WindowEvent {
                    event: WindowEvent::RedrawRequested,
                    ..
                } => {
                    let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                    let dimensions = window.inner_size();
                    if dimensions.width == 0 || dimensions.height == 0 {
                        return;
                    }

                    // It is important to call this function from time to time, otherwise resources will keep
                    // accumulating and you will eventually reach an out of memory error.
                    // Calling this function polls various fences in order to determine what the GPU has
                    // already processed, and frees the resources that are no longer needed.
                    previous_frame_end.as_mut().unwrap().cleanup_finished();

                    if recreate_swapchain {
                        let (new_swapchain, new_images) = swapchain
                            .recreate(SwapchainCreateInfo {
                                image_extent: dimensions.into(),
                                ..swapchain.create_info()
                            })
                            .unwrap();

                        let image_extent: [u32; 2] = window.inner_size().into();
                        viewport.extent = [image_extent[0] as f32, image_extent[1] as f32];

                        let new_swapchain_image_views: Vec<Arc<ImageView>> = new_images
                            .iter()
                            .map(|image| ImageView::new_default(image.clone()).unwrap())
                            .collect();

                        swapchain = new_swapchain;
                        scene_renderer.resize(swapchain.image_count(), swapchain.image_extent());
                        bloom_renderer.resize(scene_renderer.output_images().clone());
                        quad_renderer
                            .resize(bloom_renderer.output_images(), &new_swapchain_image_views);

                        recreate_swapchain = false;
                    }

                    let (image_index, suboptimal, acquire_future) = match acquire_next_image(
                        swapchain.clone(),
                        None,
                    )
                    .map_err(Validated::unwrap)
                    {
                        Ok(r) => r,
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            return;
                        }
                        Err(e) => panic!("Failed to acquire next image: {:?}", e),
                    };

                    // acquire_next_image can be successful, but suboptimal. This means that the swapchain image
                    // will still work, but it may not display correctly. With some drivers this can be when
                    // the window resizes, but it may not cause the swapchain to become out of date.
                    if suboptimal {
                        recreate_swapchain = true;
                    }

                    let future = previous_frame_end.take().unwrap().join(acquire_future);

                    let future = scene_renderer.render(&context, future, image_index, &viewport);
                    let future = bloom_renderer.compute(&context, future, image_index);
                    let future = quad_renderer.render(&context, future, image_index, &viewport);

                    let future = future
                        .then_swapchain_present(
                            queue.clone(),
                            SwapchainPresentInfo::swapchain_image_index(
                                swapchain.clone(),
                                image_index,
                            ),
                        )
                        .then_signal_fence_and_flush();

                    match future.map_err(Validated::unwrap) {
                        Ok(future) => {
                            previous_frame_end = Some(future.boxed());
                        }
                        Err(VulkanError::OutOfDate) => {
                            recreate_swapchain = true;
                            previous_frame_end = Some(sync::now(device.clone()).boxed());
                        }
                        Err(e) => {
                            panic!("Failed to flush future: {:?}", e);
                            // previous_frame_end = Some(sync::now(device.clone()).boxed());
                        }
                    }
                }
                _ => (),
            }
        })
        .unwrap();
}
