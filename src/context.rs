use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::swapchain::Surface;
use vulkano::{Version, VulkanLibrary};
use winit::event_loop::EventLoop;
use winit::window::Window;

pub struct Context {
    surface: Arc<Surface>,
    device: Arc<Device>,
    physical_device: Arc<PhysicalDevice>,
    queue: Arc<Queue>,
}

impl Context {
    pub fn new(window: Arc<Window>, event_loop: &EventLoop<()>) -> Context {
        let instance = create_instance(event_loop);

        let surface = Surface::from_window(instance.clone(), window.clone()).unwrap();

        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };

        let (physical_device, queue_family_index) =
            find_physical_device(instance.clone(), surface.clone(), &device_extensions);

        let (device, queue) = create_logical_device(
            physical_device.clone(),
            queue_family_index,
            &device_extensions,
        );

        Context {
            surface,
            device,
            physical_device,
            queue,
        }
    }

    pub fn surface(&self) -> Arc<Surface> {
        self.surface.clone()
    }

    pub fn physical_device(&self) -> Arc<PhysicalDevice> {
        self.physical_device.clone()
    }

    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    pub fn queue(&self) -> Arc<Queue> {
        self.queue.clone()
    }
}

fn create_instance(event_loop: &EventLoop<()>) -> Arc<Instance> {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

    let required_extensions = Surface::required_extensions(&event_loop).unwrap();

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            max_api_version: Some(Version::major_minor(1, 2)),
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    instance
}

fn find_physical_device(
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    device_extensions: &DeviceExtensions,
) -> (Arc<PhysicalDevice>, u32) {
    instance
        .enumerate_physical_devices()
        .expect("could not enumerate physical devices")
        .filter(|p| {
            // check if device extensions are supported
            p.supported_extensions().contains(&device_extensions)
        })
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // check for graphics flag in queue family
                    q.queue_flags.intersects(QueueFlags::GRAPHICS)
                        && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // prefer discrete gpus
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("No suitable physical device found")
}

fn create_logical_device(
    physical_device: Arc<PhysicalDevice>,
    queue_family_index: u32,
    device_extensions: &DeviceExtensions,
) -> (Arc<Device>, Arc<Queue>) {
    let (device, mut queues) = Device::new(
        physical_device.clone(),
        DeviceCreateInfo {
            enabled_extensions: *device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("could not create logical device");

    let graphics_queue = queues.next().expect("could not fetch queue");

    (device, graphics_queue)
}
