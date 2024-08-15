use std::sync::Arc;
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{
    Device, DeviceCreateInfo, DeviceExtensions, Queue, QueueCreateInfo, QueueFlags,
};
use vulkano::instance::debug::{
    DebugUtilsMessageSeverity, DebugUtilsMessageType, DebugUtilsMessenger,
    DebugUtilsMessengerCallback, DebugUtilsMessengerCreateInfo,
};
use vulkano::instance::{Instance, InstanceCreateInfo, InstanceExtensions};
use vulkano::swapchain::Surface;
use vulkano::{Version, VulkanLibrary};
use winit::event_loop::EventLoop;
use winit::window::Window;

pub struct Context {
    surface: Arc<Surface>,
    device: Arc<Device>,
    physical_device: Arc<PhysicalDevice>,
    queue: Arc<Queue>,
    _debug_callback: Option<DebugUtilsMessenger>,
}

impl Context {
    pub fn new(window: Arc<Window>, event_loop: &EventLoop<()>) -> Context {
        let (instance, debug_callback) = create_instance(event_loop);

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
            _debug_callback: debug_callback,
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

fn create_instance(event_loop: &EventLoop<()>) -> (Arc<Instance>, Option<DebugUtilsMessenger>) {
    let library = VulkanLibrary::new().expect("no local Vulkan library/DLL");

    let supported_extensions = library.supported_extensions();
    let supported_layers: Vec<_> = library
        .layer_properties()
        .expect("could not enumerate layers")
        .collect();

    // enable debugging if available
    let debug_extension_name = String::from("VK_LAYER_KHRONOS_validation");
    let debug_enabled = supported_extensions.ext_debug_utils
        && supported_layers
            .iter()
            .any(|l| l.name() == debug_extension_name);

    let required_extensions = Surface::required_extensions(&event_loop).unwrap();

    let instance_extensions = InstanceExtensions {
        ext_debug_utils: debug_enabled,
        ..required_extensions
    };

    let mut layers = vec![];
    if debug_enabled {
        layers.push(debug_extension_name);
    }

    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: instance_extensions,
            enabled_layers: layers,
            max_api_version: Some(Version::major_minor(1, 2)),
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    // the debug callback should stay alive as long as the instance
    // otherwise the callback will be dropped and no longer print any messages
    let debug_callback = if debug_enabled {
        create_debug_callback(instance.clone())
    } else {
        None
    };
    (instance, debug_callback)
}

fn create_debug_callback(instance: Arc<Instance>) -> Option<DebugUtilsMessenger> {
    let debug_callback = unsafe {
        DebugUtilsMessenger::new(
            instance.clone(),
            DebugUtilsMessengerCreateInfo {
                message_severity: DebugUtilsMessageSeverity::ERROR
                    | DebugUtilsMessageSeverity::WARNING
                    | DebugUtilsMessageSeverity::INFO
                    | DebugUtilsMessageSeverity::VERBOSE,
                message_type: DebugUtilsMessageType::GENERAL
                    | DebugUtilsMessageType::VALIDATION
                    | DebugUtilsMessageType::PERFORMANCE,
                ..DebugUtilsMessengerCreateInfo::user_callback(DebugUtilsMessengerCallback::new(
                    |message_severity, message_type, callback_data| {
                        let severity = if message_severity
                            .intersects(DebugUtilsMessageSeverity::ERROR)
                        {
                            "error"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::WARNING) {
                            "warning"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::INFO) {
                            "information"
                        } else if message_severity.intersects(DebugUtilsMessageSeverity::VERBOSE) {
                            "verbose"
                        } else {
                            panic!("no-impl");
                        };

                        let ty = if message_type.intersects(DebugUtilsMessageType::GENERAL) {
                            "general"
                        } else if message_type.intersects(DebugUtilsMessageType::VALIDATION) {
                            "validation"
                        } else if message_type.intersects(DebugUtilsMessageType::PERFORMANCE) {
                            "performance"
                        } else {
                            panic!("no-impl");
                        };

                        if message_severity.intersects(DebugUtilsMessageSeverity::VERBOSE)
                            || message_severity.intersects(DebugUtilsMessageSeverity::INFO)
                        {
                            return;
                        }

                        println!(
                            "{} {} {}: {}",
                            callback_data.message_id_name.unwrap_or("unknown"),
                            ty,
                            severity,
                            callback_data.message
                        );
                    },
                ))
            },
        )
        .ok()
    };

    debug_callback
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
