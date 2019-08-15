extern crate winit;
#[macro_use]
extern crate log;
extern crate arrayvec;
extern crate env_logger;
extern crate gfx_hal;
extern crate image;

use arrayvec::ArrayVec;
use core::mem::ManuallyDrop;
#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;
use gfx_hal::{
    adapter::{Adapter, MemoryTypeId, PhysicalDevice},
    buffer::{self, IndexBufferView},
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
    memory::{Properties, Requirements},
    pass::{Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, Subpass, SubpassDesc},
    pool::{CommandPool, CommandPoolCreateFlags},
    pso::{
        AttributeDesc, BakedStates, BasePipeline, BlendDesc, BlendOp, BlendState, ColorBlendDesc,
        ColorMask, DepthStencilDesc, Descriptor, DescriptorRangeDesc, DescriptorSetLayoutBinding,
        DescriptorSetWrite, DescriptorType, ElemOffset, ElemStride, Element, EntryPoint, Face,
        Factor, FrontFace, GraphicsPipelineDesc, GraphicsShaderSet, InputAssemblerDesc, LogicOp,
        PipelineCreationFlags, PipelineStage, PolygonMode, PrimitiveRestart, Rasterizer, Rect,
        ShaderStageFlags, Specialization, VertexBufferDesc, VertexInputRate, Viewport,
    },
    queue::{family::QueueGroup, Submission},
    window::{Extent2D, PresentMode, Suboptimal, Surface, Swapchain, SwapchainConfig},
    Backend, Capability, CommandQueue, DescriptorPool, Features, Gpu, Graphics, IndexType,
    Instance, Primitive, QueueFamily, Supports, Transfer,
};
use image::RgbaImage;
use std::{borrow::Cow, marker::PhantomData, mem::size_of, ops::Deref, time::Instant};
use winit::{dpi::LogicalSize, *};

const WINDOW_NAME: &str = "Hello World!";

pub const VERTEX_SOURCE: &str = include_str!("shaders/vert_default.vert");
pub const FRAGMENT_SOURCE: &str = include_str!("shaders/frag_default.frag");
pub const ZELDA: &[u8] = include_bytes!("img/test_png.png");

macro_rules! manual_drop {
    ($this_val:expr) => {
        ManuallyDrop::into_inner(read(&$this_val))
    };
}

macro_rules! manual_new {
    ($this_val:ident) => {
        ManuallyDrop::new($this_val)
    };
}

fn main() {
    env_logger::init();

    let logical_size = LogicalSize::new(1920.0, 1080.0);
    let mut window_state =
        WinitState::new(WINDOW_NAME, logical_size).expect("Error on windows creation.");
    let mut hal_state = HalState::new(&window_state.window).unwrap();
    let mut local_state = LocalState::new(logical_size);

    loop {
        let inputs = UserInput::poll_events_loop(&mut window_state.events_loop);
        if inputs.end_requested {
            break;
        }
        if inputs.new_frame_size.is_some() {
            debug!("Window changed size, restarting HalState...");
            drop(hal_state);

            hal_state = HalState::new(&window_state.window).unwrap();
        }

        local_state.update_from_input(inputs);
        if let Err(e) = do_the_render(&mut hal_state, &local_state) {
            error!("Rendering Error: {:?}", e);
            debug!("Auto-restarting HalState...");
            drop(hal_state);
            hal_state = HalState::new(&window_state.window).unwrap();
        }
    }
}

pub fn do_the_render(
    hal_state: &mut HalState,
    local_state: &LocalState,
) -> Result<Option<Suboptimal>, &'static str> {
    let x1 = 100.0;
    let y1 = 100.0;
    let x2 = local_state.mouse_x as f32;
    let y2 = local_state.mouse_y as f32;
    let quad = Quad {
        x: (x1 / local_state.frame_width as f32) * 2.0 - 1.0,
        y: (y1 / local_state.frame_height as f32) * 2.0 - 1.0,
        w: ((x2 - x1) / local_state.frame_width as f32) * 2.0,
        h: ((y2 - y1) / local_state.frame_height as f32) * 2.0,
    };
    hal_state.draw_quad_frame(quad)
}

pub struct WinitState {
    pub events_loop: EventsLoop,
    pub window: Window,
}
impl WinitState {
    pub fn new<T: Into<String>>(title: T, size: LogicalSize) -> Result<Self, CreationError> {
        let events_loop = EventsLoop::new();
        let output = WindowBuilder::new()
            .with_title(title)
            .with_dimensions(size)
            .build(&events_loop);

        output.map(|window| Self {
            events_loop,
            window,
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct UserInput {
    pub end_requested: bool,
    pub new_frame_size: Option<(f64, f64)>,
    pub new_mouse_position: Option<(f64, f64)>,
}
impl UserInput {
    pub fn poll_events_loop(events_loop: &mut EventsLoop) -> Self {
        let mut output = UserInput::default();
        events_loop.poll_events(|event| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                output.end_requested = true;
                debug!("End was requested!");
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(logical),
                ..
            } => {
                output.new_frame_size = Some((logical.width, logical.height));
                debug!("Our new frame size is {:?}", output.new_frame_size);
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                output.new_mouse_position = Some((position.x, position.y));
            }
            _ => (),
        });
        output
    }
}

#[derive(Debug)]
pub struct LocalState {
    pub frame_width: f64,
    pub frame_height: f64,
    pub mouse_x: f64,
    pub mouse_y: f64,
}
impl LocalState {
    pub fn new(logical_size: LogicalSize) -> LocalState {
        LocalState {
            frame_height: logical_size.height,
            frame_width: logical_size.width,
            mouse_x: 0.0,
            mouse_y: 0.0,
        }
    }

    pub fn update_from_input(&mut self, input: UserInput) {
        if let Some(frame_size) = input.new_frame_size {
            self.frame_width = frame_size.0;
            self.frame_height = frame_size.1;
        }
        if let Some(position) = input.new_mouse_position {
            self.mouse_x = position.0;
            self.mouse_y = position.1;
        }
    }
}

pub struct HalState {
    // Top
    _instance: ManuallyDrop<back::Instance>,
    _surface: <back::Backend as Backend>::Surface,
    _adapter: Adapter<back::Backend>,
    queue_group: ManuallyDrop<QueueGroup<back::Backend, Graphics>>,
    device: ManuallyDrop<back::Device>,

    // Pipeline nonsense
    vertices: BufferBundle<back::Backend, back::Device>,
    indexes: BufferBundle<back::Backend, back::Device>,
    descriptor_set_layouts: Vec<<back::Backend as Backend>::DescriptorSetLayout>,
    descriptor_set: ManuallyDrop<<back::Backend as Backend>::DescriptorSet>,
    descriptor_pool: ManuallyDrop<<back::Backend as Backend>::DescriptorPool>,
    pipeline_layout: ManuallyDrop<<back::Backend as Backend>::PipelineLayout>,
    graphics_pipeline: ManuallyDrop<<back::Backend as Backend>::GraphicsPipeline>,

    // GPU Swapchain
    swapchain: ManuallyDrop<<back::Backend as Backend>::Swapchain>,
    render_area: Rect,
    in_flight_fences: Vec<<back::Backend as Backend>::Fence>,
    frames_in_flight: usize,
    image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore>,
    render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore>,

    // Render Pass
    render_pass: ManuallyDrop<<back::Backend as Backend>::RenderPass>,

    // Render Targets
    image_views: Vec<(<back::Backend as Backend>::ImageView)>,
    framebuffers: Vec<<back::Backend as Backend>::Framebuffer>,

    // Command Issues
    command_pool: ManuallyDrop<CommandPool<back::Backend, Graphics>>,
    command_buffers: Vec<CommandBuffer<back::Backend, Graphics, MultiShot, Primary>>,

    // Mis
    current_frame: usize,
    creation_time: Instant,
}

impl HalState {
    pub fn new(window: &Window) -> Result<Self, &'static str> {
        let instance = back::Instance::create(WINDOW_NAME, 1);
        let mut surface = instance.create_surface(window);

        let creation_time = Instant::now();

        let adapter = instance
            .enumerate_adapters()
            .into_iter()
            .find(|a| {
                a.queue_families
                    .iter()
                    .any(|qf| qf.supports_graphics() && surface.supports_queue_family(qf))
            })
            .ok_or("Couldn't find a graphical adapter!")?;

        // open it up!
        let (mut device, mut queue_group) = {
            let queue_family = adapter
                .queue_families
                .iter()
                .find(|qf| qf.supports_graphics() && surface.supports_queue_family(qf))
                .ok_or("Couldn't find a QueueFamily with graphics!")?;

            let Gpu { device, mut queues } = unsafe {
                adapter
                    .physical_device
                    .open(&[(queue_family, &[1.0; 1])], Features::empty())
                    .map_err(|_| "Couldn't open the PhysicalDevice!")?
            };

            let queue_group = queues
                .take::<Graphics>(queue_family.id())
                .expect("Couldn't take ownership of the QueueGroup!");

            if queue_group.queues.len() == 0 {
                return Err("The QueueGroup did not have any CommandQueues available");
            }
            (device, queue_group)
        };

        let (swapchain, extent, backbuffer, format, frames_in_flight) = {
            // no composite alpha here
            let (caps, preferred_formats, present_modes) =
                surface.compatibility(&adapter.physical_device);
            trace!("{:?}", caps);
            trace!("Preferred Formats: {:?}", preferred_formats);
            trace!("Present Modes: {:?}", present_modes);

            let present_mode = {
                use gfx_hal::window::PresentMode::*;
                [Mailbox, Fifo, Relaxed, Immediate]
                    .iter()
                    .cloned()
                    .find(|pm| present_modes.contains(pm))
                    .ok_or("No PresentMode values specified!")?
            };

            use gfx_hal::window::CompositeAlpha;
            trace!("We're setting composite alpha to opaque...Need to figure out where to find the user's intent.");
            let composite_alpha = CompositeAlpha::OPAQUE;

            let format = match preferred_formats {
                None => Format::Rgba8Srgb,
                Some(formats) => match formats
                    .iter()
                    .find(|format| format.base_format().1 == ChannelType::Srgb)
                    .cloned()
                {
                    Some(srgb_format) => srgb_format,
                    None => formats
                        .get(0)
                        .cloned()
                        .ok_or("Preferred format list was empty!")?,
                },
            };

            let extent = {
                let window_client_area = window
                    .get_inner_size()
                    .ok_or("Window doesn't exist!")?
                    .to_physical(window.get_hidpi_factor());

                Extent2D {
                    width: caps
                        .extents
                        .end()
                        .width
                        .min(window_client_area.width as u32),
                    height: caps
                        .extents
                        .end()
                        .height
                        .min(window_client_area.height as u32),
                }
            };

            let image_count = if present_mode == PresentMode::Mailbox {
                (caps.image_count.end() - 1).min(*caps.image_count.start().max(&3))
            } else {
                (caps.image_count.end() - 1).min(*caps.image_count.start().max(&2))
            };

            let image_layers = 1;
            if caps.usage.contains(Usage::COLOR_ATTACHMENT) == false {
                return Err("The Surface isn't capable of supporting color!");
            }

            let image_usage = Usage::COLOR_ATTACHMENT;

            let swapchain_config = SwapchainConfig {
                present_mode,
                composite_alpha,
                format,
                extent,
                image_count,
                image_layers,
                image_usage,
            };

            trace!("{:?}", swapchain_config);

            // Final pop out. PHEW!
            let (swapchain, backbuffer) = unsafe {
                device
                    .create_swapchain(&mut surface, swapchain_config, None)
                    .map_err(|_| "Failed to create the swapchain on the last step!")?
            };

            (swapchain, extent, backbuffer, format, image_count as usize)
        };

        let (image_available_semaphores, render_finished_semaphores, in_flight_fences) = {
            let mut image_available_semaphores: Vec<<back::Backend as Backend>::Semaphore> = vec![];
            let mut render_finished_semaphores: Vec<<back::Backend as Backend>::Semaphore> = vec![];
            let mut in_flight_fences: Vec<<back::Backend as Backend>::Fence> = vec![];
            for _ in 0..frames_in_flight {
                in_flight_fences.push(
                    device
                        .create_fence(true)
                        .map_err(|_| "Could not create a fence!")?,
                );
                image_available_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create a semaphore!")?,
                );
                render_finished_semaphores.push(
                    device
                        .create_semaphore()
                        .map_err(|_| "Could not create a semaphore!")?,
                );
            }
            (
                image_available_semaphores,
                render_finished_semaphores,
                in_flight_fences,
            )
        };

        let render_pass = {
            let color_attachment = Attachment {
                format: Some(format),
                samples: 1,
                ops: AttachmentOps {
                    load: AttachmentLoadOp::Clear,
                    store: AttachmentStoreOp::Store,
                },
                stencil_ops: AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };

            let subpass = SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            unsafe {
                device
                    .create_render_pass(&[color_attachment], &[subpass], &[])
                    .map_err(|_| "Couldn't create a render pass!")?
            }
        };

        let image_views = {
            backbuffer
                .into_iter()
                .map(|image| unsafe {
                    device
                        .create_image_view(
                            &image,
                            ViewKind::D2,
                            format,
                            Swizzle::NO,
                            SubresourceRange {
                                aspects: Aspects::COLOR,
                                levels: 0..1,
                                layers: 0..1,
                            },
                        )
                        .map_err(|_| "Couldn't create the image_view for the image!")
                })
                .collect::<Result<Vec<_>, &str>>()?
        };

        let framebuffers: Vec<<back::Backend as Backend>::Framebuffer> = {
            image_views
                .iter()
                .map(|image_view| unsafe {
                    device
                        .create_framebuffer(
                            &render_pass,
                            vec![image_view],
                            Extent {
                                width: extent.width as u32,
                                height: extent.height as u32,
                                depth: 1,
                            },
                        )
                        .map_err(|_| "Failed to create a framebuffer!")
                })
                .collect::<Result<Vec<_>, &str>>()?
        };

        let mut command_pool = unsafe {
            device
                .create_command_pool_typed(&queue_group, CommandPoolCreateFlags::RESET_INDIVIDUAL)
                .map_err(|_| "Could not create the raw command pool!")?
        };

        let command_buffers: Vec<_> = framebuffers
            .iter()
            .map(|_| command_pool.acquire_command_buffer())
            .collect();

        let (
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            graphics_pipeline,
        ) = Self::create_pipeline(&mut device, extent, &render_pass)?;

        const F32_XY_RGB_UV_QUAD: u64 = (size_of::<f32>() * 2 * 3) as u64;
        let vertices =
            BufferBundle::new(&adapter, &device, F32_XY_RGB_UV_QUAD, buffer::Usage::VERTEX)?;

        const U16_QUAD_INDICES: u64 = (size_of::<u16>() * 2 * 3) as u64;
        let indexes = BufferBundle::new(&adapter, &device, U16_QUAD_INDICES, buffer::Usage::INDEX)?;

        // WRITE INDEX DATA
        unsafe {
            let mut data_target = device
                .acquire_mapping_writer(&indexes.memory, 0..indexes.requirements.size)
                .map_err(|_| "Failed to acquire an index buffer mapping writer!")?;

            const INDEX_DATA: &[u16] = &[0, 1, 2, 2, 3, 0];
            data_target[..INDEX_DATA.len()].copy_from_slice(&INDEX_DATA);

            device
                .release_mapping_writer(data_target)
                .map_err(|_| "Couldn't release the index buffer mapping writer!")?;
        }

        // Create the texture
        let texture = LoadedImage::new(
            &adapter,
            &device,
            &mut command_pool,
            &mut queue_group.queues[0],
            image::load_from_memory(ZELDA)
                .expect("Binary Corrupted!")
                .to_rgba(),
        )?;

        // Write the descriptors into the descriptor set
        unsafe {
            device.write_descriptor_sets(vec![
                DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Image(
                        texture.image_view.deref(),
                        Layout::ShaderReadOnlyOptimal,
                    )),
                },
                DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 1,
                    array_offset: 0,
                    descriptors: Some(Descriptor::Sampler(texture.sampler.deref())),
                },
            ]);
        }

        Ok(Self {
            _instance: manual_new!(instance),
            _surface: surface,
            _adapter: adapter,
            device: manual_new!(device),
            queue_group: manual_new!(queue_group),
            swapchain: manual_new!(swapchain),
            render_area: extent.to_extent().rect(),
            render_pass: manual_new!(render_pass),
            image_views,
            framebuffers,
            command_pool: manual_new!(command_pool),
            command_buffers,
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            frames_in_flight,
            current_frame: 0,

            vertices,
            indexes,
            descriptor_set_layouts,
            descriptor_pool: manual_new!(descriptor_pool),
            descriptor_set: manual_new!(descriptor_set),
            pipeline_layout: manual_new!(pipeline_layout),
            graphics_pipeline: manual_new!(graphics_pipeline),
            creation_time,
        })
    }

    fn create_pipeline(
        device: &mut back::Device,
        extent: Extent2D,
        render_pass: &<back::Backend as Backend>::RenderPass,
    ) -> Result<
        (
            Vec<<back::Backend as Backend>::DescriptorSetLayout>,
            <back::Backend as Backend>::DescriptorPool,
            <back::Backend as Backend>::DescriptorSet,
            <back::Backend as Backend>::PipelineLayout,
            <back::Backend as Backend>::GraphicsPipeline,
        ),
        &'static str,
    > {
        let mut compiler = shaderc::Compiler::new().ok_or("shaderc not found!")?;
        let vertex_compile_artifact = compiler
            .compile_into_spirv(
                VERTEX_SOURCE,
                shaderc::ShaderKind::Vertex,
                "vertex.vert",
                "main",
                None,
            )
            .map_err(|_| "Couldn't compile vertex shader!")?;

        let fragment_compile_artifact = compiler
            .compile_into_spirv(
                FRAGMENT_SOURCE,
                shaderc::ShaderKind::Fragment,
                "fragment.frag",
                "main",
                None,
            )
            .map_err(|e| {
                error!("{}", e);
                "Couldn't compile fragment shader!"
            })?;

        let vertex_shader_module = unsafe {
            device
                .create_shader_module(vertex_compile_artifact.as_binary())
                .map_err(|_| "Couldn't make the vertex module!")?
        };

        let fragment_shader_module = unsafe {
            device
                .create_shader_module(fragment_compile_artifact.as_binary())
                .map_err(|_| "Couldn't make the fragment module!")?
        };

        let (vs_entry, fs_entry) = (
            EntryPoint {
                entry: "main",
                module: &vertex_shader_module,
                specialization: Specialization {
                    constants: Cow::Borrowed(&[]),
                    data: Cow::Borrowed(&[]),
                },
            },
            EntryPoint {
                entry: "main",
                module: &fragment_shader_module,
                specialization: Specialization {
                    constants: Cow::Borrowed(&[]),
                    data: Cow::Borrowed(&[]),
                },
            },
        );

        let input_assembler = InputAssemblerDesc {
            primitive: Primitive::TriangleList,
            primitive_restart: PrimitiveRestart::Disabled,
        };

        let shaders = GraphicsShaderSet {
            vertex: vs_entry,
            fragment: Some(fs_entry),
            domain: None,
            geometry: None,
            hull: None,
        };

        const VERTEX_STRIDE: ElemStride = (size_of::<f32>() * (2 + 3 + 2)) as ElemStride;
        let vertex_buffers = vec![VertexBufferDesc {
            binding: 0,
            stride: VERTEX_STRIDE,
            rate: VertexInputRate::Vertex,
        }];

        let position_attribute = AttributeDesc {
            location: 0,
            binding: 0,
            element: Element {
                format: Format::Rg32Sfloat,
                offset: 0,
            },
        };

        let color_attribute = AttributeDesc {
            location: 1,
            binding: 0,
            element: Element {
                format: Format::Rgb32Sfloat,
                offset: (size_of::<f32>() * 2) as ElemOffset,
            },
        };

        const UV_STRIDE: ElemStride = (size_of::<f32>() * 5) as ElemStride;
        let uv_attribute = AttributeDesc {
            location: 2,
            binding: 0,
            element: Element {
                format: Format::Rg32Sfloat,
                offset: UV_STRIDE,
            },
        };

        let attributes = vec![position_attribute, color_attribute, uv_attribute];

        let rasterizer = Rasterizer {
            depth_clamping: false,
            polygon_mode: PolygonMode::Fill,
            cull_face: Face::NONE,
            front_face: FrontFace::Clockwise,
            depth_bias: None,
            conservative: false,
        };

        let depth_stencil = DepthStencilDesc {
            depth: None,
            depth_bounds: false,
            stencil: None,
        };

        let blender = {
            let blend_state = BlendState {
                color: BlendOp::Add {
                    src: Factor::One,
                    dst: Factor::Zero,
                },
                alpha: BlendOp::Add {
                    src: Factor::One,
                    dst: Factor::Zero,
                },
            };
            BlendDesc {
                logic_op: Some(LogicOp::Copy),
                targets: vec![ColorBlendDesc {
                    mask: ColorMask::ALL,
                    blend: Some(blend_state),
                }],
            }
        };

        let baked_states = BakedStates {
            viewport: Some(Viewport {
                rect: extent.to_extent().rect(),
                depth: (0.0..1.0),
            }),
            scissor: Some(extent.to_extent().rect()),
            blend_color: None,
            depth_bounds: None,
        };

        let descriptor_set_layouts = vec![unsafe {
            device
                .create_descriptor_set_layout(
                    &[
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: DescriptorType::SampledImage,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding {
                            binding: 1,
                            ty: DescriptorType::Sampler,
                            count: 1,
                            stage_flags: ShaderStageFlags::FRAGMENT,
                            immutable_samplers: false,
                        },
                    ],
                    &[],
                )
                .map_err(|_| "Couldn't make a DescriptorSetLayout!")?
        }];

        let mut descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    1,
                    &[
                        DescriptorRangeDesc {
                            ty: DescriptorType::SampledImage,
                            count: 1,
                        },
                        DescriptorRangeDesc {
                            ty: DescriptorType::Sampler,
                            count: 1,
                        },
                    ],
                    gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
                )
                .map_err(|_| "Couldn't create a descriptor pool!")?
        };

        // 3. you allocate said descriptor set from the pool you made earlier
        let descriptor_set = unsafe {
            descriptor_pool
                .allocate_set(&descriptor_set_layouts[0])
                .map_err(|_| "Couldn't make a Descriptor Set!")?
        };

        let push_constants = vec![(ShaderStageFlags::FRAGMENT, 0..1)];
        let layout = unsafe {
            device
                .create_pipeline_layout(&descriptor_set_layouts, push_constants)
                .map_err(|_| "Couldn't create a pipeline layout")?
        };

        let gfx_pipeline = {
            let desc = GraphicsPipelineDesc {
                shaders,
                rasterizer,
                vertex_buffers,
                attributes,
                input_assembler,
                blender,
                depth_stencil,
                multisampling: None,
                baked_states,
                layout: &layout,
                subpass: Subpass {
                    index: 0,
                    main_pass: render_pass,
                },
                flags: PipelineCreationFlags::empty(),
                parent: BasePipeline::None,
            };

            unsafe {
                device
                    .create_graphics_pipeline(&desc, None)
                    .map_err(|_| "Couldn't create a graphics pipeline!")?
            }
        };

        Ok((
            descriptor_set_layouts,
            descriptor_pool,
            descriptor_set,
            layout,
            gfx_pipeline,
        ))
    }

    pub fn draw_clear_frame(
        &mut self,
        color: [f32; 4],
    ) -> Result<Option<Suboptimal>, &'static str> {
        // SETUP FOR THIS FRAME
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        // Advance the frame _before_ we start using the `?` operator
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let image_index = self
                .swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
            (image_index.0, image_index.0 as usize)
        };
        let flight_fence = &self.in_flight_fences[i_usize];

        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence!")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence!")?;
        }

        // RECORD COMMANDS
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            let clear_values = [ClearValue::Color(ClearColor::Sfloat(color))];
            buffer.begin(false);
            buffer.begin_render_pass_inline(
                &self.render_pass,
                &self.framebuffers[i_usize],
                self.render_area,
                clear_values.iter(),
            );
            buffer.finish();
        }

        // SUBMISSION AND PRESENT
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        // yes, you have to write it twice like this. yes, it's silly.
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
        let the_command_queue = &mut self.queue_group.queues[0];
        unsafe {
            the_command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(the_command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!")
        }
    }

    pub fn draw_quad_frame(&mut self, quad: Quad) -> Result<Option<Suboptimal>, &'static str> {
        // SETUP FOR THIS FRAME
        let image_available = &self.image_available_semaphores[self.current_frame];
        let render_finished = &self.render_finished_semaphores[self.current_frame];
        // Advance the frame _before_ we start using the `?` operator
        self.current_frame = (self.current_frame + 1) % self.frames_in_flight;

        let (i_u32, i_usize) = unsafe {
            let image_index = self
                .swapchain
                .acquire_image(core::u64::MAX, Some(image_available), None)
                .map_err(|_| "Couldn't acquire an image from the swapchain!")?;
            (image_index.0, image_index.0 as usize)
        };

        // Get the fence, and wait for the fence
        let flight_fence = &self.in_flight_fences[i_usize];
        unsafe {
            self.device
                .wait_for_fence(flight_fence, core::u64::MAX)
                .map_err(|_| "Failed to wait on the fence!")?;
            self.device
                .reset_fence(flight_fence)
                .map_err(|_| "Couldn't reset the fence!")?;
        }

        // WRITE QUAD DATA
        unsafe {
            let mut data_target = self
                .device
                .acquire_mapping_writer(&self.vertices.memory, 0..self.vertices.requirements.size)
                .map_err(|_| "Failed to acquire a memory writer!")?;

            let data = quad.vertex_attributes();
            data_target[..data.len()].copy_from_slice(&data);

            self.device
                .release_mapping_writer(data_target)
                .map_err(|_| "Couldn't release the mapping writing!")?;
        }

        // DETERMINE THE TIME DATA
        let duration = Instant::now().duration_since(self.creation_time);
        let time_f32 = duration.as_secs() as f32 + duration.subsec_nanos() as f32 * 1e-9;

        // RECORD COMMANDS
        unsafe {
            let buffer = &mut self.command_buffers[i_usize];
            const TRIANGLE_CLEAR: [ClearValue; 1] =
                [ClearValue::Color(ClearColor::Sfloat([0.1, 0.2, 0.3, 1.0]))];
            buffer.begin(false);
            {
                let mut encoder = buffer.begin_render_pass_inline(
                    &self.render_pass,
                    &self.framebuffers[i_usize],
                    self.render_area,
                    TRIANGLE_CLEAR.iter(),
                );
                encoder.bind_graphics_pipeline(&self.graphics_pipeline);

                let vertex_buffers: ArrayVec<[_; 1]> = [(self.vertices.buffer.deref(), 0)].into();
                encoder.bind_vertex_buffers(0, vertex_buffers);
                encoder.bind_index_buffer(IndexBufferView {
                    buffer: &self.indexes.buffer,
                    offset: 0,
                    index_type: IndexType::U16,
                });

                encoder.bind_graphics_descriptor_sets(
                    &self.pipeline_layout,
                    0,
                    Some(self.descriptor_set.deref()),
                    &[],
                );

                encoder.push_graphics_constants(
                    &self.pipeline_layout,
                    ShaderStageFlags::FRAGMENT,
                    0,
                    &[time_f32.to_bits()],
                );
                encoder.draw_indexed(0..6, 0, 0..1);
            }
            buffer.finish();
        }

        // SUBMISSION AND PRESENT
        let command_buffers = &self.command_buffers[i_usize..=i_usize];
        let wait_semaphores: ArrayVec<[_; 1]> =
            [(image_available, PipelineStage::COLOR_ATTACHMENT_OUTPUT)].into();
        let signal_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        // yes, you have to write it twice like this. yes, it's silly.
        let present_wait_semaphores: ArrayVec<[_; 1]> = [render_finished].into();
        let submission = Submission {
            command_buffers,
            wait_semaphores,
            signal_semaphores,
        };
        let the_command_queue = &mut self.queue_group.queues[0];
        unsafe {
            the_command_queue.submit(submission, Some(flight_fence));
            self.swapchain
                .present(the_command_queue, i_u32, present_wait_semaphores)
                .map_err(|_| "Failed to present into the swapchain!")
        }
    }
}

impl core::ops::Drop for HalState {
    fn drop(&mut self) {
        error!("Dropping HALState.");
        self.device.wait_idle().unwrap();

        unsafe {
            for fence in self.in_flight_fences.drain(..) {
                self.device.destroy_fence(fence);
            }

            for semaphore in self.render_finished_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for semaphore in self.image_available_semaphores.drain(..) {
                self.device.destroy_semaphore(semaphore)
            }
            for framebuffer in self.framebuffers.drain(..) {
                self.device.destroy_framebuffer(framebuffer);
            }
            for image_view in self.image_views.drain(..) {
                self.device.destroy_image_view(image_view);
            }

            for this_layout in self.descriptor_set_layouts.drain(..) {
                self.device.destroy_descriptor_set_layout(this_layout);
            }

            // LAST RESORT STYLE CODE, NOT TO BE IMITATED LIGHTLY
            use core::ptr::read;
            self.vertices.manually_drop(&self.device);
            self.indexes.manually_drop(&self.device);

            self.device
                .destroy_pipeline_layout(manual_drop!(self.pipeline_layout));
            self.device
                .destroy_graphics_pipeline(manual_drop!(self.graphics_pipeline));
            self.device
                .destroy_command_pool(manual_drop!(self.command_pool).into_raw());
            self.device
                .destroy_render_pass(manual_drop!(self.render_pass));
            self.device.destroy_swapchain(manual_drop!(self.swapchain));
            self.device
                .destroy_descriptor_pool(manual_drop!(self.descriptor_pool));

            ManuallyDrop::drop(&mut self.device);
            ManuallyDrop::drop(&mut self._instance);
        }
    }
}

pub struct BufferBundle<B: Backend, D: Device<B>> {
    pub buffer: ManuallyDrop<B::Buffer>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub phantom: PhantomData<D>,
}

impl<B: Backend, D: Device<B>> BufferBundle<B, D> {
    pub fn new(
        adapter: &Adapter<B>,
        device: &D,
        size: u64,
        usage: buffer::Usage,
    ) -> Result<Self, &'static str> {
        unsafe {
            let mut buffer = device
                .create_buffer(size, usage)
                .map_err(|_| "Couldn't create a buffer for the vertices")?;

            let requirements = device.get_buffer_requirements(&buffer);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::CPU_VISIBLE)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the vertex buffer!")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate vertex buffer memory")?;

            device
                .bind_buffer_memory(&memory, 0, &mut buffer)
                .map_err(|_| "Couldn't bind the buffer memory!")?;

            Ok(Self {
                buffer: manual_new!(buffer),
                requirements,
                memory: manual_new!(memory),
                phantom: PhantomData,
            })
        }
    }

    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;
        device.destroy_buffer(ManuallyDrop::into_inner(read(&self.buffer)));
        device.free_memory(ManuallyDrop::into_inner(read(&self.memory)));
    }
}

pub struct LoadedImage<B: Backend, D: Device<B>> {
    pub image: ManuallyDrop<B::Image>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,
    pub sampler: ManuallyDrop<B::Sampler>,
    pub phantom: PhantomData<D>,
}

impl<B: Backend, D: Device<B>> LoadedImage<B, D> {
    pub fn new<C: Capability + Supports<Transfer>>(
        adapter: &Adapter<B>,
        device: &D,
        command_pool: &mut CommandPool<B, C>,
        command_queue: &mut CommandQueue<B, C>,
        img: RgbaImage,
    ) -> Result<Self, &'static str> {
        unsafe {
            // 0.   First we compute some memory related values:
            let pixel_size = size_of::<image::Rgba<u8>>();
            let row_size = pixel_size * (img.width() as usize);
            let limits = adapter.physical_device.limits();
            let row_alignment_mask = limits.optimal_buffer_copy_pitch_alignment as u32 - 1;
            let row_pitch = ((row_size as u32 + row_alignment_mask) & !row_alignment_mask) as usize;
            debug_assert!(row_pitch as usize >= row_size);

            // 1.   Make a staging buffer with enough memory for the image
            //      and a trsnfer_src image
            let required_bytes = (row_pitch * img.height() as usize) as u64;
            let staging_bundle = BufferBundle::new(
                &adapter,
                device,
                required_bytes,
                buffer::Usage::TRANSFER_SRC,
            )?;

            // 2.   Use a mapping writer to put the image data into the buffer
            let mut writer = device
                .acquire_mapping_writer::<u8>(
                    &staging_bundle.memory,
                    0..staging_bundle.requirements.size,
                )
                .map_err(|_| "Couldn't acquire a mapping writer to the staging buffer!")?;

            for y in 0..img.height() as usize {
                let row = &(*img)[y * row_size..(y + 1) * row_size];
                let dest_base = y * row_pitch;
                writer[dest_base..dest_base + row.len()].copy_from_slice(row);
            }
            device
                .release_mapping_writer(writer)
                .map_err(|_| "Couldn't release the mapping writer to the staging buffer!")?;

            //  3. Make the image
            let image_object = device
                .create_image(
                    gfx_hal::image::Kind::D2(img.width(), img.height(), 1, 1),
                    1,
                    Format::Rgba8Srgb,
                    gfx_hal::image::Tiling::Optimal,
                    Usage::TRANSFER_DST | Usage::SAMPLED,
                    gfx_hal::image::ViewCapabilities::empty(),
                )
                .map_err(|_| "Couldn't create the image!")?;

            //  4. allocate the memory and bind it
            let requirements = device.get_image_requirements(&image_object);
            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::DEVICE_LOCAL)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the image!")?;
            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate image memory!")?;

            // 5. create image view and sampler
            let image_view = device
                .create_image_view(
                    &image_object,
                    gfx_hal::image::ViewKind::D2,
                    Format::Rgba8Srgb,
                    gfx_hal::format::Swizzle::NO,
                    SubresourceRange {
                        aspects: Aspects::COLOR,
                        levels: 0..1,
                        layers: 0..1,
                    },
                )
                .map_err(|_| "Couldn't create the image view!")?;


            let sampler = device
                .create_sampler(gfx_hal::image::SamplerInfo::new(
                    gfx_hal::image::Filter::Nearest,
                    gfx_hal::image::WrapMode::Tile,
                ))
                .map_err(|_| "Couldn't create the sampler!")?;

            // 6. create the command buffer
            let mut cmd_buffer = command_pool.acquire_command_buffer::<gfx_hal::command::OneShot>();
            cmd_buffer.begin();

            // 7. Use a pipeline barrier to transition the image from empty/undefined
            //    to TRANSFER_WRITE/TransferDstOptimal
            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (gfx_hal::image::Access::empty(), Layout::Undefined)
                    ..(
                        gfx_hal::image::Access::TRANSFER_WRITE,
                        Layout::TransferDstOptimal,
                    ),
                target: &image_object,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TOP_OF_PIPE..PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            //  8. perform copy!
            cmd_buffer.copy_buffer_to_image(
                &staging_bundle.buffer,
                &image_object,
                Layout::TransferDstOptimal,
                &[gfx_hal::command::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_width: (row_pitch / pixel_size) as u32,
                    buffer_height: img.height(),
                    image_layers: gfx_hal::image::SubresourceLayers {
                        aspects: Aspects::COLOR,
                        level: 0,
                        layers: 0..1,
                    },
                    image_offset: gfx_hal::image::Offset { x: 0, y: 0, z: 0 },
                    image_extent: gfx_hal::image::Extent {
                        width: img.width(),
                        height: img.height(),
                        depth: 1,
                    },
                }],
            );

            // 9. use pipeline barrier to transition the image to SHADER_READ access/
            //    ShaderReadOnlyOptimal layout
            let image_barrier = gfx_hal::memory::Barrier::Image {
                states: (
                    gfx_hal::image::Access::TRANSFER_WRITE,
                    Layout::TransferDstOptimal,
                )
                    ..(
                        gfx_hal::image::Access::SHADER_READ,
                        Layout::ShaderReadOnlyOptimal,
                    ),
                target: &image_object,
                families: None,
                range: SubresourceRange {
                    aspects: Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };
            cmd_buffer.pipeline_barrier(
                PipelineStage::TRANSFER..PipelineStage::FRAGMENT_SHADER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            //  10. Submit it!
            cmd_buffer.finish();

            let upload_fence = device
                .create_fence(false)
                .map_err(|_| "Couldn't create upload fence!")?;
            command_queue.submit_without_semaphores(Some(&cmd_buffer), Some(&upload_fence));
            device
                .wait_for_fence(&upload_fence, core::u64::MAX)
                .map_err(|_| "Couldn't wait for the fence!")?;
            device.destroy_fence(upload_fence);

            //  11. Kill off our buffer!
            staging_bundle.manually_drop(device);
            command_pool.free(Some(cmd_buffer));

            Ok(Self {
                image: manual_new!(image_object),
                requirements,
                memory: manual_new!(memory),
                image_view: manual_new!(image_view),
                sampler: manual_new!(sampler),
                phantom: PhantomData,
            })
        }
    }

    pub unsafe fn manually_drop(&self, device: &D) {
        use core::ptr::read;
        device.destroy_sampler(ManuallyDrop::into_inner(read(&self.sampler)));
        device.destroy_image(manual_drop!(self.image));
        device.destroy_image_view(manual_drop!(self.image_view));
        device.free_memory(manual_drop!(self.memory));
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Triangle {
    pub points: [[f32; 2]; 3],
}
impl Triangle {
    pub fn vertex_attributes(self) -> [f32; 3 * (2 + 3)] {
        let [[a, b], [c, d], [e, f]] = self.points;
        [
            a, b, 1.0, 0.0, 0.0, // red
            c, d, 0.0, 1.0, 0.0, // green
            e, f, 0.0, 0.0, 1.0, // blue
        ]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Quad {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Quad {
    pub fn vertex_attributes(self) -> [f32; 4 * (2 + 3 + 2)] {
        let x = self.x;
        let y = self.y;
        let w = self.w;
        let h = self.h;

        #[cfg_attr(rustfmt, rustfmt_skip)]
        [
            // X    Y       R       G       B       type        X       Y       human location
            x,      y + h,  1.0,    0.0,    0.0, /* red   */    0.0,    1.0, /* bottom left */
            x,      y,      0.0,    1.0,    0.0, /* green */    0.0,    0.0, /* top left*/
            x + w,  y,      0.0,    0.0,    1.0, /* blue  */    1.0,    0.0, /* top right */
            x + w,  y + h,  1.0,    0.0,    1.0, /* magenta*/   1.0,    1.0, /* bottom right */
        ]
    }
}
