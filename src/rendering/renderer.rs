use arrayvec::ArrayVec;
use core::mem::ManuallyDrop;
use gfx_hal::{
    adapter::{Adapter, PhysicalDevice},
    buffer::{self, IndexBufferView},
    command::{ClearColor, ClearValue, CommandBuffer, MultiShot, Primary},
    device::Device,
    format::{Aspects, ChannelType, Format, Swizzle},
    image::{Extent, Layout, SubresourceRange, Usage, ViewKind},
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
    Backend, DescriptorPool, Features, Gpu, Graphics, IndexType, Instance, Primitive, QueueFamily,
};
use std::{borrow::Cow, mem::size_of, ops::Deref, time::Instant};
use winit::Window;

#[cfg(feature = "dx12")]
use gfx_backend_dx12 as back;
#[cfg(feature = "metal")]
use gfx_backend_metal as back;
#[cfg(feature = "vulkan")]
use gfx_backend_vulkan as back;

use super::{BufferBundle, LoadedImage, Quad, Sprite, SpriteName};

pub const VERTEX_SOURCE: &str = include_str!("shaders/vert_default.vert");
pub const FRAGMENT_SOURCE: &str = include_str!("shaders/frag_default.frag");
pub const ZELDA: &[u8] = include_bytes!("../../resources/sprites/zelda.png");

pub struct Renderer<I: Instance> {
    // Top
    _instance: ManuallyDrop<I>,
    _surface: <I::Backend as Backend>::Surface,
    _adapter: Adapter<I::Backend>,
    queue_group: ManuallyDrop<QueueGroup<I::Backend, Graphics>>,
    device: ManuallyDrop<<I::Backend as Backend>::Device>,

    // Pipeline nonsense
    vertices: BufferBundle<I::Backend>,
    indexes: BufferBundle<I::Backend>,
    descriptor_set_layouts: Vec<<I::Backend as Backend>::DescriptorSetLayout>,
    descriptor_pool: ManuallyDrop<<I::Backend as Backend>::DescriptorPool>,
    pipeline_layout: ManuallyDrop<<I::Backend as Backend>::PipelineLayout>,
    graphics_pipeline: ManuallyDrop<<I::Backend as Backend>::GraphicsPipeline>,

    // GPU Swapchain
    textures: Vec<LoadedImage<I::Backend>>,
    swapchain: ManuallyDrop<<I::Backend as Backend>::Swapchain>,
    render_area: Rect,
    in_flight_fences: Vec<<I::Backend as Backend>::Fence>,
    frames_in_flight: usize,
    image_available_semaphores: Vec<<I::Backend as Backend>::Semaphore>,
    render_finished_semaphores: Vec<<I::Backend as Backend>::Semaphore>,

    // Render Pass
    render_pass: ManuallyDrop<<I::Backend as Backend>::RenderPass>,

    // Render Targets
    image_views: Vec<(<I::Backend as Backend>::ImageView)>,
    framebuffers: Vec<<I::Backend as Backend>::Framebuffer>,

    // Command Issues
    command_pool: ManuallyDrop<CommandPool<I::Backend, Graphics>>,
    command_buffers: Vec<CommandBuffer<I::Backend, Graphics, MultiShot, Primary>>,

    // Mis
    current_frame: usize,
    creation_time: Instant,
}

pub type TypedRenderer = Renderer<back::Instance>;
impl<I: Instance> Renderer<I> {
    pub fn typed_new(
        window: &Window,
        window_name: &str,
        sprite_info: &'static [(SpriteName, &[u8])],
    ) -> Result<(TypedRenderer, Vec<Sprite>), &'static str> {
        // Create An Instance
        let instance = back::Instance::create(window_name, 1);
        // Create A Surface
        let surface = instance.create_surface(window);
        // Create A HalState
        let mut tr = TypedRenderer::new(window, instance, surface)?;
        let sprites = tr.record_textures(sprite_info)?;

        Ok((tr, sprites))
    }

    pub fn new(
        window: &Window,
        instance: I,
        mut surface: <I::Backend as Backend>::Surface,
    ) -> Result<Self, &'static str> {
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
            let mut image_available_semaphores = vec![];
            let mut render_finished_semaphores = vec![];
            let mut in_flight_fences = vec![];
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

        let framebuffers = {
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
            textures: vec![],
            descriptor_set_layouts,
            descriptor_pool: manual_new!(descriptor_pool),
            pipeline_layout: manual_new!(pipeline_layout),
            graphics_pipeline: manual_new!(graphics_pipeline),
            creation_time,
        })
    }

    fn create_pipeline(
        device: &mut <I::Backend as Backend>::Device,
        extent: Extent2D,
        render_pass: &<I::Backend as Backend>::RenderPass,
    ) -> Result<
        (
            Vec<<I::Backend as Backend>::DescriptorSetLayout>,
            <I::Backend as Backend>::DescriptorPool,
            <I::Backend as Backend>::PipelineLayout,
            <I::Backend as Backend>::GraphicsPipeline,
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

        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(
                    1,
                    &[
                        DescriptorRangeDesc {
                            ty: DescriptorType::SampledImage,
                            count: 2,
                        },
                        DescriptorRangeDesc {
                            ty: DescriptorType::Sampler,
                            count: 2,
                        },
                    ],
                    gfx_hal::pso::DescriptorPoolCreateFlags::empty(),
                )
                .map_err(|_| "Couldn't create a descriptor pool!")?
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
            layout,
            gfx_pipeline,
        ))
    }

    fn record_textures<'a>(
        &mut self,
        sprite_info: &'static [(SpriteName, &'static [u8])],
    ) -> Result<Vec<Sprite>, &'static str> {
        let mut ret = vec![];
        for (name, file) in sprite_info {
            let texture = unsafe {
                let descriptor_set = self
                    .descriptor_pool
                    .allocate_set(&self.descriptor_set_layouts[0])
                    .map_err(|_| "Couldn't make a Descriptor Set!")?;

                println!("Made one descriptor set!");

                LoadedImage::new(
                    &self._adapter,
                    &*self.device,
                    &mut self.command_pool,
                    &mut self.queue_group.queues[0],
                    image::load_from_memory(file)
                        .expect("Binary Corrupted!")
                        .to_rgba(),
                    descriptor_set,
                )?
            };
            let len = self.textures.len();
            self.textures.push(texture);
            ret.push(Sprite {
                name,
                file,
                texture: len,
            });
        }

        Ok(ret)
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

    pub fn draw_quad_frame(&mut self, quad: Quad, sprite: &Sprite) -> Result<Option<Suboptimal>, &'static str> {
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
                    Some(self.textures[sprite.texture].descriptor_set.deref()),
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

impl<I: Instance> core::ops::Drop for Renderer<I> {
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
            for texture in self.textures.drain(..) {
                texture.manually_drop(&self.device);
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
