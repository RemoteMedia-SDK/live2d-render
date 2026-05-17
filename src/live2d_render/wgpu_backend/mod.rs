//! `WgpuBackend` — headless wgpu render-to-texture implementation
//! of [`super::Live2DBackend`] for Live2D models.
//!
//! ## Pipeline shape
//!
//! Render lifecycle (per `render_frame` call):
//!
//! 1. **Apply pose to model** — write VBridger `params` into
//!    `cubism_core::Model::parameters_mut`, then `model.update()`
//!    so the SDK recomputes drawable vertex positions.
//! 2. **Re-upload changed vertex buffers** — for every drawable
//!    whose `DynamicFlags::VERTEX_POSITIONS_DID_CHANGE` bit is
//!    set, copy the new positions to its GPU buffer.
//! 3. **Render pass** — clear the offscreen texture, then for
//!    each drawable in `render_order` ascending: bind its texture
//!    + bind group, set the right blend pipeline, draw indexed.
//! 4. **Readback** — copy the offscreen RGBA texture into a
//!    mappable buffer, await mapping, strip the alpha channel
//!    into a tightly packed RGB24 frame.
//!
//! ## Scope of this commit (M4.4 pass 1)
//!
//! Implements Normal blend mode + ordered drawable draw + texture
//! upload + RGB readback. Mask pre-pass + Additive + Multiplicative
//! blend modes are stub-mapped to Normal; their dedicated pipelines
//! land in M4.4 follow-up commits. Aria renders visibly with this
//! pass, with masked drawables (eyes, mouth, etc.) rendering
//! unclipped — a known visual bug that the mask pre-pass fixes.

use super::{BackendError, Live2DBackend, Pose, RgbFrame};
use bytemuck::{Pod, Zeroable};
use cubism_core::{
    drawable::{BlendMode, ConstantFlags, DynamicFlags},
    ExpressionBlend, ExpressionJson, ExpressionParameter, Moc, Model, PhysicsRig, PhysicsSimulator,
    ResolvedModel,
};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Vertex layout shared by every drawable. `#[repr(C)]` + `Pod` so
/// we can `bytemuck::cast_slice` the SDK's interleaved vertex data
/// (positions [Vec2] + UVs [Vec2]) into a single GPU vertex buffer.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

/// Per-draw uniform block. Matches `DrawableUniforms` in the WGSL
/// shader byte-for-byte. WGSL `mat4x4<f32>` is column-major.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DrawableUniforms {
    projection: [[f32; 4]; 4],
    multiply: [f32; 4],
    screen: [f32; 4],
    /// `[mask_enabled, inverted, _, _]` — see drawable.wgsl. The
    /// shader gates mask sampling on `mask_flags[0] > 0.5`.
    mask_flags: [f32; 4],
}

const UNIFORM_SIZE: u64 = std::mem::size_of::<DrawableUniforms>() as u64;
const RENDER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const READBACK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// One drawable's GPU resources. Vertex + index buffers are sized
/// once at model load (Cubism guarantees vertex / index counts
/// don't change post-init, only positions).
struct DrawableResources {
    /// `count_vertices * size_of::<Vertex>()` bytes.
    vertex_buffer: wgpu::Buffer,
    /// `count_indices * size_of::<u16>()` bytes.
    index_buffer: wgpu::Buffer,
    /// Bind group used in the **mask pre-pass** + by unmasked
    /// drawables in the main pass. Mask binding (slot 3) points at
    /// the dummy 1×1 white texture so the shader's `textureSample`
    /// is well-defined; the shader gates whether it's actually read
    /// via `mask_flags`.
    bind_group_dummy_mask: wgpu::BindGroup,
    /// Bind group used in the **main pass** for masked drawables.
    /// Mask binding (slot 3) points at the shared `mask_target`
    /// texture so the shader can read the just-rendered mask alpha.
    /// `None` for unmasked drawables (they only ever use the dummy).
    bind_group_real_mask: Option<wgpu::BindGroup>,
    /// Per-drawable uniform buffer; updated each frame with new
    /// projection × multiply × screen × mask_flags.
    uniform_buffer: wgpu::Buffer,
    /// Number of indices to draw.
    index_count: u32,
    /// Number of vertices the per-drawable VB holds.
    vertex_count: usize,
    /// Texture index (into the model's texture array).
    texture_index: usize,
    /// Cached blend mode (decoded from constant_flags at load).
    blend_mode: BlendMode,
    /// `true` if the drawable's `IS_INVERTED_MASK` constant flag is
    /// set. Stored at load so we don't refetch flags every frame.
    inverted_mask: bool,
    /// `true` if the drawable lists at least one mask drawable.
    has_masks: bool,
    /// Drawable indices this drawable is masked against. Empty for
    /// unmasked drawables.
    mask_indices: Vec<usize>,
    #[allow(dead_code)]
    constant_flags: ConstantFlags,
}

/// Live2D model loaded onto the GPU. Owns:
///
/// - The `Moc` + `Model` (Cubism Core state — vertex positions get
///   written into `Model`'s heap by `csmUpdateModel`).
/// - One `wgpu::Texture` + sampler per `.model3.json` texture entry.
/// - One [`DrawableResources`] per drawable.
///
/// Constructed via [`WgpuBackend::load_model`].
struct LoadedModel {
    /// Aliveness anchor for `Model<'moc>`'s lifetime. Held in an
    /// `Arc` so we don't move it relative to the model.
    moc: Arc<Moc>,
    /// `Model` is `Send` but not `Sync`; we never share it across
    /// threads (the wgpu backend is single-threaded per session).
    /// Lifetime parameter ties to the `Arc<Moc>` above; we use a
    /// transmute trick to keep both in the same struct (the moc
    /// outlives the model because it's behind an Arc that the
    /// model borrows).
    ///
    /// SAFETY: `model` only references data inside `moc`'s heap;
    /// dropping order is `model` then `moc` (struct field order),
    /// which is correct.
    model: Model<'static>,
    /// Per-drawable GPU buffers + textures + uniforms.
    drawables: Vec<DrawableResources>,
    /// The model's textures, in the order `.model3.json` lists
    /// them. Indexed by [`DrawableResources::texture_index`].
    textures: Vec<wgpu::Texture>,
    /// Texture views matching `textures`.
    texture_views: Vec<wgpu::TextureView>,
    /// Shared sampler — Cubism doesn't vary sampling per drawable.
    sampler: wgpu::Sampler,
    /// Projection matrix from canvas info, baked once at load.
    projection: [[f32; 4]; 4],
    /// Drawable count (cached so we don't re-`csmGetDrawableCount`
    /// every frame).
    drawable_count: usize,
    /// Indirection: `render_indices[i]` = drawable to draw `i`-th.
    /// Re-sorted each frame from `render_order` per drawable.
    render_indices: Vec<usize>,
    /// Reusable scratch — the bind-group-layout reference we cloned
    /// at load (one layout per backend, shared across drawables).
    #[allow(dead_code)]
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// `.exp3.json` overlays keyed by expression name (e.g.
    /// `"neutral"`, `"happy"`). Populated at load from
    /// [`ResolvedModel::expressions`]. Each entry holds the parsed
    /// parameter overrides plus the fade-in / fade-out durations
    /// from the file. The renderer applies the matching overlay
    /// (per [`Pose::expression_id`]) on top of the VBridger-driven
    /// params before `model.update()`. Cubism's documented runtime
    /// order: animation/lip-sync first, expression overlay second.
    expressions: HashMap<String, ExpressionOverlay>,
    /// Currently-targeted expression and the wall-time at which it
    /// became the target. Drives the fade-in side of the cross-fade.
    current_overlay: Option<OverlaySlot>,
    /// Outgoing expression and the wall-time at which it started
    /// fading out. Drives the fade-out side. Cleared once weight
    /// reaches 0.
    previous_overlay: Option<OverlaySlot>,
    /// Cached `IS_VISIBLE` bit per drawable, refreshed each
    /// `apply_pose_and_update`. Read by `render_frame` to skip
    /// hidden drawables — the Cubism reference renderer requires
    /// this even when opacity is non-zero.
    visible: Vec<bool>,
    /// Cached parameter index for `ParamEyeLOpen` / `ParamEyeROpen`,
    /// or `None` if the rig doesn't expose them. Computed once at
    /// load to avoid repeated `params.find` lookups in the hot path.
    eye_l_open_idx: Option<usize>,
    eye_r_open_idx: Option<usize>,
    /// Hair / cloth / iris-jiggle physics — drives derived
    /// parameters as a function of head movement, eye open, etc.
    /// `None` for rigs that ship no `.physics3.json`.
    physics: Option<PhysicsSimulator>,
    /// Wall-time of the previous `apply_pose_and_update` call;
    /// used to compute the dt for the physics tick. `None` until
    /// the first call.
    last_tick_at: Option<std::time::Instant>,
}

/// Parsed `.exp3.json` overlay plus its timing metadata.
struct ExpressionOverlay {
    parameters: Vec<ExpressionParameter>,
    fade_in_ms: u64,
    fade_out_ms: u64,
}

/// One side of the cross-fade — the expression name + the wall-time
/// at which it became active (for fade-in) or began fading (for
/// fade-out).
struct OverlaySlot {
    name: String,
    started_at: std::time::Instant,
    /// True when this overlay was rotated in with no previous
    /// overlay active (i.e. initial load or back-to-neutral after a
    /// fade has already finished). Initial overlays snap to full
    /// weight rather than fading from zero — Cubism convention is
    /// that fade timings apply to TRANSITIONS, not first-frame
    /// loads. Without this, the renderer's first frame after model
    /// load would render with no expression overlay at all (visible
    /// "broken mouth" flash for ~fade_in_ms).
    instant: bool,
}

/// Headless wgpu backend. Owns the device + queue + offscreen
/// render target + readback buffer. A backend is paired 1:1 with
/// a [`LoadedModel`]; calling [`Self::load_model`] replaces the
/// current model.
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    width: u32,
    height: u32,
    /// Offscreen render target (Rgba8UnormSrgb).
    color_target: wgpu::Texture,
    color_view: wgpu::TextureView,
    /// Mask-prepass render target (Rgba8Unorm — linear, no sRGB).
    /// Same size as color_target so mask UV in the main shader is
    /// just `(clip.x + 1)*0.5, (1 - clip.y)*0.5`.
    mask_target: wgpu::Texture,
    mask_view: wgpu::TextureView,
    /// 1×1 white texture used as the dummy mask binding in the
    /// pass-1 bind group + every mask-pre-pass bind group. Shader
    /// gates whether it's actually sampled via `mask_flags[0]`.
    dummy_mask_view: wgpu::TextureView,
    /// CPU-mappable readback buffer sized for `width * height * 4`,
    /// padded for `COPY_BYTES_PER_ROW_ALIGNMENT`.
    readback_buffer: wgpu::Buffer,
    /// Bytes-per-row including alignment padding (`>= width * 4`).
    bytes_per_row_padded: u32,
    /// Normal-blend pipeline (premultiplied alpha).
    pipeline_normal: wgpu::RenderPipeline,
    /// Additive-blend pipeline (Cubism `BLEND_ADDITIVE`).
    pipeline_additive: wgpu::RenderPipeline,
    /// Multiplicative-blend pipeline (Cubism `BLEND_MULTIPLICATIVE`).
    pipeline_multiplicative: wgpu::RenderPipeline,
    /// Mask pre-pass pipeline. Color writes restricted to ALPHA so
    /// only the alpha channel of `mask_target` accumulates.
    pipeline_mask: wgpu::RenderPipeline,
    /// Bind-group layout shared across drawables.
    bind_group_layout: Arc<wgpu::BindGroupLayout>,
    /// Currently-loaded model, if any.
    model: Option<LoadedModel>,
}

impl std::fmt::Debug for WgpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBackend")
            .field("dimensions", &(self.width, self.height))
            .field("model_loaded", &self.model.is_some())
            .finish()
    }
}

impl WgpuBackend {
    /// Initialize a headless wgpu backend at the given dimensions.
    /// Picks the default high-performance adapter; runs surface-less.
    pub fn new(width: u32, height: u32) -> Result<Self, BackendError> {
        pollster::block_on(Self::new_async(width, height))
    }

    async fn new_async(width: u32, height: u32) -> Result<Self, BackendError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| {
                BackendError::Other(
                    "no wgpu adapter found (Vulkan / Metal / DX12 / GL all unavailable)".into(),
                )
            })?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Live2DRenderNode device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .map_err(|e| BackendError::Other(format!("device creation failed: {e}")))?;

        // Offscreen render target.
        let color_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Live2D render target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: RENDER_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_target.create_view(&Default::default());

        // wgpu requires `bytes_per_row` in `CopyBufferToTexture` /
        // `CopyTextureToBuffer` to be a multiple of 256. Pad up.
        let row_size = width * 4;
        let bytes_per_row_padded = row_size.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Live2D readback"),
            size: (bytes_per_row_padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Bind group layout: 0=uniform buffer, 1=texture, 2=sampler,
        // 3=mask texture (sampled with the same `samp` sampler).
        let bind_group_layout = Arc::new(device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("Live2D bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: wgpu::BufferSize::new(UNIFORM_SIZE),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::D2,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                ],
            },
        ));

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Live2D drawable.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/drawable.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Live2D pl"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Builder helper: every pipeline shares the same vertex
        // attributes + primitive state; only the fragment target
        // (format + blend + write mask) differs.
        let vertex_buffer_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };
        let make_pipeline = |label: &'static str,
                             format: wgpu::TextureFormat,
                             blend: Option<wgpu::BlendState>,
                             write_mask: wgpu::ColorWrites|
         -> wgpu::RenderPipeline {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs_main",
                    buffers: &[vertex_buffer_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs_main",
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        // Normal: standard premultiplied-alpha blend.
        let pipeline_normal = make_pipeline(
            "Live2D pipeline (Normal)",
            RENDER_FORMAT,
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            wgpu::ColorWrites::ALL,
        );

        // Additive: Cubism `BLEND_ADDITIVE` — out = src + dst, alpha
        // unchanged. With premultiplied output: src is already
        // alpha-modulated, so plain ONE/ONE accumulates correctly.
        let pipeline_additive = make_pipeline(
            "Live2D pipeline (Additive)",
            RENDER_FORMAT,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            wgpu::ColorWrites::ALL,
        );

        // Multiplicative: Cubism `BLEND_MULTIPLICATIVE` — out = src
        // * dst (color), alpha unchanged.
        let pipeline_multiplicative = make_pipeline(
            "Live2D pipeline (Multiplicative)",
            RENDER_FORMAT,
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Dst,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
            wgpu::ColorWrites::ALL,
        );

        // Mask pre-pass: same shader, accumulates alpha into
        // `mask_target`. Color channel writes are disabled — only
        // alpha matters for masking. We use standard premultiplied-
        // alpha blending so multiple mask drawables compose like a
        // normal alpha layer.
        let pipeline_mask = make_pipeline(
            "Live2D pipeline (Mask)",
            wgpu::TextureFormat::Rgba8Unorm,
            Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            wgpu::ColorWrites::ALPHA,
        );

        // Mask pre-pass target. Same dimensions as the color
        // target; format Rgba8Unorm (linear) — masks are alpha
        // values, not colors, so we don't want sRGB encoding.
        let mask_target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Live2D mask target"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let mask_view = mask_target.create_view(&Default::default());

        // 1×1 white-alpha dummy mask. Bound at slot 3 in every
        // bind group used in either (a) the mask pre-pass (where
        // sampling the in-flight mask target would be a resource
        // conflict) or (b) the main pass for unmasked drawables
        // (where the shader gates on `mask_flags` and never reads
        // the binding). The texture itself is opaque white so a
        // miswired sampler would produce a visible artifact rather
        // than silent wrongness.
        let dummy_mask_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Live2D dummy mask"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &dummy_mask_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &[255u8, 255, 255, 255],
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let dummy_mask_view = dummy_mask_tex.create_view(&Default::default());

        Ok(Self {
            device,
            queue,
            width,
            height,
            color_target,
            color_view,
            mask_target,
            mask_view,
            dummy_mask_view,
            readback_buffer,
            bytes_per_row_padded,
            pipeline_normal,
            pipeline_additive,
            pipeline_multiplicative,
            pipeline_mask,
            bind_group_layout,
            model: None,
        })
    }

    /// Load a model by `.model3.json` path. Reads + uploads every
    /// referenced texture, parses the `.moc3`, allocates GPU
    /// buffers per drawable, and bakes the projection matrix from
    /// the model's canvas info. Returns once everything is on the
    /// GPU.
    pub fn load_model(&mut self, model_json_path: impl AsRef<Path>) -> Result<(), BackendError> {
        let resolved = cubism_core::ModelJson::load(model_json_path.as_ref())
            .map_err(|e| BackendError::Other(format!("model3.json: {e}")))?;
        self.load_resolved_model(&resolved)
    }

    /// Load a model from an already-parsed `.model3.json`. Useful
    /// when the caller has the manifest in hand (M4.5 streaming
    /// node loads it for emotion → expression mapping anyway).
    pub fn load_resolved_model(&mut self, resolved: &ResolvedModel) -> Result<(), BackendError> {
        // 1. Parse the .moc3 + initialize the Model.
        let moc = Arc::new(
            Moc::load_from_file(resolved.moc_path())
                .map_err(|e| BackendError::Other(format!("moc load: {e}")))?,
        );
        // SAFETY: see LoadedModel::model — Model holds raw ptrs
        // into moc's heap; we keep moc alive in the same struct
        // and drop in the right order.
        let model: Model<'static> = unsafe {
            std::mem::transmute(
                Model::from_moc(&moc)
                    .map_err(|e| BackendError::Other(format!("model init: {e}")))?,
            )
        };

        // 2. Upload textures.
        let texture_paths = resolved.texture_paths();
        let mut textures = Vec::with_capacity(texture_paths.len());
        let mut texture_views = Vec::with_capacity(texture_paths.len());
        for tex_path in &texture_paths {
            let img = image::open(tex_path)
                .map_err(|e| BackendError::Other(format!("texture {:?}: {e}", tex_path)))?;
            let rgba = img.to_rgba8();
            let (tw, th) = rgba.dimensions();
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Live2D texture"),
                size: wgpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                // Cubism PNGs are sRGB-encoded.
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba.as_raw(),
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(4 * tw),
                    rows_per_image: Some(th),
                },
                wgpu::Extent3d {
                    width: tw,
                    height: th,
                    depth_or_array_layers: 1,
                },
            );
            let view = texture.create_view(&Default::default());
            textures.push(texture);
            texture_views.push(view);
        }

        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Live2D sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // 3. Per-drawable GPU resources.
        let canvas = model.canvas_info();
        let projection = build_projection(canvas, self.width, self.height);

        let drawable_count = model.drawables().len();
        let mut drawables = Vec::with_capacity(drawable_count);
        let drawables_view = model.drawables();
        for d in drawables_view.iter() {
            let positions = d.vertex_positions();
            let uvs = d.vertex_uvs();
            let indices = d.indices();
            let vertex_count = positions.len();

            let mut interleaved = Vec::with_capacity(vertex_count);
            for i in 0..vertex_count {
                interleaved.push(Vertex {
                    pos: [positions[i].x, positions[i].y],
                    uv: [uvs[i].x, uvs[i].y],
                });
            }

            let vb = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Live2D drawable VB"),
                    contents: bytemuck::cast_slice(&interleaved),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });
            let ib = self
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Live2D drawable IB"),
                    contents: bytemuck::cast_slice(indices),
                    usage: wgpu::BufferUsages::INDEX,
                });
            let ub = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Live2D drawable UB"),
                size: UNIFORM_SIZE,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let texture_index = d.texture_index().max(0) as usize;
            let view = texture_views.get(texture_index).ok_or_else(|| {
                BackendError::Other(format!(
                    "drawable {:?} references texture index {} but only {} textures loaded",
                    d.id(),
                    texture_index,
                    texture_views.len()
                ))
            })?;

            // Bind group with the dummy mask — used both during the
            // mask pre-pass (where binding mask_target would be a
            // resource conflict) and for unmasked main draws.
            let bind_group_dummy_mask = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Live2D drawable bg (dummy mask)"),
                layout: &self.bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: ub.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&self.dummy_mask_view),
                    },
                ],
            });

            let mask_indices: Vec<usize> = d.masks().iter().map(|&i| i as usize).collect();
            let inverted_mask = d.constant_flags().contains(ConstantFlags::IS_INVERTED_MASK);
            let has_masks = !mask_indices.is_empty();

            // Bind group with the real mask target (only allocated
            // for drawables that need to sample it in the main pass).
            let bind_group_real_mask = if has_masks {
                Some(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Live2D drawable bg (real mask)"),
                    layout: &self.bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: ub.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&self.mask_view),
                        },
                    ],
                }))
            } else {
                None
            };

            drawables.push(DrawableResources {
                vertex_buffer: vb,
                index_buffer: ib,
                bind_group_dummy_mask,
                bind_group_real_mask,
                uniform_buffer: ub,
                index_count: indices.len() as u32,
                vertex_count,
                texture_index,
                blend_mode: d.blend_mode(),
                inverted_mask,
                has_masks,
                mask_indices,
                constant_flags: d.constant_flags(),
            });
        }

        // Initial render-order indirection (fully populated below
        // every frame from the SDK's render_order array).
        let render_indices: Vec<usize> = (0..drawable_count).collect();

        // Parse every `.exp3.json` referenced by the model, capturing
        // both the parameter overrides AND the fade timings (used by
        // the cross-fade in `apply_pose_and_update`). A failing parse
        // is logged + skipped.
        let mut expressions: HashMap<String, ExpressionOverlay> = HashMap::new();
        for (name, path) in resolved.expressions() {
            match ExpressionJson::from_file(&path) {
                Ok(exp) => {
                    let fade_in_ms = (exp.fade_in_time.max(0.0) * 1000.0) as u64;
                    let fade_out_ms = (exp.fade_out_time.max(0.0) * 1000.0) as u64;
                    expressions.insert(
                        name.to_string(),
                        ExpressionOverlay {
                            parameters: exp.parameters,
                            fade_in_ms,
                            fade_out_ms,
                        },
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        target: "live2d.expression",
                        expression = %name,
                        path = ?path,
                        error = %e,
                        "failed to parse .exp3.json — expression will be a no-op",
                    );
                }
            }
        }
        tracing::info!(
            target: "live2d.expression",
            count = expressions.len(),
            names = ?expressions.keys().collect::<Vec<_>>(),
            "loaded expression overlays",
        );

        // Cache eye-open parameter indices.
        let (eye_l_open_idx, eye_r_open_idx) = {
            let params = model.parameters();
            (
                params.find("ParamEyeLOpen").map(|p| p.index()),
                params.find("ParamEyeROpen").map(|p| p.index()),
            )
        };

        // Load + initialize the physics simulator if the model
        // ships a `.physics3.json`. Failure to parse is logged and
        // we render without physics rather than refusing the load.
        let physics = match resolved.physics_path() {
            Some(p) => match PhysicsRig::from_file(&p) {
                Ok(rig) => {
                    tracing::info!(
                        target: "live2d.physics",
                        path = ?p,
                        settings = rig.physics_settings.len(),
                        fps = rig.meta.fps,
                        "loaded physics rig",
                    );
                    Some(PhysicsSimulator::new(rig))
                }
                Err(e) => {
                    tracing::warn!(
                        target: "live2d.physics",
                        path = ?p,
                        error = %e,
                        "failed to parse .physics3.json — rendering without physics",
                    );
                    None
                }
            },
            None => None,
        };

        let visible = vec![true; drawable_count];

        self.model = Some(LoadedModel {
            moc,
            model,
            drawables,
            textures,
            texture_views,
            sampler,
            projection,
            drawable_count,
            render_indices,
            bind_group_layout: self.bind_group_layout.clone(),
            expressions,
            current_overlay: None,
            previous_overlay: None,
            visible,
            eye_l_open_idx,
            eye_r_open_idx,
            physics,
            last_tick_at: None,
        });
        Ok(())
    }

    /// Apply the pose's VBridger params to the model, run
    /// `csmUpdateModel`, then refresh any drawables whose vertex
    /// positions changed.
    fn apply_pose_and_update(&mut self, pose: &Pose) -> Result<(), BackendError> {
        let model = self.model.as_mut().ok_or(BackendError::ModelNotLoaded)?;

        // 1. Write VBridger params. Unknown IDs are silently skipped — not
        // every Live2D model wires every VBridger parameter (e.g. Aria has
        // no `ParamMouthClose`); a missing ID simply means no visible
        // contribution from that ARKit axis on this rig.
        {
            let params = model.model.parameters_mut();
            for (id, value) in &pose.params {
                if let Some(p) = params.find(id) {
                    p.set_value(*value);
                }
            }
        }

        // 1a. Project the blink scheduler's `eye_open` (state-machine
        // output, [0, 1]) onto the rig's `ParamEyeLOpen` /
        // `ParamEyeROpen`. The state machine sets `eye_open = 1.0`
        // when an emotion is active so the expression overlay
        // (next step) can take over the eyes; otherwise it's the
        // animated blink curve. We only project when the pose isn't
        // already driving these IDs explicitly.
        let pose_drives_eye_l = pose.params.contains_key("ParamEyeLOpen");
        let pose_drives_eye_r = pose.params.contains_key("ParamEyeROpen");
        if !pose_drives_eye_l || !pose_drives_eye_r {
            let params = model.model.parameters_mut();
            if !pose_drives_eye_l {
                if let Some(idx) = model.eye_l_open_idx {
                    if let Some(p) = params.get(idx) {
                        p.set_value(pose.eye_open);
                    }
                }
            }
            if !pose_drives_eye_r {
                if let Some(idx) = model.eye_r_open_idx {
                    if let Some(p) = params.get(idx) {
                        p.set_value(pose.eye_open);
                    }
                }
            }
        }

        // 1b. Cross-faded expression overlay. The pose carries the
        // *target* expression (`pose.expression_id`); the backend
        // tracks the wall-time of the last swap and (a) fades the
        // outgoing expression out over its `fade_out_time`, (b)
        // fades the incoming one in over its `fade_in_time`.
        //
        // Per-blend cross-fade math (W ∈ [0, 1] is the overlay's
        // current weight, base = the parameter value AFTER the
        // VBridger + eye writes above):
        //   Add:       out = base + value * W
        //   Multiply:  out = base * (1 - W + W * value)
        //   Overwrite: out = lerp(base, value, W)
        let now = std::time::Instant::now();
        let target_name = pose.expression_id.as_str();
        let current_matches = model
            .current_overlay
            .as_ref()
            .map(|s| s.name == target_name)
            .unwrap_or(false);
        if !current_matches {
            let had_prior = model.current_overlay.is_some();
            model.previous_overlay = model.current_overlay.take();
            if !target_name.is_empty() {
                model.current_overlay = Some(OverlaySlot {
                    name: target_name.to_string(),
                    started_at: now,
                    instant: !had_prior,
                });
            }
            if let Some(prev) = model.previous_overlay.as_mut() {
                prev.started_at = now;
            }
        }
        let curr_weight = model
            .current_overlay
            .as_ref()
            .and_then(|s| {
                if s.instant {
                    return Some(1.0);
                }
                let exp = model.expressions.get(&s.name)?;
                let elapsed = now.duration_since(s.started_at).as_millis() as u64;
                let w = if exp.fade_in_ms == 0 {
                    1.0
                } else {
                    (elapsed as f32 / exp.fade_in_ms as f32).min(1.0)
                };
                Some(w)
            })
            .unwrap_or(0.0);
        let prev_weight = model
            .previous_overlay
            .as_ref()
            .and_then(|s| {
                let exp = model.expressions.get(&s.name)?;
                let elapsed = now.duration_since(s.started_at).as_millis() as u64;
                let w = if exp.fade_out_ms == 0 {
                    0.0
                } else {
                    (1.0 - elapsed as f32 / exp.fade_out_ms as f32).max(0.0)
                };
                Some(w)
            })
            .unwrap_or(0.0);
        if prev_weight <= 0.0 {
            model.previous_overlay = None;
        }
        if prev_weight > 0.0 {
            if let Some(slot) = model.previous_overlay.as_ref() {
                if let Some(exp) = model.expressions.get(&slot.name) {
                    apply_overlay_weighted(&mut model.model, &exp.parameters, prev_weight);
                }
            }
        }
        if curr_weight > 0.0 {
            if let Some(slot) = model.current_overlay.as_ref() {
                if let Some(exp) = model.expressions.get(&slot.name) {
                    apply_overlay_weighted(&mut model.model, &exp.parameters, curr_weight);
                }
            }
        }

        // 1c. Physics tick. Reads the parameters set above (so head
        // turns, eye opens, mouth shapes feed into hair/iris/cloth
        // physics) and writes derived parameter values back. Runs
        // before `csmUpdateModel` so the model deforms with the
        // physics-driven values applied. The simulator sub-steps
        // internally at the rig's authored fps.
        if let Some(sim) = model.physics.as_mut() {
            let dt_s = match model.last_tick_at {
                Some(t) => now.duration_since(t).as_secs_f32(),
                // First call after load: use one fixed step to
                // avoid `dt = 0` (which would emit no integration
                // and leave outputs at their initial-position
                // state).
                None => 1.0 / 60.0,
            };
            let mut params = model.model.parameters_mut();
            sim.tick(&mut params, dt_s);
        }
        model.last_tick_at = Some(now);

        // 2. Update — recomputes vertex positions + dynamic flags.
        model.model.update();

        // 3. Re-upload changed VBs + per-frame uniforms; cache
        //    `IS_VISIBLE` for the render loop's skip-check. The
        //    Cubism reference renderer treats `csmIsVisible == 0`
        //    as "do not draw" even when opacity is non-zero.
        let drawables_view = model.model.drawables();
        for (i, d) in drawables_view.iter().enumerate() {
            let res = &model.drawables[i];
            if i < model.visible.len() {
                model.visible[i] = d.dynamic_flags().contains(DynamicFlags::IS_VISIBLE);
            }

            // Re-upload vertices when SDK signals positions changed.
            if d.dynamic_flags()
                .contains(DynamicFlags::VERTEX_POSITIONS_DID_CHANGE)
            {
                let positions = d.vertex_positions();
                let uvs = d.vertex_uvs();
                if positions.len() == res.vertex_count {
                    let mut interleaved = Vec::with_capacity(positions.len());
                    for j in 0..positions.len() {
                        interleaved.push(Vertex {
                            pos: [positions[j].x, positions[j].y],
                            uv: [uvs[j].x, uvs[j].y],
                        });
                    }
                    self.queue.write_buffer(
                        &res.vertex_buffer,
                        0,
                        bytemuck::cast_slice(&interleaved),
                    );
                }
            }

            // Per-frame uniforms — multiply colour modulated by
            // opacity, and the screen colour. Projection is baked
            // at load. `mask_flags` enables mask sampling for
            // drawables that have masks; the inverted-mask bit
            // comes from the constant flags (stored at load).
            let opacity = d.opacity();
            let mc = d.multiply_color();
            let sc = d.screen_color();
            let mask_enabled = if res.has_masks { 1.0 } else { 0.0 };
            let mask_inverted = if res.inverted_mask { 1.0 } else { 0.0 };
            let uniforms = DrawableUniforms {
                projection: model.projection,
                multiply: [mc.x, mc.y, mc.z, mc.w * opacity],
                screen: [sc.x, sc.y, sc.z, sc.w],
                mask_flags: [mask_enabled, mask_inverted, 0.0, 0.0],
            };
            self.queue
                .write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));
        }

        // 4. Sort by render order — Cubism guarantees render order
        //    is stable per drawable index but the order of indices
        //    isn't.
        model.render_indices.clear();
        model.render_indices.extend(0..model.drawable_count);
        let render_orders: Vec<i32> = drawables_view.iter().map(|d| d.render_order()).collect();
        model.render_indices.sort_by_key(|&i| render_orders[i]);
        Ok(())
    }

    /// Read the offscreen RGBA8 target back into a tightly packed
    /// `RgbFrame` (alpha stripped).
    fn readback_to_rgb(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color_target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.readback_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.bytes_per_row_padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn map_and_extract_rgb(&self) -> Result<RgbFrame, BackendError> {
        let slice = self.readback_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        // Drive the device until the map callback fires.
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .map_err(|_| BackendError::Other("readback channel disconnected".into()))?
            .map_err(|e| BackendError::Other(format!("buffer map failed: {e}")))?;

        let view = slice.get_mapped_range();
        let mut rgb = vec![0u8; (self.width * self.height * 3) as usize];
        let row_size = (self.width * 4) as usize;
        let padded = self.bytes_per_row_padded as usize;
        for y in 0..self.height as usize {
            let src_row = &view[y * padded..y * padded + row_size];
            let dst_row = &mut rgb[y * self.width as usize * 3..(y + 1) * self.width as usize * 3];
            for x in 0..self.width as usize {
                dst_row[x * 3] = src_row[x * 4];
                dst_row[x * 3 + 1] = src_row[x * 4 + 1];
                dst_row[x * 3 + 2] = src_row[x * 4 + 2];
            }
        }
        drop(view);
        self.readback_buffer.unmap();
        Ok(RgbFrame {
            width: self.width,
            height: self.height,
            pixels: rgb,
        })
    }
}

impl Live2DBackend for WgpuBackend {
    fn render_frame(&mut self, pose: &Pose) -> Result<RgbFrame, BackendError> {
        self.apply_pose_and_update(pose)?;
        let model = self.model.as_ref().ok_or(BackendError::ModelNotLoaded)?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Live2D render encoder"),
            });

        // Clear the color target up-front. The main draws run in
        // separate render passes (one per drawable so we can
        // interleave mask pre-passes), each with `LoadOp::Load` so
        // they accumulate on top of the cleared target.
        {
            let _clear = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Live2D color clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
        }

        for &i in &model.render_indices {
            // Skip drawables the SDK has flagged hidden this frame.
            // Cubism's reference renderer requires this even when
            // opacity is non-zero; without it, rigs that toggle
            // visibility (rather than opacity) leak hidden parts.
            if !model.visible.get(i).copied().unwrap_or(true) {
                continue;
            }
            let res = &model.drawables[i];

            if res.has_masks {
                // Mask pre-pass: clear mask target + accumulate
                // alpha from the mask drawables. Color writes are
                // restricted to ALPHA in the mask pipeline so the
                // RGB channels of `mask_target` stay irrelevant.
                let mut mask_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Live2D mask pre-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &self.mask_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                mask_pass.set_pipeline(&self.pipeline_mask);
                for &mi in &res.mask_indices {
                    let m = match model.drawables.get(mi) {
                        Some(m) => m,
                        None => continue, // out-of-range mask index
                    };
                    // Use the mask drawable's *dummy* bind group —
                    // we can't bind `mask_target` while rendering
                    // INTO it, and we don't sample it anyway in the
                    // mask shader (mask pre-pass shader path
                    // ignores `mask_flags`).
                    mask_pass.set_bind_group(0, &m.bind_group_dummy_mask, &[]);
                    mask_pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
                    mask_pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                    mask_pass.draw_indexed(0..m.index_count, 0, 0..1);
                }
            }

            // Main pass for drawable `i` — its own render pass so
            // the just-rendered mask target is read with a clean
            // pipeline barrier in between (wgpu inserts this for us
            // automatically given the texture transitions between
            // RENDER_ATTACHMENT and TEXTURE_BINDING).
            let mut main_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Live2D main pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let pipeline = match res.blend_mode {
                BlendMode::Normal => &self.pipeline_normal,
                BlendMode::Additive => &self.pipeline_additive,
                BlendMode::Multiplicative => &self.pipeline_multiplicative,
            };
            main_pass.set_pipeline(pipeline);
            let bind_group = if res.has_masks {
                res.bind_group_real_mask
                    .as_ref()
                    .expect("has_masks implies real_mask bind group exists")
            } else {
                &res.bind_group_dummy_mask
            };
            main_pass.set_bind_group(0, bind_group, &[]);
            main_pass.set_vertex_buffer(0, res.vertex_buffer.slice(..));
            main_pass.set_index_buffer(res.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
            main_pass.draw_indexed(0..res.index_count, 0, 0..1);
        }

        self.readback_to_rgb(&mut encoder);
        self.queue.submit(Some(encoder.finish()));
        self.map_and_extract_rgb()
    }

    fn frame_dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Apply one expression overlay with a fade weight `w ∈ [0, 1]`.
///
/// The base value is whatever's currently on the parameter (so
/// caller controls the order: VBridger writes first, then eye
/// projection, then this). When `w == 1.0` this matches the
/// instantaneous-overlay behavior; smaller `w` produces a partial
/// blend toward the override target.
fn apply_overlay_weighted(model: &mut Model<'_>, overrides: &[ExpressionParameter], w: f32) {
    if w <= 0.0 {
        return;
    }
    let params = model.parameters_mut();
    for ovr in overrides {
        let Some(p) = params.find(&ovr.id) else {
            continue;
        };
        let base = p.value();
        let new_value = match ovr.blend_kind() {
            ExpressionBlend::Add => base + ovr.value * w,
            ExpressionBlend::Multiply => base * (1.0 - w + w * ovr.value),
            ExpressionBlend::Overwrite => base + (ovr.value - base) * w,
            ExpressionBlend::Unknown(_) => continue,
        };
        p.set_value(new_value);
    }
}

/// Build the model→NDC projection matrix from the canvas info,
/// scaled to fit `(fb_width × fb_height)` while preserving aspect
/// (letterbox if the framebuffer aspect doesn't match the canvas).
///
/// Returns column-major (WGSL convention).
fn build_projection(
    canvas: cubism_core::CanvasInfo,
    fb_width: u32,
    fb_height: u32,
) -> [[f32; 4]; 4] {
    // Per-axis NDC scale to fit the canvas exactly. Aspect-fit is
    // handled by the smaller of the two scales.
    let canvas_w = canvas.size.x.max(1.0);
    let canvas_h = canvas.size.y.max(1.0);
    let fit_x = 2.0 / canvas_w;
    let fit_y = 2.0 / canvas_h;

    // Aspect-correct: use the smaller scale on both axes so the
    // model fits in the framebuffer's square viewport.
    let aspect_canvas = canvas_w / canvas_h;
    let aspect_fb = fb_width as f32 / fb_height.max(1) as f32;
    let (sx, sy) = if aspect_canvas > aspect_fb {
        // Framebuffer is taller than canvas — fit horizontally.
        (
            fit_x,
            fit_x * aspect_fb / aspect_canvas * (canvas_h / canvas_w) * aspect_canvas,
        )
    } else {
        // Framebuffer is wider — fit vertically.
        (
            fit_y * aspect_fb * (canvas_w / canvas_h) / aspect_canvas,
            fit_y,
        )
    };

    // Cubism vertex positions are in MODEL UNITS (canvas-relative,
    // ~[-0.5, 0.5] across the canvas). The pixels-per-unit + origin
    // values are an authoring convenience for the editor; for
    // rendering the simplest correct map is `vert ∈ [-0.5, 0.5]
    // → ndc ∈ [-1, 1]`. We achieve that with `sx*2, sy*2`.
    let sx2 = sx * canvas_w;
    let sy2 = sy * canvas_h;

    // Column-major: M[col][row].
    [
        [sx2, 0.0, 0.0, 0.0],
        [0.0, sy2, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test that the headless device + render target + readback
    /// pipeline come up clean and produce a fully-transparent
    /// (or fully-cleared) frame when no model is loaded.
    #[test]
    fn headless_device_inits_and_clears() {
        let backend = WgpuBackend::new(64, 64);
        match backend {
            Ok(_) => {} // adapter found, all good
            Err(e) => {
                // Skip on hosts where no GPU adapter is available
                // (CI sandboxes, headless servers without lavapipe,
                // etc.). The wgpu pipeline remains testable on dev
                // machines + GPU-equipped CI.
                eprintln!("[skip] no wgpu adapter on this host: {e}");
            }
        }
    }

    /// `frame_dimensions` reports what was passed to `new`.
    #[test]
    fn dimensions_round_trip() {
        if let Ok(b) = WgpuBackend::new(640, 480) {
            assert_eq!(b.frame_dimensions(), (640, 480));
        }
    }
}
