//! `Live2DBackend` trait — the seam between the input-arbitration
//! state machine (M4.3) and the wgpu+CubismCore renderer (M4.4).
//!
//! The state machine produces a [`super::Pose`] per tick — a flat
//! `HashMap<String, f32>` of VBridger parameter values, the active
//! expression name, the active motion group name, and a few
//! bookkeeping fields. The backend takes that pose and produces a
//! frame of pixels.
//!
//! Two impls land in this codebase:
//! - [`super::MockBackend`] (this milestone, M4.3) — records every
//!   `render_frame` call so input-arbitration assertions can run
//!   without a GPU.
//! - `WgpuBackend` (M4.4) — the real renderer.
//!
//! Future impls (Vulkan / Metal / native window-target) plug in via
//! the same trait.

use super::Pose;

/// One backend-rendered frame's pixels + metadata.
///
/// `pixels` is laid out per the documented `format` — currently
/// always `Rgb24` (R, G, B sequential, no alpha) packed
/// row-major. The wgpu backend produces this via a render-to-
/// texture pass + a CPU readback.
#[derive(Debug, Clone)]
pub struct RgbFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Tightly packed `width * height * 3` RGB bytes.
    pub pixels: Vec<u8>,
}

impl RgbFrame {
    /// Allocate a fresh black frame at `width × height`. Useful
    /// fallback when the backend can't render (model unloaded, GPU
    /// hung) — keeps the video track alive with neutral content.
    pub fn black(width: u32, height: u32) -> Self {
        let pixels = vec![0u8; (width as usize) * (height as usize) * 3];
        Self {
            width,
            height,
            pixels,
        }
    }

    /// Number of non-zero RGB bytes. Cheap sanity check the test
    /// + e2e suites use to assert "non-trivial pixel coverage."
    pub fn nonzero_byte_count(&self) -> usize {
        self.pixels.iter().filter(|&&b| b != 0).count()
    }
}

/// Render-backend trait — one method, one input shape, one output
/// shape. Implementations may be stateful (caching uploaded
/// textures, vertex buffers, GPU resources) but every `render_frame`
/// call is independent.
pub trait Live2DBackend: Send {
    /// Render the current pose into an RGB frame. Returning `Err`
    /// signals the renderer is not in a state to produce a useful
    /// frame; callers fall back to [`RgbFrame::black`] or skip the
    /// emit.
    fn render_frame(&mut self, pose: &Pose) -> Result<RgbFrame, BackendError>;

    /// Frame dimensions. Backends pin these at construction so the
    /// streaming-node wiring (M4.5) can stamp `RuntimeData::Video`
    /// with consistent metadata.
    fn frame_dimensions(&self) -> (u32, u32);
}

/// Backend-side errors. Surfaced as a single category since the
/// state machine doesn't act on them differently — it just logs
/// and re-emits a black frame.
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Backend has no model loaded yet.
    #[error("backend has no Live2D model loaded")]
    ModelNotLoaded,

    /// Backend hit an internal error (GPU, IO, etc.).
    #[error("backend error: {0}")]
    Other(String),
}
