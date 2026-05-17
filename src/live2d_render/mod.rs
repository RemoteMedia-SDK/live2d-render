//! Live2D render node — input arbitration state machine + backend
//! trait. The wgpu+CubismCore backend slots in via [`Live2DBackend`].
//!
//! Per spec [§6.1] the renderer is a free-running 30 fps sampler:
//! it ticks on its own clock, samples the blendshape ring against
//! the audio playback clock, layers an active emotion expression +
//! motion on top, and renders one frame per tick. Input pressure
//! never gates a render.
//!
//! What lives here:
//!
//! - [`Live2DRenderState`] — the state machine. Pure Rust, no GPU.
//! - [`Live2DBackend`] — trait every render backend (wgpu, future
//!   Vulkan/Metal/etc.) implements.
//! - [`StateConfig`], [`EmotionEntry`], [`Pose`] — config + output.
//!
//! The host-side `MockBackend` + `cfg(test)` harness types are NOT
//! ported into this Path-3 plugin — they pulled in crate-internal
//! test scaffolding that doesn't exist outside the workspace.

pub mod backend_trait;
pub mod node;
mod render_worker;
pub mod state;
pub mod wgpu_backend;

pub use backend_trait::{BackendError, Live2DBackend, RgbFrame};
pub use node::{Live2DRenderConfig, Live2DRenderNode};
pub use state::{
    default_emotion_mapping, ArkitToVBridger, EmotionEntry, Live2DRenderState, Pose, StateConfig,
};
pub use wgpu_backend::WgpuBackend;
