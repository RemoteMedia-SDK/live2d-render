//! `Live2DRenderNode` as a standalone Path 3 loadable plugin.
//!
//! Headless wgpu+CubismCore renderer for Live2D Cubism models. Consumes
//! the renderer's four input envelope shapes on its main port:
//!
//! - `{kind: "blendshapes", arkit_52, pts_ms, turn_id?}` — from
//!   `Audio2FaceLipSyncNode` / `SyntheticLipSyncNode`. Pushed into the
//!   state machine's ring; sampled by the pacer's `tick`.
//! - `{kind: "emotion", emoji, …}` — from `EmotionExtractorNode`.
//!   Updates state but does NOT render.
//! - `{kind: "audio_clock", pts_ms, …}` — from the outbound
//!   `AudioSender`'s `audio.out.clock` tap. Anchors the audio
//!   timeline.
//! - `{kind: "barge_in"}` — from coordinator-driven barge. Resets
//!   state; the next blendshape re-anchors the audio timeline.
//!
//! The pacer drives `tick()` at the bound wire clock (typically the
//! outbound audio media clock) and emits `RuntimeData::Video {format:
//! Rgb24, ...}` per tick stamped with the configured `video_stream_id`
//! (default `"avatar"`).
//!
//! Originally lived in `remotemedia-core` under
//! `nodes/live2d_render`; extracted here so the host crate doesn't
//! drag in `wgpu` + `image` + the Cubism SDK build-script dep just
//! for this renderer.
//!
//! ## Node types exported
//!
//!   Live2DRenderNode — blendshape/emotion/audio_clock/barge envelopes
//!                       → Video (Rgb24) at the bound media clock rate.

mod arkit;
mod live2d_render;
mod session_control;

use std::path::PathBuf;

use remotemedia_plugin_sdk::abi_stable::sabi_trait::TD_Opaque;
use remotemedia_plugin_sdk::abi_stable::std_types::{RErr, ROk, RResult, RString};
use remotemedia_plugin_sdk::adapter::StreamingNodeFfiAdapter;
use remotemedia_plugin_sdk::{FfiNodeBox, FfiNodeFactory, FfiNode_TO};

use crate::live2d_render::{
    BackendError, Live2DBackend, Live2DRenderConfig, Live2DRenderNode, StateConfig, WgpuBackend,
};

// ---------------------------------------------------------------------------
// Manifest params
// ---------------------------------------------------------------------------

/// JSON-shaped params accepted by the factory. Mirrors the host's
/// historical wiring: `modelPath`, `framerate`, `videoStreamId`,
/// `width`, `height`.
///
/// `StateConfig` knobs (blink intervals, expression hold, neutral
/// interp, emotion mapping, mapper) are left at the published
/// defaults — the renderer's `StateConfig` carries an
/// `Arc<dyn ArkitToVBridger>` that has no canonical wire shape, so
/// per-knob overrides land in a follow-up.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Live2DRenderManifestParams {
    /// Path to the Live2D `.model3.json` to render. Required for a
    /// non-trivial pipeline; if omitted the wgpu backend still
    /// initializes but `render_frame` returns `ModelNotLoaded`.
    pub model_path: Option<PathBuf>,
    /// Output framerate (frames per second). Defaults to 30 per
    /// spec §6.1.
    pub framerate: u32,
    /// Output `RuntimeData::Video.stream_id`. Defaults to `"avatar"`.
    pub video_stream_id: String,
    /// Render-target width in pixels (default 1280).
    pub width: u32,
    /// Render-target height in pixels (default 720).
    pub height: u32,
}

impl Default for Live2DRenderManifestParams {
    fn default() -> Self {
        Self {
            model_path: None,
            framerate: 30,
            video_stream_id: "avatar".to_string(),
            width: 1280,
            height: 720,
        }
    }
}

// ---------------------------------------------------------------------------
// Factory + plugin registration
// ---------------------------------------------------------------------------

/// Build the wgpu backend + load the model + assemble the node from
/// JSON manifest params. Heavy: initializes a wgpu device, uploads
/// model textures, parses `.exp3.json` overlays, loads physics.
fn build_node(params: &Live2DRenderManifestParams) -> Result<Live2DRenderNode, String> {
    let mut backend = WgpuBackend::new(params.width.max(1), params.height.max(1))
        .map_err(|e: BackendError| format!("wgpu backend init failed: {e}"))?;

    if let Some(path) = params.model_path.as_ref() {
        backend
            .load_model(path)
            .map_err(|e| format!("load_model({path:?}) failed: {e}"))?;
    } else {
        tracing::warn!(
            target: "live2d.factory",
            "Live2DRenderNode constructed with no modelPath — backend has no model loaded; \
             render_frame() will return ModelNotLoaded until a model is wired in."
        );
    }

    let config = Live2DRenderConfig {
        model_path: params.model_path.clone(),
        framerate: params.framerate.max(1),
        video_stream_id: params.video_stream_id.clone(),
        state_config: StateConfig::default_config(),
    };

    let boxed: Box<dyn Live2DBackend + Send> = Box::new(backend);
    Ok(Live2DRenderNode::new_with_backend(boxed, config))
}

#[derive(Default)]
pub struct Live2DRenderNodeFactory;

impl FfiNodeFactory for Live2DRenderNodeFactory {
    fn node_type(&self) -> RString {
        RString::from("Live2DRenderNode")
    }

    fn create(&self, params: RString) -> RResult<FfiNodeBox, RString> {
        // Accepts both snake_case and camelCase manifest keys via
        // plugin-sdk's lenient deserializer.
        let parsed: Live2DRenderManifestParams = match remotemedia_plugin_sdk::params::deserialize_params(
            params.as_str(),
        ) {
            Ok(p) => p,
            Err(e) => {
                return RErr(RString::from(format!(
                    "Live2DRenderNode params parse failed: {e}"
                )));
            }
        };
        let node = match build_node(&parsed) {
            Ok(n) => n,
            Err(e) => return RErr(RString::from(format!("Live2DRenderNode load failed: {e}"))),
        };
        ROk(FfiNode_TO::from_value(
            StreamingNodeFfiAdapter::new(node),
            TD_Opaque,
        ))
    }
}

// Emits the abi_stable root-module symbol for dlopen. Gated behind
// the `plugin-export` cargo feature so the rlib can be linked
// alongside other plugins without duplicate-symbol collisions.
#[cfg(feature = "plugin-export")]
remotemedia_plugin_sdk::plugin_export!(Live2DRenderNodeFactory);
