//! `Live2DRenderNode` — streaming wire-up for the Live2D renderer.
//!
//! Audio-clock-driven: each input envelope synchronously produces at
//! most one rendered Video frame, stamped with the input's audio-time
//! pts (relative to the first input's pts as t=0). Audio2Face emits
//! 30 blendshapes per second of audio, so the resulting video is
//! 30 fps with frame_N at audio-time `N * 33 ms` — perfectly
//! pts-aligned with the saved `audio.wav`, so `ffmpeg -i video.y4m
//! -i audio.wav out.mp4` muxes them in lip-sync without re-stamping.
//!
//! This module wires the M4.3 state machine + the M4.4 backend trait
//! into an [`AsyncStreamingNode`] that:
//!
//! - Accepts four kinds of `RuntimeData::Json` input on its main port:
//!   - `{kind: "blendshapes", arkit_52, pts_ms, turn_id?}` — from
//!     `Audio2FaceLipSyncNode` / `SyntheticLipSyncNode`. Renders
//!     a frame.
//!   - `{kind: "emotion", emoji, …}` — from `EmotionExtractorNode`.
//!     Updates state but does NOT render.
//!   - `{kind: "audio_clock", pts_ms, …}` — from the WebRTC
//!     `AudioSender`'s `audio.out.clock` tap. Renders a frame.
//!   - `{kind: "barge_in"}` — from coordinator-driven barge. Resets
//!     state; the next blendshape re-anchors the audio timeline.
//!   The barge-in aux-port envelope (M2.6 plumbing) is also handled
//!   via [`AsyncStreamingNode::process_control_message`].
//!
//! - Emits `RuntimeData::Video {pixel_data, width, height, format,
//!   stream_id, …}` synchronously through the runtime callback,
//!   stamped with the configured `video_stream_id` (default
//!   `"avatar"`) and a `timestamp_us` rooted at the first input's
//!   pts as audio-time zero.
//!
//! Idle/blink animation during silence is not emitted — when no
//! blendshape arrives, no Video frame is produced. The state
//! machine's `tick_blink_scheduler` runs on every emit, so blinks
//! fire during the speaking section.

use super::backend_trait::Live2DBackend;
use super::render_worker::RenderWorker;
use super::state::{Live2DRenderState, StateConfig};
use crate::arkit::BlendshapeFrame;
use crate::session_control::{aux_port_of, BARGE_IN_PORT};
use async_trait::async_trait;
use remotemedia_plugin_sdk::traits::streaming::{AsyncStreamingNode, PacingNature, Tick};
use remotemedia_plugin_sdk::types::{Error, PixelFormat, RuntimeData};

type Result<T> = std::result::Result<T, Error>;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex as AsyncMutex;

/// Configuration for [`Live2DRenderNode`].
///
/// Not currently `Serialize`/`Deserialize` because [`StateConfig`]
/// holds an `Arc<dyn ArkitToVBridger>` (the ARKit→VBridger mapper)
/// that doesn't have a single canonical wire shape. M4.6 will wire
/// a factory that builds this from a manifest's flat YAML/JSON
/// fields directly.
#[derive(Debug, Clone)]
pub struct Live2DRenderConfig {
    /// Path to the Live2D `.model3.json` to render. Backend
    /// implementations (e.g. `WgpuBackend::load_model`) consume
    /// this on construction. The state-machine-only path used by
    /// `MockBackend` ignores it.
    pub model_path: Option<PathBuf>,
    /// Output framerate (frames per second). Defaults to 30 per
    /// spec §6.1.
    pub framerate: u32,
    /// Output `RuntimeData::Video.stream_id`. Defaults to `"avatar"`.
    /// Useful for multi-avatar pipelines (one renderer per avatar
    /// with a different `stream_id`).
    pub video_stream_id: String,
    /// State machine knobs forwarded to [`Live2DRenderState`].
    /// Defaults match persona-engine's published values.
    pub state_config: StateConfig,
}

impl Default for Live2DRenderConfig {
    fn default() -> Self {
        Self {
            model_path: None,
            framerate: 30,
            video_stream_id: "avatar".to_string(),
            state_config: StateConfig::default_config(),
        }
    }
}

/// Synchronous, input-driven Live2D render node.
///
/// One incoming envelope → at most one rendered Video frame, forwarded
/// through the runtime callback once the render thread completes.
/// State mutation runs under an `AsyncMutex`; the GPU `render_frame`
/// call runs on a dedicated [`RenderWorker`] OS thread so the
/// blocking `wgpu::Device::poll(Maintain::Wait)` inside the wgpu
/// backend never parks a tokio worker.
pub struct Live2DRenderNode {
    config: Live2DRenderConfig,
    /// State machine + ARKit-time anchor. Behind a mutex because
    /// `process_streaming` takes `&self` (the runtime calls it with
    /// a shared borrow) but we need exclusive access to the state
    /// machine's mutators. The lock is released before dispatching
    /// to the render worker so concurrent calls can pipeline state
    /// updates against an in-flight GPU render.
    inner: Arc<AsyncMutex<RendererInner>>,
    /// Backend lives on a dedicated OS thread; calls dispatch over
    /// an `mpsc` channel and `await` the result via `oneshot`.
    render_worker: Arc<RenderWorker>,
    /// Frame counter for the emitted Video frames.
    frame_counter: Arc<AtomicU64>,
}

struct RendererInner {
    state: Live2DRenderState,
    /// First blendshape (or AudioClock) pts becomes audio-time zero.
    /// All subsequent video pts are relative to this so video.y4m
    /// starts at frame 0 = audio sample 0 of the saved audio.wav.
    anchor_pts_ms: Option<u64>,
    /// Most recent blendshape `pts_ms` pushed to the ring (from
    /// audio2face). Used purely for the `lipsync.delta` telemetry log.
    latest_blendshape_pts_ms: Option<u64>,
    /// Most recent `audio_clock` pts received from the AudioSender's
    /// clock tap. Used purely for the `lipsync.delta` telemetry log.
    /// `None` means the WebRTC `set_clock_tap` wiring isn't installed
    /// for this peer, in which case we fall back to the simulated
    /// wall-clock below so `compute_pose` still has a valid clock.
    latest_audio_clock_ms: Option<u64>,
    /// Wall-time of the last `lipsync.delta` emission. Throttles the
    /// log to once per second so the output stays readable.
    last_delta_log: std::time::Instant,
    /// Simulated-audio-clock anchor: `(wall, content_pts_ms)` captured
    /// at the first blendshape (or audio_clock) of a turn. While no
    /// real `audio_clock` envelope arrives from the AudioSender's
    /// clock tap (the avatar `set_clock_tap` wiring is currently
    /// unimplemented end-to-end), we drive the state machine's
    /// audio clock forward at wall-time pace from this anchor — i.e.
    /// `sim_pts = anchor_pts + (now - anchor_wall)` ms. This relies
    /// on the AudioSender producing at strict real-time (50 fps × 20
    /// ms) and on the producer respecting `enqueue_frame().await`
    /// back-pressure (which it does after the 2 s ring-buffer cap),
    /// so wall-time-since-anchor ≈ playback content time.
    /// Reset on barge.
    playback_anchor: Option<(std::time::Instant, u64)>,
}

impl std::fmt::Debug for Live2DRenderNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Live2DRenderNode")
            .field("framerate", &self.config.framerate)
            .field("video_stream_id", &self.config.video_stream_id)
            .field(
                "frames_emitted",
                &self.frame_counter.load(Ordering::Relaxed),
            )
            .finish()
    }
}

/// Decoded input parsed from a `RuntimeData::Json` envelope.
#[derive(Debug)]
enum RendererInput {
    Blendshape(BlendshapeFrame),
    Emotion(String),
    AudioClock(u64),
    BargeIn,
}

impl Live2DRenderNode {
    /// Build the node with the given backend + config.
    ///
    /// Caller responsibility: the backend must already have its
    /// model loaded. For `WgpuBackend` that means calling
    /// `load_model(&config.model_path)` before this.
    pub fn new_with_backend(
        backend: Box<dyn Live2DBackend + Send>,
        config: Live2DRenderConfig,
    ) -> Self {
        let inner = RendererInner {
            state: Live2DRenderState::new(config.state_config.clone()),
            anchor_pts_ms: None,
            latest_blendshape_pts_ms: None,
            latest_audio_clock_ms: None,
            last_delta_log: std::time::Instant::now(),
            playback_anchor: None,
        };
        Self {
            config,
            inner: Arc::new(AsyncMutex::new(inner)),
            render_worker: Arc::new(RenderWorker::spawn(backend)),
            frame_counter: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Total frames emitted since construction (read for tests +
    /// diagnostics).
    pub fn frames_emitted(&self) -> u64 {
        self.frame_counter.load(Ordering::Relaxed)
    }

    /// Parse one envelope into a [`RendererInput`]. Returns `None`
    /// for envelopes the renderer doesn't recognize.
    fn decode_envelope(data: &RuntimeData) -> Option<RendererInput> {
        // Wrapped barge envelope from the session router (M2.6 path).
        if matches!(aux_port_of(data), Some(BARGE_IN_PORT)) {
            return Some(RendererInput::BargeIn);
        }
        let RuntimeData::Json(v) = data else {
            return None;
        };
        let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        match kind {
            "blendshapes" => BlendshapeFrame::from_json(v)
                .ok()
                .map(RendererInput::Blendshape),
            "emotion" => v
                .get("emoji")
                .and_then(|e| e.as_str())
                .map(|s| RendererInput::Emotion(s.to_string())),
            "audio_clock" => v
                .get("pts_ms")
                .and_then(|p| p.as_u64())
                .map(RendererInput::AudioClock),
            "barge_in" => Some(RendererInput::BargeIn),
            _ => None,
        }
    }
}

#[async_trait]
impl AsyncStreamingNode for Live2DRenderNode {
    fn node_type(&self) -> &str {
        "Live2DRenderNode"
    }

    async fn process(&self, _data: RuntimeData) -> Result<RuntimeData> {
        Err(Error::Execution(
            "Live2DRenderNode requires streaming mode — use process_streaming()".into(),
        ))
    }

    /// Reactive input path: ingest blendshape / emotion / audio_clock
    /// / barge envelopes and update the state machine. Frame emission
    /// moved to `tick()` (pacing-domains migration) — this method
    /// never invokes the callback. The pacer drives `tick()` at the
    /// bound wire clock or `fallback_rate_hz` (= `config.framerate`).
    async fn process_streaming<F>(
        &self,
        data: RuntimeData,
        _session_id: Option<String>,
        _callback: F,
    ) -> Result<usize>
    where
        F: FnMut(RuntimeData) -> Result<()> + Send,
    {
        let Some(input) = Self::decode_envelope(&data) else {
            return Ok(0);
        };
        let mut inner = self.inner.lock().await;
        let now = std::time::Instant::now();
        match input {
            RendererInput::Blendshape(f) => {
                let pts = f.pts_ms;
                if inner.anchor_pts_ms.is_none() {
                    inner.anchor_pts_ms = Some(pts);
                }
                if inner.playback_anchor.is_none() {
                    inner.playback_anchor = Some((now, pts));
                }
                inner.latest_blendshape_pts_ms = Some(pts);
                inner.state.push_blendshape(f);
            }
            RendererInput::Emotion(e) => {
                inner.state.push_emotion(&e);
            }
            RendererInput::AudioClock(pts) => {
                if inner.anchor_pts_ms.is_none() {
                    inner.anchor_pts_ms = Some(pts);
                }
                if inner.playback_anchor.is_none() {
                    inner.playback_anchor = Some((now, pts));
                }
                inner.latest_audio_clock_ms = Some(pts);
            }
            RendererInput::BargeIn => {
                inner.state.handle_barge();
                inner.anchor_pts_ms = None;
                inner.latest_blendshape_pts_ms = None;
                inner.latest_audio_clock_ms = None;
                inner.playback_anchor = None;
            }
        }
        Ok(0)
    }

    /// Universal barge handler — the session router (M2.6) forwards
    /// `<node>.in.barge_in` aux-port envelopes here in addition to
    /// firing `cancel.notify_waiters()`. We reset the renderer's
    /// state + anchor so the next blendshape stream re-anchors at
    /// its own pts.
    async fn process_control_message(
        &self,
        message: RuntimeData,
        _session_id: Option<String>,
    ) -> Result<bool> {
        if matches!(aux_port_of(&message), Some(BARGE_IN_PORT)) {
            let mut inner = self.inner.lock().await;
            inner.state.handle_barge();
            inner.anchor_pts_ms = None;
            inner.playback_anchor = None;
            return Ok(true);
        }
        Ok(false)
    }

    // ─── Pacing-domains: ClockedToOutboundMedia ───────────────────────
    //
    // The pacer drives `tick()` at the bound wire clock — typically
    // the outbound audio/video media_clock — falling back to a wall
    // pacer at `config.framerate` when no wire clock is bound (idle /
    // headless). Each tick samples the state machine at the effective
    // audio clock (taken from `latest_audio_clock_ms` if available,
    // else simulated from wall-time since the playback anchor) and
    // emits one rendered frame. No anchor → no emit (silence path).

    fn pacing_nature(&self) -> PacingNature {
        PacingNature::ClockedToOutboundMedia
    }

    fn fallback_rate_hz(&self) -> u32 {
        self.config.framerate.max(1)
    }

    async fn tick(
        &self,
        tick: Tick,
        _session_id: Option<String>,
        mut callback: Box<dyn FnMut(RuntimeData) -> std::result::Result<(), Error> + Send>,
    ) -> Result<()> {
        let (pose, cursor_ms) = {
            let mut inner = self.inner.lock().await;
            let now = std::time::Instant::now();

            // Effective audio clock: real tap > wall-sim from the
            // playback anchor. Without either we still emit an idle
            // frame (neutral pose); the state machine's no-clock
            // branch handles it.
            let effective_clock_ms: Option<u64> = inner.latest_audio_clock_ms.or_else(|| {
                inner.playback_anchor.map(|(t0, pts0)| {
                    let elapsed_ms = now.duration_since(t0).as_millis() as u64;
                    pts0.saturating_add(elapsed_ms)
                })
            });

            if now.duration_since(inner.last_delta_log).as_secs() >= 1 {
                inner.last_delta_log = now;
                let bs_ms = inner.latest_blendshape_pts_ms;
                let delta_ms = match (effective_clock_ms, bs_ms) {
                    (Some(a), Some(b)) => Some(a as i64 - b as i64),
                    _ => None,
                };
                let clock_source = if inner.latest_audio_clock_ms.is_some() {
                    "tap"
                } else if inner.playback_anchor.is_some() {
                    "wall_sim"
                } else {
                    "none"
                };
                tracing::info!(
                    target: "lipsync.delta",
                    audio_clock_ms = ?effective_clock_ms,
                    latest_blendshape_pts_ms = ?bs_ms,
                    delta_ms = ?delta_ms,
                    clock_source,
                    "audio_lead_vs_animation",
                );
            }

            if let Some(clock) = effective_clock_ms {
                inner.state.update_audio_clock(clock);
            }

            // pts: prefer the audio-time cursor (anchor-relative); fall
            // back to the Pacer's `tick.pts_us` for idle frames before
            // any anchor exists.
            let cursor_ms = match (inner.anchor_pts_ms, effective_clock_ms) {
                (Some(anchor), Some(absolute_pts)) => absolute_pts.saturating_sub(anchor),
                _ => tick.pts_us / 1000,
            };
            let frame_interval_ms: u64 = 1000_u64 / self.config.framerate.max(1) as u64;
            inner.state.tick(frame_interval_ms);
            (inner.state.compute_pose(), cursor_ms)
        };

        let frame = match self.render_worker.render(pose).await {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!(
                    "Live2DRenderNode backend.render_frame failed: {} (skipping frame)",
                    e
                );
                return Ok(());
            }
        };
        let frame_number = self.frame_counter.fetch_add(1, Ordering::AcqRel);
        let video = RuntimeData::Video {
            pixel_data: frame.pixels,
            width: frame.width,
            height: frame.height,
            format: PixelFormat::Rgb24,
            codec: None,
            frame_number,
            timestamp_us: cursor_ms.saturating_mul(1000),
            is_keyframe: true,
            stream_id: Some(self.config.video_stream_id.clone()),
            arrival_ts_us: None,
        };
        callback(video)?;
        Ok(())
    }
}
