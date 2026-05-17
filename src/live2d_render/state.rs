//! `Live2DRenderState` — input arbitration state machine for the
//! Live2D renderer. Per spec [§6.1]:
//!
//! 1. Drain pending input frames since last tick (blendshapes,
//!    emotions, audio clock ticks).
//! 2. Update internal state (ring eviction, emotion expiration,
//!    blink scheduler).
//! 3. Compute pose:
//!    - mouth: blendshape ring sample (linear interp between
//!      bounding keyframes), if audio clock has a recent value;
//!      else interpolate toward neutral over ~150 ms.
//!    - emotion: expression+motion lookup against config map;
//!      expression hangs around for `expression_hold_seconds`
//!      then reverts to `neutral`.
//!    - eyes: blink scheduler (when no emotion active).
//! 4. Backend renders the pose.
//! 5. Streaming node emits `RuntimeData::Video`.
//!
//! Steps 4 + 5 live in M4.4 + M4.5 respectively. This module
//! covers steps 1–3.

use crate::arkit::BlendshapeFrame;

/// Number of ARKit blendshapes in a `BlendshapeFrame.arkit_52`
/// (mirrors the const in `lip_sync::blendshape`; re-stated here so
/// the renderer crate doesn't reach into a private module).
pub const ARKIT_52: usize = 52;
use std::collections::HashMap;

/// Default neutral mouth interpolation window per spec §6.1
/// ("interpolate toward neutral over ~150 ms").
pub const DEFAULT_NEUTRAL_INTERP_MS: u64 = 150;

/// Default stale-blendshape eviction window per spec §6.1
/// (`pts_ms < audio_clock_ms - 200`).
pub const DEFAULT_STALE_WINDOW_MS: u64 = 200;

/// Default expression hold per Live2D.md
/// ("EXPRESSION_HOLD_DURATION_SECONDS, default: 3 seconds").
pub const DEFAULT_EXPRESSION_HOLD_SECONDS: f32 = 3.0;

/// Default neutral expression name per Live2D.md
/// ("NEUTRAL_EXPRESSION_ID, default: neutral").
pub const DEFAULT_NEUTRAL_EXPRESSION_ID: &str = "neutral";

/// Default neutral motion group when no emotion is active.
pub const DEFAULT_NEUTRAL_MOTION_GROUP: &str = "Idle";

/// Default talking motion group during audio playback without an
/// active emotion (Live2D.md `NEUTRAL_TALKING_MOTION_GROUP = "Talking"`).
pub const DEFAULT_TALKING_MOTION_GROUP: &str = "Talking";

/// Default blink interval bounds — naturalistic average ~4 s.
pub const DEFAULT_BLINK_INTERVAL_MIN_MS: u64 = 3_000;
pub const DEFAULT_BLINK_INTERVAL_MAX_MS: u64 = 6_000;

/// Default blink animation length (open → closed → open).
pub const DEFAULT_BLINK_DURATION_MS: u64 = 200;

/// Map one ARKit-52 vector → VBridger Live2D parameter values.
///
/// The default mapping mirrors persona-engine's
/// `VBridgerLipSyncService` aliasing rules:
/// - `ParamJawOpen`, `ParamMouthOpenY` ← `arkit[jawOpen]`
/// - `ParamMouthForm` ← (smileLeft+smileRight)/2 - (frownLeft+frownRight)/2
/// - `ParamMouthSmile` ← (smileLeft + smileRight) / 2
/// - `ParamMouthPucker` ← `arkit[mouthPucker]`
/// - `ParamMouthFunnel` ← `arkit[mouthFunnel]`
/// - `ParamMouthClose` ← `arkit[mouthClose]`
///
/// Custom riggers can swap this out by handing a different
/// implementation to [`StateConfig::mapper`]; default ships the
/// VBridger-canonical mapping.
pub trait ArkitToVBridger: Send + Sync + std::fmt::Debug {
    /// Write the mapped params into `out` (typically a HashMap
    /// pre-cleared). Implementations should write deterministically
    /// — same input → same output.
    fn map(&self, arkit: &[f32; ARKIT_52], out: &mut HashMap<String, f32>);
}

/// Default mapping — Rust port of persona-engine's
/// `ARKitToLive2DMapper.cs`. Mirrors the VBridger-canonical formulas
/// + `ResponseCurves::ease_in` on the jaw/mouth-open axes so small
/// audio responses produce visible motion (without ease_in, audio-driven
/// jaw values around 0.05-0.25 read as a barely-moving mouth).
#[derive(Debug, Clone, Copy, Default)]
pub struct DefaultArkitMapper;

impl ArkitToVBridger for DefaultArkitMapper {
    fn map(&self, arkit: &[f32; ARKIT_52], out: &mut HashMap<String, f32>) {
        use crate::arkit::response_curves::center_weighted;

        // Indices match `nodes::lip_sync::blendshape::ARKIT_BLENDSHAPE_NAMES`.
        let jaw_open = arkit[17];
        let mouth_close = arkit[18];
        let mouth_funnel = arkit[19];
        let mouth_pucker = arkit[20];
        let mouth_left = arkit[21];
        let mouth_right = arkit[22];
        let mouth_smile_l = arkit[23];
        let mouth_smile_r = arkit[24];
        let mouth_frown_l = arkit[25];
        let mouth_frown_r = arkit[26];
        let mouth_dimple_l = arkit[27];
        let mouth_dimple_r = arkit[28];
        let mouth_shrug_lower = arkit[31];
        let mouth_shrug_upper = arkit[32];
        let mouth_press_l = arkit[33];
        let mouth_press_r = arkit[34];
        let mouth_lower_down_l = arkit[35];
        let mouth_lower_down_r = arkit[36];
        let mouth_upper_up_l = arkit[37];
        let mouth_upper_up_r = arkit[38];
        let mouth_roll_lower = arkit[29];
        let mouth_roll_upper = arkit[30];

        // JawOpen + MouthOpenY: noise-gated, aggressively amplified
        // response. Audio2Face's Claire identity outputs jaw values
        // that peak around 0.15-0.3 for normal speech, which after
        // a plain `ease_in(t) = t*(2-t)` mapping (per persona-engine
        // reference) only opens the mouth 28-51% of full range. That
        // reads as "floaty"/under-responsive on Aria. We amplify more
        // aggressively (`pow(t, 0.4)` over a noise-gated input):
        //   jaw=0.05 → 0     (gated; no pseudo-motion during silence)
        //   jaw=0.15 → 0.46  (vs 0.28 with ease_in)
        //   jaw=0.30 → 0.62  (vs 0.51)
        //   jaw=0.50 → 0.76  (vs 0.75 — saturates similarly at high end)
        //   jaw=1.00 → 1.00
        // Result: mouth visibly snaps between closed and open with the
        // same audio2face output.
        let jaw_open_out = snappy_open(jaw_open);

        // MouthOpenY: VBridger formula with the same snappy curve.
        let mouth_open_raw = ((jaw_open - mouth_close)
            - (mouth_roll_upper + mouth_roll_lower) * 0.2
            + mouth_funnel * 0.2)
            .clamp(0.0, 1.0);
        let mouth_open_y = snappy_open(mouth_open_raw);

        // MouthForm: VBridger compound smile-frown axis.
        let dimple_avg = (mouth_dimple_l + mouth_dimple_r) * 0.5;
        let mouth_form = ((2.0 - mouth_frown_l - mouth_frown_r - mouth_pucker
            + mouth_smile_r
            + mouth_smile_l
            + dimple_avg)
            / 4.0)
            .clamp(-1.0, 1.0);

        // MouthFunnel: raw funnel minus jaw artifact.
        let mouth_funnel_out = (mouth_funnel - jaw_open * 0.2).clamp(0.0, 1.0);

        // MouthPressLipOpen: center-weighted Hermite spline.
        let press_raw =
            ((mouth_upper_up_r + mouth_upper_up_l + mouth_lower_down_r + mouth_lower_down_l) / 1.8
                - (mouth_roll_lower + mouth_roll_upper))
                .clamp(-1.3, 1.3);
        let mouth_press_lip_open = center_weighted(press_raw, -1.3, 1.3);

        // MouthPuckerWiden: spread-vs-pucker.
        let mouth_pucker_widen =
            ((mouth_dimple_r + mouth_dimple_l) * 2.0 - mouth_pucker).clamp(-1.0, 1.0);

        // MouthX: lateral shift + asymmetric smile.
        let mouth_x =
            ((mouth_left - mouth_right) + (mouth_smile_l - mouth_smile_r)).clamp(-1.0, 1.0);

        // MouthShrug: chin raise + lip compression.
        let mouth_shrug = ((mouth_shrug_upper + mouth_shrug_lower + mouth_press_r + mouth_press_l)
            / 4.0)
            .clamp(0.0, 1.0);

        out.insert("ParamJawOpen".to_string(), jaw_open_out);
        out.insert("ParamMouthOpenY".to_string(), mouth_open_y);
        out.insert("ParamMouthForm".to_string(), mouth_form);
        out.insert("ParamMouthFunnel".to_string(), mouth_funnel_out);
        out.insert("ParamMouthPressLipOpen".to_string(), mouth_press_lip_open);
        out.insert("ParamMouthPuckerWiden".to_string(), mouth_pucker_widen);
        out.insert("ParamMouthX".to_string(), mouth_x);
        out.insert("ParamMouthShrug".to_string(), mouth_shrug);
    }
}

/// Noise-gated jaw/mouth-open response that snaps cleanly between
/// closed and open.
///
/// Audio2Face's Claire identity outputs jaw values that:
/// - Sit around 0.0-0.10 during silence/breathing (audio-only noise)
/// - Peak around 0.15-0.65 during normal speech phonemes
/// - Trail off slowly after a phoneme (e.g. 0.6 → 0.5 → 0.4 → 0.3
///   → 0.2 → 0.1 → 0.05 over 5-9 frames)
///
/// A plain `ease_in(t) = t * (2 - t)` (persona-engine default)
/// amplifies the low-value tail (jaw=0.1 → 0.19), so the mouth
/// stays visibly open during the closing tail of every phoneme —
/// reads as "floaty"/"slow to close".
///
/// `pow(t, 0.4)` was even worse: jaw=0.1 → 0.40, mouth visibly
/// half-open during the tail.
///
/// This remaps audio2face's effective speaking range `[0.10, 0.55]`
/// linearly to the model's full `[0, 1]` parameter range with a
/// smoothstep ease, then clamps the rest. The cutoff at 0.10 snaps
/// the tail closed sharply, and the saturation at 0.55 keeps peak
/// vowels at full mouth-open even when audio2face plateaus below 1.0.
fn snappy_open(t: f32) -> f32 {
    const LOW: f32 = 0.10;
    const HIGH: f32 = 0.55;
    let t = t.clamp(0.0, 1.0);
    if t <= LOW {
        return 0.0;
    }
    if t >= HIGH {
        return 1.0;
    }
    let s = (t - LOW) / (HIGH - LOW); // 0..1
                                      // Smoothstep: 3s² - 2s³. Eases in and out symmetrically so the
                                      // mouth doesn't snap with a hard edge at either bound.
    s * s * (3.0 - 2.0 * s)
}

/// One emoji's expression + motion target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmotionEntry {
    /// Expression name (e.g. `excited_star`). Looked up in the
    /// model's `.exp3.json` table at render time.
    pub expression_id: String,
    /// Motion-group name (e.g. `Excited`). Looked up in the
    /// model3.json `Motions` block.
    pub motion_group: String,
}

impl EmotionEntry {
    /// Convenience constructor for callers building config tables
    /// inline (e.g. tests + the default map below).
    pub fn new(expression_id: impl Into<String>, motion_group: impl Into<String>) -> Self {
        Self {
            expression_id: expression_id.into(),
            motion_group: motion_group.into(),
        }
    }
}

/// Default emoji → emotion map per
/// [`external/handcrafted-persona-engine/Live2D.md`]. Non-Aria
/// models can swap this out via [`StateConfig::emotion_mapping`].
///
/// Aria specifically only ships rigging for `neutral`/`smug`/`happy`/
/// `frustrated`/`sad` expressions and `Idle`/`Talking`/`Happy`/
/// `Confident` motions; the renderer falls back to `neutral` +
/// `Idle` when an emoji's mapping points at a missing expression
/// or motion (M4.5 will own the fallback path).
pub fn default_emotion_mapping() -> HashMap<String, EmotionEntry> {
    let raw: &[(&str, &str, &str)] = &[
        ("\u{1F60A}", "happy", "Happy"),          // 😊
        ("\u{1F929}", "excited_star", "Excited"), // 🤩
        ("\u{1F60E}", "cool", "Confident"),       // 😎
        ("\u{1F60F}", "smug", "Confident"),       // 😏
        ("\u{1F4AA}", "determined", "Confident"), // 💪
        ("\u{1F633}", "embarrassed", "Nervous"),  // 😳
        ("\u{1F632}", "shocked", "Surprised"),    // 😲
        ("\u{1F914}", "thinking", "Thinking"),    // 🤔
        ("\u{1F440}", "suspicious", "Thinking"),  // 👀
        ("\u{1F624}", "frustrated", "Angry"),     // 😤
        ("\u{1F622}", "sad", "Sad"),              // 😢
        ("\u{1F605}", "awkward", "Nervous"),      // 😅
        ("\u{1F644}", "dismissive", "Annoyed"),   // 🙄
        ("\u{1F495}", "adoring", "Happy"),        // 💕
        ("\u{1F602}", "laughing", "Happy"),       // 😂
        ("\u{1F525}", "passionate", "Excited"),   // 🔥
        ("\u{2728}", "sparkle", "Happy"),         // ✨
    ];
    raw.iter()
        .map(|(emoji, expr, motion)| ((*emoji).to_string(), EmotionEntry::new(*expr, *motion)))
        .collect()
}

/// Configuration for [`Live2DRenderState`].
#[derive(Debug, Clone)]
pub struct StateConfig {
    /// How long an emotion expression sticks around after triggering
    /// before reverting to neutral.
    pub expression_hold_seconds: f32,
    /// Neutral expression's name in the model. Renderer falls back
    /// to this when no emotion is active or after expiration.
    pub neutral_expression_id: String,
    /// Motion group played by default when no emotion is active.
    pub neutral_motion_group: String,
    /// Motion group played during audio playback without an active
    /// emotion (subtle background movement).
    pub talking_motion_group: String,
    /// How long the mouth interpolates toward neutral after the
    /// audio clock goes quiet.
    pub neutral_interp_ms: u64,
    /// Stale-pts eviction window: ring entries with
    /// `pts_ms < audio_clock_ms - stale_blendshape_window_ms` are
    /// dropped.
    pub stale_blendshape_window_ms: u64,
    /// Lower bound on blink interval (wall ms between blinks).
    pub blink_interval_min_ms: u64,
    /// Upper bound on blink interval.
    pub blink_interval_max_ms: u64,
    /// One full blink animation duration (open→closed→open).
    pub blink_duration_ms: u64,
    /// Emoji → emotion table; defaults to
    /// [`default_emotion_mapping`].
    pub emotion_mapping: HashMap<String, EmotionEntry>,
    /// ARKit-52 → VBridger param mapper (boxed to allow custom
    /// implementations without changing the type signature). The
    /// default is [`DefaultArkitMapper`].
    pub mapper: std::sync::Arc<dyn ArkitToVBridger>,
    /// PRNG seed for the blink scheduler. Same seed → same blink
    /// timing across runs (test determinism).
    pub blink_seed: u64,
}

impl StateConfig {
    /// Default config — matches the persona-engine + spec defaults
    /// for every knob.
    pub fn default_config() -> Self {
        Self {
            expression_hold_seconds: DEFAULT_EXPRESSION_HOLD_SECONDS,
            neutral_expression_id: DEFAULT_NEUTRAL_EXPRESSION_ID.to_string(),
            neutral_motion_group: DEFAULT_NEUTRAL_MOTION_GROUP.to_string(),
            talking_motion_group: DEFAULT_TALKING_MOTION_GROUP.to_string(),
            neutral_interp_ms: DEFAULT_NEUTRAL_INTERP_MS,
            stale_blendshape_window_ms: DEFAULT_STALE_WINDOW_MS,
            blink_interval_min_ms: DEFAULT_BLINK_INTERVAL_MIN_MS,
            blink_interval_max_ms: DEFAULT_BLINK_INTERVAL_MAX_MS,
            blink_duration_ms: DEFAULT_BLINK_DURATION_MS,
            emotion_mapping: default_emotion_mapping(),
            mapper: std::sync::Arc::new(DefaultArkitMapper),
            blink_seed: 0xCAFE_F00D,
        }
    }
}

impl Default for StateConfig {
    fn default() -> Self {
        Self::default_config()
    }
}

/// Pose computed for one render tick. Every backend takes this as
/// input.
///
/// `params` carries the VBridger Live2D parameter values (mouth
/// shape, eyes, brows, …); `part_opacities` carries any
/// expression-driven part-opacity overrides (M4.2's `.exp3.json`
/// `Parameters` with `Target=Part` will populate this in M4.5).
/// `expression_id` + `motion_group` are passed through from the
/// active emotion state.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Pose {
    /// VBridger Live2D parameter values keyed by parameter ID
    /// (`ParamJawOpen`, `ParamEyeLOpen`, etc.).
    pub params: HashMap<String, f32>,
    /// Expression-driven part-opacity overrides, keyed by part ID.
    /// Empty in M4.3 (no .exp3.json evaluation yet).
    pub part_opacities: HashMap<String, f32>,
    /// Currently-active expression name. `"neutral"` when nothing
    /// emotion-relevant is firing.
    pub expression_id: String,
    /// Currently-active motion group. `"Idle"` when nothing
    /// emotion-relevant is firing.
    pub motion_group: String,
    /// Convenience: eye-open value [0, 1] for the blink scheduler.
    /// `1.0` = fully open, `0.0` = fully closed. M4.5 will project
    /// this onto `ParamEyeLOpen` + `ParamEyeROpen` if the rig has
    /// them.
    pub eye_open: f32,
}

impl Pose {
    /// Look up a mouth-shape param value, defaulting to `0.0` when
    /// the param isn't in the map.
    pub fn mouth_value(&self, param_id: &str) -> f32 {
        self.params.get(param_id).copied().unwrap_or(0.0)
    }
}

// ─── Live2DRenderState ───────────────────────────────────────────────────────

/// Input arbitration state machine. Per spec §6.1.
///
/// Owns:
///
/// - **Blendshape ring** — keyframes pushed by the lip-sync node;
///   evicted when stale.
/// - **Audio clock** — last known `pts_ms` from `audio.out.clock`
///   tap, plus wall time of last update (for "no clock" detection).
/// - **Emotion state** — active emotion + its wall-time expiry.
/// - **Blink scheduler** — wall-time-based, seeded PRNG so tests
///   are deterministic.
/// - **Wall-clock** — internal counter advanced by `tick_wall` /
///   `tick`. Decoupled from `Instant` so the state machine is
///   testable without time mocking.
#[derive(Debug)]
pub struct Live2DRenderState {
    config: StateConfig,
    /// Insertion-ordered ring (older entries at front; eviction is
    /// front-truncating + range-deleting since pts_ms isn't strictly
    /// sorted under burst arrival in real use).
    ring: Vec<BlendshapeFrame>,
    /// Last audio clock value seen (None until first update).
    audio_clock_ms: Option<u64>,
    /// Wall-time when audio clock was last updated.
    audio_clock_wall_ms: Option<u64>,
    /// Currently-active emotion, if any.
    active_emotion: Option<ActiveEmotion>,
    /// Cumulative virtual wall-time. Advances via `tick_wall` / `tick`.
    wall_now_ms: u64,
    /// Wall-time of the next scheduled blink start.
    next_blink_start_ms: u64,
    /// Wall-time the current blink started (None when not blinking).
    blink_in_progress_start_ms: Option<u64>,
    /// SplitMix64 RNG state for blink-interval picks.
    rng_state: u64,
}

#[derive(Debug, Clone)]
struct ActiveEmotion {
    entry: EmotionEntry,
    wall_expires_ms: u64,
}

impl Live2DRenderState {
    /// Build a state machine with the given config. Initial
    /// wall-time is 0; first blink is scheduled per the configured
    /// interval.
    pub fn new(config: StateConfig) -> Self {
        let mut s = Self {
            rng_state: config.blink_seed,
            config,
            ring: Vec::new(),
            audio_clock_ms: None,
            audio_clock_wall_ms: None,
            active_emotion: None,
            wall_now_ms: 0,
            next_blink_start_ms: 0,
            blink_in_progress_start_ms: None,
        };
        s.schedule_next_blink();
        s
    }

    // ── Inputs ──────────────────────────────────────────────────────────────

    /// Push a blendshape keyframe into the ring. The state evicts
    /// stale entries (`pts_ms < audio_clock_ms - stale_window_ms`)
    /// at this point so the ring doesn't grow unboundedly when the
    /// audio clock advances.
    pub fn push_blendshape(&mut self, frame: BlendshapeFrame) {
        self.ring.push(frame);
        self.evict_stale();
    }

    /// Push an emotion event (typically an emoji string from
    /// `EmotionExtractorNode`). Looks up the emoji in the config
    /// map; unknown emojis are silently ignored.
    pub fn push_emotion(&mut self, emoji: &str) {
        if let Some(entry) = self.config.emotion_mapping.get(emoji).cloned() {
            let hold_ms = (self.config.expression_hold_seconds * 1000.0) as u64;
            self.active_emotion = Some(ActiveEmotion {
                entry,
                wall_expires_ms: self.wall_now_ms.saturating_add(hold_ms),
            });
        }
    }

    /// Update the audio playback clock. The wall time is captured
    /// implicitly (`self.wall_now_ms`) so "no clock for N ms"
    /// detection works — `tick_wall(d)` advances wall_now_ms but
    /// not audio_clock_wall_ms.
    pub fn update_audio_clock(&mut self, pts_ms: u64) {
        self.audio_clock_ms = Some(pts_ms);
        self.audio_clock_wall_ms = Some(self.wall_now_ms);
        self.evict_stale();
    }

    /// Advance the virtual wall clock. The renderer (M4.5) calls
    /// this each render tick with the elapsed Instant delta; tests
    /// drive it directly.
    pub fn tick(&mut self, wall_elapsed_ms: u64) {
        self.wall_now_ms = self.wall_now_ms.saturating_add(wall_elapsed_ms);

        // Expire stale emotions.
        if let Some(ae) = &self.active_emotion {
            if self.wall_now_ms >= ae.wall_expires_ms {
                self.active_emotion = None;
            }
        }
    }

    /// Convenience: advance the wall clock by a `Duration`. Mirrors
    /// the spec test signature.
    pub fn tick_wall(&mut self, d: std::time::Duration) {
        self.tick(d.as_millis() as u64);
    }

    /// Advance the wall clock by `ms` while *not* updating the
    /// audio clock. Used in tests to simulate "audio clock went
    /// quiet for N ms" to exercise the neutral-mouth interpolation.
    pub fn tick_no_clock_for(&mut self, ms: u64) {
        self.tick(ms);
    }

    /// Handle a barge-in. Spec §6.3: clears the blendshape ring +
    /// snaps mouth target to neutral (audio_clock_ms is dropped so
    /// `compute_pose` enters the "no clock" branch), but **emotion
    /// is preserved** ("the avatar shouldn't go emotionally blank
    /// just because the user interrupted").
    pub fn handle_barge(&mut self) {
        self.ring.clear();
        // Drop the audio clock too — the listener's playback was
        // cancelled; new clock ticks will arrive when fresh audio
        // starts. Spec §6.1: "interpolate toward neutral over
        // ~150 ms" engages within one tick.
        self.audio_clock_ms = None;
        self.audio_clock_wall_ms = None;
    }

    // ── Outputs ─────────────────────────────────────────────────────────────

    /// Compute the pose for the current tick. Idempotent — calling
    /// twice in a row yields the same value (pure read of state).
    pub fn compute_pose(&mut self) -> Pose {
        // Mouth: sample blendshape ring at audio_clock_ms; fall
        // back to neutral interp.
        let mouth_arkit = self.sample_mouth();
        let mut params = HashMap::new();
        self.config.mapper.map(&mouth_arkit, &mut params);

        // Emotion → expression + motion.
        let (expression_id, motion_group) = match &self.active_emotion {
            Some(ae) => (
                ae.entry.expression_id.clone(),
                ae.entry.motion_group.clone(),
            ),
            None => {
                // No active emotion. Pick `Talking` if audio is
                // playing, `Idle` otherwise.
                let group = if self.audio_clock_recently_active() {
                    self.config.talking_motion_group.clone()
                } else {
                    self.config.neutral_motion_group.clone()
                };
                (self.config.neutral_expression_id.clone(), group)
            }
        };

        // Blink (only when no emotion is active — emotion
        // expressions take over the eyes).
        let eye_open = if self.active_emotion.is_some() {
            1.0
        } else {
            self.tick_blink_scheduler()
        };

        Pose {
            params,
            part_opacities: HashMap::new(),
            expression_id,
            motion_group,
            eye_open,
        }
    }

    // ── Test-visibility hooks ───────────────────────────────────────────────

    /// Borrow the current ring. Test-only — not part of the public
    /// API; the renderer doesn't need to inspect it.
    #[doc(hidden)]
    pub fn blendshape_ring_for_test(&self) -> &[BlendshapeFrame] {
        &self.ring
    }

    /// Current audio-clock value (for diagnostic / test use).
    #[doc(hidden)]
    pub fn audio_clock_ms_for_test(&self) -> Option<u64> {
        self.audio_clock_ms
    }

    /// Current internal wall-clock value (for diagnostic / test use).
    #[doc(hidden)]
    pub fn wall_now_ms_for_test(&self) -> u64 {
        self.wall_now_ms
    }

    // ── Internals ───────────────────────────────────────────────────────────

    fn evict_stale(&mut self) {
        let Some(ac) = self.audio_clock_ms else {
            return;
        };
        let cutoff = ac.saturating_sub(self.config.stale_blendshape_window_ms);
        self.ring.retain(|f| f.pts_ms >= cutoff);
    }

    /// Returns the ARKit-52 vector to feed the mapper. Branches:
    /// - Audio clock recent + at least one ring entry with
    ///   `pts_ms <= audio_clock_ms`: lerp between the two bounding
    ///   keyframes.
    /// - Audio clock recent but no covering ring entry: hold
    ///   neutral (zeros).
    /// - Audio clock missing or stale: interpolate from "what
    ///   mouth was at last clock" toward neutral over
    ///   `neutral_interp_ms`.
    fn sample_mouth(&self) -> [f32; ARKIT_52] {
        let neutral = [0.0f32; ARKIT_52];

        let Some(ac) = self.audio_clock_ms else {
            return neutral;
        };

        // Time since last audio clock update (wall ms).
        let since_ac_wall = self
            .audio_clock_wall_ms
            .map(|w| self.wall_now_ms.saturating_sub(w))
            .unwrap_or(u64::MAX);

        if since_ac_wall <= self.config.neutral_interp_ms {
            // Audio clock fresh — sample the ring.
            if let Some(sampled) = self.lerp_ring_at(ac) {
                return sampled;
            }
            // Audio clock fresh but ring empty — hold neutral.
            return neutral;
        }

        // Audio clock stale — interpolate from last sample toward
        // neutral. Use the most recent ring sample at the moment
        // the clock was current, eased toward zero.
        let t = ((since_ac_wall - self.config.neutral_interp_ms) as f32
            / self.config.neutral_interp_ms.max(1) as f32)
            .clamp(0.0, 1.0);
        // After 2× neutral_interp_ms wall time without clock, fully
        // neutral. Linear ease for simplicity (spec doesn't pin
        // a curve shape).
        if t >= 1.0 {
            return neutral;
        }
        let last = self.lerp_ring_at(ac).unwrap_or(neutral);
        let mut out = [0.0f32; ARKIT_52];
        for i in 0..ARKIT_52 {
            out[i] = last[i] * (1.0 - t);
        }
        out
    }

    /// Linearly interpolate the ring at `pts`. Returns `None` if no
    /// ring entry has `pts_ms <= pts` (i.e. the clock points at a
    /// time before any keyframe arrived — happens at session start
    /// before the lip-sync node has produced its first window).
    fn lerp_ring_at(&self, pts: u64) -> Option<[f32; ARKIT_52]> {
        if self.ring.is_empty() {
            return None;
        }

        // Find the bracketing pair `(left, right)` such that
        // `left.pts_ms <= pts <= right.pts_ms`. Sort by pts on the
        // fly — ring isn't guaranteed sorted (callers may push out
        // of order under burst). For typical ring sizes (<60 entries
        // at 30 fps × 2 s buffer) sorting per query is cheap; if it
        // ever shows up in a profile we can add insertion-sort on
        // push.
        let mut by_pts: Vec<&BlendshapeFrame> = self.ring.iter().collect();
        by_pts.sort_by_key(|f| f.pts_ms);

        // First entry with pts_ms >= pts — that's `right`.
        let right_idx = by_pts.partition_point(|f| f.pts_ms < pts);
        if right_idx >= by_pts.len() {
            // pts is past the last frame. Hold the last frame's
            // values rather than extrapolating.
            return Some(by_pts.last().unwrap().arkit_52);
        }
        let right = by_pts[right_idx];
        if right_idx == 0 {
            // pts is before the first frame. Return None so caller
            // can hold neutral (avoids extrapolating into garbage).
            if right.pts_ms == pts {
                return Some(right.arkit_52);
            }
            return None;
        }
        let left = by_pts[right_idx - 1];

        if left.pts_ms == right.pts_ms {
            return Some(left.arkit_52);
        }
        let span = (right.pts_ms - left.pts_ms) as f32;
        let t = ((pts - left.pts_ms) as f32 / span).clamp(0.0, 1.0);

        let mut out = [0.0f32; ARKIT_52];
        for i in 0..ARKIT_52 {
            out[i] = left.arkit_52[i] + (right.arkit_52[i] - left.arkit_52[i]) * t;
        }
        Some(out)
    }

    /// `true` if the audio clock has been updated within
    /// `neutral_interp_ms`. Used to pick `Talking` vs `Idle`
    /// motion group.
    fn audio_clock_recently_active(&self) -> bool {
        self.audio_clock_wall_ms
            .map(|w| self.wall_now_ms.saturating_sub(w) <= self.config.neutral_interp_ms)
            .unwrap_or(false)
    }

    /// Tick the blink scheduler. Returns the eye_open value
    /// (`0.0` = fully closed, `1.0` = fully open). Schedule advances
    /// state-mutably even though the function looks read-only —
    /// this is a `&mut self` method called from `compute_pose`.
    fn tick_blink_scheduler(&mut self) -> f32 {
        // Start a new blink if scheduled. Anchor `start` at the
        // *scheduled* time, not the time we noticed — otherwise a
        // long inter-tick gap (e.g. 200 ms while a single render
        // tick lands) would lose the elapsed-into-blink portion
        // and the blink would always look like it just started.
        if self.blink_in_progress_start_ms.is_none() && self.wall_now_ms >= self.next_blink_start_ms
        {
            self.blink_in_progress_start_ms = Some(self.next_blink_start_ms);
        }

        // Compute current eye_open if a blink is in progress.
        let eye_open = if let Some(start) = self.blink_in_progress_start_ms {
            let elapsed = self.wall_now_ms.saturating_sub(start);
            if elapsed >= self.config.blink_duration_ms {
                // Blink complete — schedule the next one.
                self.blink_in_progress_start_ms = None;
                self.schedule_next_blink();
                1.0
            } else {
                // Symmetric V-shape: linear close then open.
                let half = self.config.blink_duration_ms / 2;
                if half == 0 {
                    1.0
                } else if elapsed <= half {
                    // Closing.
                    1.0 - (elapsed as f32 / half as f32)
                } else {
                    // Opening.
                    (elapsed as f32 - half as f32) / half as f32
                }
            }
        } else {
            1.0
        };
        eye_open
    }

    fn schedule_next_blink(&mut self) {
        let lo = self.config.blink_interval_min_ms;
        let hi = self.config.blink_interval_max_ms.max(lo);
        let interval = if lo == hi {
            lo
        } else {
            lo + (self.next_u64() % (hi - lo + 1))
        };
        self.next_blink_start_ms = self.wall_now_ms.saturating_add(interval);
    }

    /// Internal SplitMix64 RNG step. Same seed → same sequence
    /// across runs (test determinism).
    fn next_u64(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.rng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    /// Verbatim spec-test path: lerp between two bounding keyframes
    /// at audio_clock_ms = 150 with frames at pts 100 (val=0.5)
    /// and pts 200 (val=1.0). The mapper applies `ease_in` to
    /// `ParamJawOpen`, so the asserted output is `ease_in(0.75)`,
    /// not the raw lerp.
    #[test]
    fn samples_blendshape_keyframe_at_audio_clock_pts() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([0.5; ARKIT_52], 100, None));
        state.push_blendshape(BlendshapeFrame::new([1.0; ARKIT_52], 200, None));
        state.update_audio_clock(150);
        let pose = state.compute_pose();
        // snappy_open(0.75) saturates at HIGH=0.55 → 1.0
        assert!(
            approx(pose.mouth_value("ParamJawOpen"), 1.0, 1e-3),
            "expected snappy_open(0.75)=1.0 (saturated), got {}",
            pose.mouth_value("ParamJawOpen")
        );
    }

    #[test]
    fn samples_at_exact_keyframe_returns_keyframe_value() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([0.42; ARKIT_52], 500, None));
        state.update_audio_clock(500);
        let pose = state.compute_pose();
        // snappy_open(0.42): in linear range. s = (0.42 - 0.10) / (0.55 - 0.10) ≈ 0.711
        // smoothstep(s) = s² * (3 - 2s) ≈ 0.611
        let s = (0.42f32 - 0.10) / (0.55 - 0.10);
        let expected = s * s * (3.0 - 2.0 * s);
        assert!(approx(pose.mouth_value("ParamJawOpen"), expected, 1e-3));
    }

    /// Spec §6.1: evict pts_ms < audio_clock_ms - stale_window.
    /// Defaults stale window to 200 ms.
    #[test]
    fn evicts_stale_blendshape_frames_after_200ms() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([0.5; ARKIT_52], 100, None));
        state.update_audio_clock(500);
        // pts=100 < 500 - 200 = 300, so the entry should be evicted.
        state.push_blendshape(BlendshapeFrame::new([1.0; ARKIT_52], 510, None));
        let ring = state.blendshape_ring_for_test();
        assert_eq!(ring.len(), 1);
        assert_eq!(ring[0].pts_ms, 510);
    }

    /// Spec §6.1: interpolate to neutral when audio clock is quiet.
    #[test]
    fn interpolates_to_neutral_when_audio_clock_quiet() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([1.0; ARKIT_52], 100, None));
        state.update_audio_clock(110);
        // 200 ms with no clock update is past the
        // neutral_interp_ms window (default 150) so mouth has fully
        // eased to zero.
        state.tick_no_clock_for(2 * DEFAULT_NEUTRAL_INTERP_MS);
        let pose = state.compute_pose();
        assert!(
            approx(pose.mouth_value("ParamJawOpen"), 0.0, 1e-3),
            "expected ~0.0 after stale clock, got {}",
            pose.mouth_value("ParamJawOpen")
        );
    }

    #[test]
    fn emotion_event_drives_expression_and_motion() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_emotion("\u{1F929}"); // 🤩
        let pose = state.compute_pose();
        assert_eq!(pose.expression_id, "excited_star");
        assert_eq!(pose.motion_group, "Excited");
    }

    /// Spec §6.1: emotion expires after `expression_hold_seconds`.
    #[test]
    fn emotion_expires_after_hold_seconds_back_to_neutral() {
        let mut state = Live2DRenderState::new(StateConfig {
            expression_hold_seconds: 1.0,
            ..StateConfig::default_config()
        });
        state.push_emotion("\u{1F929}"); // 🤩
        state.tick_wall(std::time::Duration::from_millis(1100));
        let pose = state.compute_pose();
        assert_eq!(pose.expression_id, "neutral");
        assert_eq!(pose.motion_group, "Idle");
    }

    /// Spec §6.3: barge clears the blendshape ring but NOT the
    /// active emotion ("the avatar shouldn't go emotionally blank
    /// just because the user interrupted").
    #[test]
    fn barge_clears_ring_but_preserves_emotion() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([1.0; ARKIT_52], 100, None));
        state.push_emotion("\u{1F929}"); // 🤩
        state.handle_barge();
        assert!(state.blendshape_ring_for_test().is_empty());
        // Emotion still active.
        let pose = state.compute_pose();
        assert_eq!(pose.expression_id, "excited_star");
    }

    /// Spec §6.1: idle blink scheduler fires when no emotion is
    /// active.
    #[test]
    fn idle_blink_fires_when_no_emotion_active() {
        let mut state = Live2DRenderState::new(StateConfig {
            blink_interval_min_ms: 100,
            blink_interval_max_ms: 100,
            blink_duration_ms: 200,
            ..StateConfig::default_config()
        });
        // Advance to mid-blink: 100 ms wait + 100 ms into the
        // 200 ms blink animation = 50% closed.
        state.tick_wall(std::time::Duration::from_millis(150));
        let pose = state.compute_pose();
        assert!(
            pose.eye_open < 1.0,
            "expected blink in progress (eye_open < 1.0), got {}",
            pose.eye_open
        );
        assert!(
            pose.eye_open > 0.0,
            "expected mid-blink (eye_open > 0.0), got {}",
            pose.eye_open
        );
    }

    /// While an emotion is active, the blink scheduler is gated
    /// off — emotion expressions own the eyes.
    #[test]
    fn blink_suppressed_during_active_emotion() {
        let mut state = Live2DRenderState::new(StateConfig {
            blink_interval_min_ms: 50,
            blink_interval_max_ms: 50,
            blink_duration_ms: 200,
            ..StateConfig::default_config()
        });
        state.push_emotion("\u{1F929}"); // 🤩
        state.tick_wall(std::time::Duration::from_millis(120));
        let pose = state.compute_pose();
        assert_eq!(pose.eye_open, 1.0, "emotion suppresses blink");
    }

    /// Unknown emojis are silently ignored — keeps the renderer
    /// stable when the LLM emits emojis the rigging doesn't cover.
    #[test]
    fn unknown_emoji_is_no_op() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_emotion("\u{1F47D}"); // 👽 (not in default map)
        let pose = state.compute_pose();
        assert_eq!(pose.expression_id, "neutral");
        assert_eq!(pose.motion_group, "Idle");
    }

    /// While audio is playing (recent clock update) without an
    /// emotion, the motion group is `Talking`. Mirrors persona-
    /// engine's NEUTRAL_TALKING_MOTION_GROUP behavior.
    #[test]
    fn talking_motion_group_picked_during_active_audio() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([0.3; ARKIT_52], 100, None));
        state.update_audio_clock(100);
        let pose = state.compute_pose();
        assert_eq!(pose.expression_id, "neutral");
        assert_eq!(pose.motion_group, "Talking");
    }

    /// Default ARKit→VBridger mapper writes the documented param
    /// set with sane values. Mirrors persona-engine's
    /// `ARKitToLive2DMapper.cs` indices (smileLeft=23, smileRight=24)
    /// + ease_in on jaw/mouth-open axes.
    #[test]
    fn default_mapper_writes_expected_params() {
        let mut arkit = [0.0f32; ARKIT_52];
        arkit[17] = 0.6; // jawOpen
        arkit[23] = 0.4; // mouthSmileLeft
        arkit[24] = 0.4; // mouthSmileRight
        let mut params = HashMap::new();
        DefaultArkitMapper.map(&arkit, &mut params);
        // snappy_open(0.6) saturates at HIGH=0.55 → 1.0
        assert!(approx(params["ParamJawOpen"], 1.0, 1e-3));
        // mouth_open_raw = (0.6 - 0) - 0 + 0 = 0.6 → also saturates → 1.0
        assert!(approx(params["ParamMouthOpenY"], 1.0, 1e-3));
        // Form: (2 + smile_l + smile_r) / 4 = (2 + 0.4 + 0.4) / 4 = 0.7
        assert!(approx(params["ParamMouthForm"], 0.7, 1e-3));
    }

    /// Default emotion map covers every emoji in the persona-engine
    /// canonical table.
    #[test]
    fn default_emotion_map_covers_persona_engine_table() {
        let m = default_emotion_mapping();
        let want = [
            "\u{1F60A}",
            "\u{1F929}",
            "\u{1F60E}",
            "\u{1F60F}",
            "\u{1F4AA}",
            "\u{1F633}",
            "\u{1F632}",
            "\u{1F914}",
            "\u{1F440}",
            "\u{1F624}",
            "\u{1F622}",
            "\u{1F605}",
            "\u{1F644}",
            "\u{1F495}",
            "\u{1F602}",
            "\u{1F525}",
            "\u{2728}",
        ];
        for emoji in want {
            assert!(m.contains_key(emoji), "missing emoji {emoji:?}");
        }
        // Spot-check one mapping (matches Live2D.md table).
        assert_eq!(m["\u{1F622}"].expression_id, "sad");
        assert_eq!(m["\u{1F622}"].motion_group, "Sad");
    }

    /// Audio clock that points before any ring entry yields the
    /// neutral mouth (no extrapolation).
    #[test]
    fn audio_clock_before_first_keyframe_holds_neutral() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([0.9; ARKIT_52], 1_000, None));
        state.update_audio_clock(500); // before the keyframe
        let pose = state.compute_pose();
        assert!(approx(pose.mouth_value("ParamJawOpen"), 0.0, 1e-6));
    }

    /// Audio clock past the last ring entry holds the last frame
    /// (no extrapolation). Mapper applies ease_in.
    #[test]
    fn audio_clock_past_last_keyframe_holds_last_frame() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        state.push_blendshape(BlendshapeFrame::new([0.7; ARKIT_52], 100, None));
        state.update_audio_clock(120);
        let pose = state.compute_pose();
        // snappy_open(0.7) saturates at HIGH=0.55 → 1.0
        assert!(approx(pose.mouth_value("ParamJawOpen"), 1.0, 1e-3));
    }

    /// Pushing keyframes out of pts order still produces correct
    /// lerps — the ring sorts by pts at sample time. Mapper
    /// applies ease_in to jaw_open lerp value.
    #[test]
    fn handles_out_of_order_keyframe_pushes() {
        let mut state = Live2DRenderState::new(StateConfig::default_config());
        // Push pts 200 first, pts 100 second. Sorted internally.
        state.push_blendshape(BlendshapeFrame::new([1.0; ARKIT_52], 200, None));
        state.push_blendshape(BlendshapeFrame::new([0.0; ARKIT_52], 100, None));
        state.update_audio_clock(150);
        let pose = state.compute_pose();
        // snappy_open(0.5): in linear range. s = (0.5 - 0.10) / (0.55 - 0.10) ≈ 0.889
        // smoothstep(s) ≈ 0.940
        let s = (0.5f32 - 0.10) / (0.55 - 0.10);
        let expected = s * s * (3.0 - 2.0 * s);
        assert!(approx(pose.mouth_value("ParamJawOpen"), expected, 1e-3));
    }
}
