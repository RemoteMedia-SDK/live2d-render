//! ARKit-52 blendshape constants, `BlendshapeFrame` envelope, and the
//! response-curve helpers used by `state.rs`'s ARKit→VBridger mapper.
//!
//! Inlined from `remotemedia-core::nodes::lip_sync::{blendshape,
//! audio2face::response_curves}` so this Path-3 plugin doesn't link
//! the host crate.

use remotemedia_plugin_sdk::types::Error;
use serde_json::{json, Value};

/// Number of ARKit blendshapes (canonical face animation set).
pub const ARKIT_52: usize = 52;

/// ARKit blendshape names in canonical order. Index `i` of the
/// `arkit_52` array corresponds to `ARKIT_BLENDSHAPE_NAMES[i]`.
///
/// Mirrors Apple's `ARFaceAnchor.BlendShapeLocation` enumeration.
pub const ARKIT_BLENDSHAPE_NAMES: [&str; ARKIT_52] = [
    // Eyes (8)
    "eyeBlinkLeft",
    "eyeLookDownLeft",
    "eyeLookInLeft",
    "eyeLookOutLeft",
    "eyeLookUpLeft",
    "eyeSquintLeft",
    "eyeWideLeft",
    "eyeBlinkRight",
    // 8..14
    "eyeLookDownRight",
    "eyeLookInRight",
    "eyeLookOutRight",
    "eyeLookUpRight",
    "eyeSquintRight",
    "eyeWideRight",
    // Jaw / mouth area
    "jawForward",
    "jawLeft",
    // 16..
    "jawRight",
    "jawOpen",
    "mouthClose",
    "mouthFunnel",
    "mouthPucker",
    "mouthLeft",
    "mouthRight",
    "mouthSmileLeft",
    "mouthSmileRight",
    "mouthFrownLeft",
    "mouthFrownRight",
    "mouthDimpleLeft",
    "mouthDimpleRight",
    "mouthStretchLeft",
    "mouthStretchRight",
    "mouthRollLower",
    "mouthRollUpper",
    "mouthShrugLower",
    "mouthShrugUpper",
    "mouthPressLeft",
    "mouthPressRight",
    "mouthLowerDownLeft",
    "mouthLowerDownRight",
    "mouthUpperUpLeft",
    "mouthUpperUpRight",
    // Brows (4)
    "browDownLeft",
    "browDownRight",
    "browInnerUp",
    "browOuterUpLeft",
    "browOuterUpRight",
    // Cheeks (3)
    "cheekPuff",
    "cheekSquintLeft",
    "cheekSquintRight",
    // Nose (2)
    "noseSneerLeft",
    "noseSneerRight",
    // Tongue (1)
    "tongueOut",
];

const _ASSERT_ARKIT_NAMES_LEN: () = assert!(ARKIT_BLENDSHAPE_NAMES.len() == ARKIT_52);

/// One timed blendshape keyframe — the unit a `LipSyncNode` emits per
/// output tick. The renderer treats consecutive keyframes as a
/// sampleable timeline keyed by `pts_ms` (audio playback time).
#[derive(Debug, Clone, PartialEq)]
pub struct BlendshapeFrame {
    /// 52 ARKit blendshape activations, indexed per
    /// [`ARKIT_BLENDSHAPE_NAMES`].
    pub arkit_52: [f32; ARKIT_52],
    /// Presentation timestamp (ms) — matches the audio frame the
    /// keyframe was derived from, NOT wall time.
    pub pts_ms: u64,
    /// Conversational turn id, forwarded if upstream metadata had one.
    pub turn_id: Option<u64>,
}

impl BlendshapeFrame {
    /// Build a frame; the array is borrowed in by value.
    pub fn new(arkit_52: [f32; ARKIT_52], pts_ms: u64, turn_id: Option<u64>) -> Self {
        Self {
            arkit_52,
            pts_ms,
            turn_id,
        }
    }

    /// All-zero blendshapes — the neutral pose.
    pub fn neutral(pts_ms: u64) -> Self {
        Self::new([0.0; ARKIT_52], pts_ms, None)
    }

    /// Encode the frame as the canonical `RuntimeData::Json` payload.
    pub fn to_json(&self) -> Value {
        let mut v = json!({
            "kind": "blendshapes",
            "arkit_52": self.arkit_52.as_slice(),
            "pts_ms": self.pts_ms,
        });
        if let Some(turn) = self.turn_id {
            v["turn_id"] = json!(turn);
        }
        v
    }

    /// Inverse of [`Self::to_json`].
    pub fn from_json(v: &Value) -> Result<Self, Error> {
        let kind = v.get("kind").and_then(|k| k.as_str()).unwrap_or("");
        if kind != "blendshapes" {
            return Err(Error::InvalidData(format!(
                "BlendshapeFrame::from_json: expected kind='blendshapes', got {:?}",
                kind
            )));
        }
        let arr = v
            .get("arkit_52")
            .and_then(|a| a.as_array())
            .ok_or_else(|| Error::InvalidData("missing arkit_52 array".into()))?;
        if arr.len() != ARKIT_52 {
            return Err(Error::InvalidData(format!(
                "arkit_52 must have {} entries, got {}",
                ARKIT_52,
                arr.len()
            )));
        }
        let mut arkit_52 = [0.0f32; ARKIT_52];
        for (i, item) in arr.iter().enumerate() {
            arkit_52[i] = item
                .as_f64()
                .ok_or_else(|| Error::InvalidData(format!("arkit_52[{}] is not a number", i)))?
                as f32;
        }
        let pts_ms = v
            .get("pts_ms")
            .and_then(|p| p.as_u64())
            .ok_or_else(|| Error::InvalidData("missing or non-u64 pts_ms".into()))?;
        let turn_id = v.get("turn_id").and_then(|t| t.as_u64());
        Ok(Self::new(arkit_52, pts_ms, turn_id))
    }
}

/// Response curves — Rust port of persona-engine's
/// `ResponseCurves.cs`. Inlined here so `state.rs`'s
/// ARKit→VBridger mapper can call `center_weighted` without pulling
/// in the host's lip-sync tree.
pub mod response_curves {
    /// Hermite spline `f(t) = t · (2 − t)`. Steep tangent at `t=0`,
    /// flat at `t=1`. Output range `[0, 1]`.
    #[allow(dead_code)]
    pub fn ease_in(t: f32) -> f32 {
        let t = t.clamp(0.0, 1.0);
        t * (2.0 - t)
    }

    /// Three-key center-weighted Hermite curve. Keys at
    /// `(lo, lo, m=0)`, `(0, 0, m=2)`, `(hi, hi, m=0)`. Steep
    /// through zero, flat at the extremes. Input clamped to
    /// `[lo, hi]`.
    pub fn center_weighted(t: f32, lo: f32, hi: f32) -> f32 {
        let t = t.clamp(lo, hi);
        if t <= 0.0 {
            let span = -lo;
            if span < 1e-6 {
                return t;
            }
            let s = (t - lo) / span;
            hermite_segment(s, lo, 0.0, 0.0, 2.0 * span)
        } else {
            let span = hi;
            if span < 1e-6 {
                return t;
            }
            let s = t / span;
            hermite_segment(s, 0.0, hi, 2.0 * span, 0.0)
        }
    }

    fn hermite_segment(s: f32, p0: f32, p1: f32, m0: f32, m1: f32) -> f32 {
        let s2 = s * s;
        let s3 = s2 * s;
        let h00 = 2.0 * s3 - 3.0 * s2 + 1.0;
        let h10 = s3 - 2.0 * s2 + s;
        let h01 = -2.0 * s3 + 3.0 * s2;
        let h11 = s3 - s2;
        h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
    }
}
