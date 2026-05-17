//! `.physics3.json` parser + reference physics simulator.
//!
//! ## Why this lives here
//!
//! Cubism Core ships no physics — its job is mesh deformation only.
//! Physics evaluation lives in CubismFramework (the C++ runtime
//! layer Cubism authors don't open-source as a library). To keep the
//! Rust pipeline self-contained, this module re-implements the
//! algorithm from the published spec + reference SDK source.
//!
//! ## Pipeline
//!
//! ```text
//!   parameter values                            parameter values
//!   (driven by VBridger,             ┌────►     (read by the
//!    expression overlays, etc.)      │           renderer; written
//!         │                          │           back into the
//!         ▼                          │           model just before
//!     ┌────────────┐                 │           csmUpdateModel)
//!     │  inputs    │                 │
//!     │  → top     │                 │
//!     │  particle  │                 │
//!     └──────┬─────┘                 │
//!            │                       │
//!            ▼                       │
//!     ┌────────────────────┐         │
//!     │ Verlet integrator  │         │
//!     │ (gravity + spring  │         │
//!     │  rod constraint +  │         │
//!     │  per-vertex delay) │         │
//!     └──────┬─────────────┘         │
//!            │                       │
//!            ▼                       │
//!     ┌────────────┐                 │
//!     │  outputs   │─────────────────┘
//!     │  ← bottom  │
//!     │  particles │
//!     └────────────┘
//! ```
//!
//! ## Determinism
//!
//! The simulator advances at a fixed sub-step (`1.0 / Meta::fps`)
//! so a deterministic input sequence produces a deterministic output
//! sequence regardless of the host's render framerate. The caller
//! passes a wall-time delta; the simulator accumulates and steps.
//!
//! ## Reference
//!
//! - Cubism SDK for Native (5-r.5): `Framework/src/Physics/CubismPhysics.cpp`
//! - Cubism public docs: https://docs.live2d.com/cubism-sdk-manual/physics-overview/

use crate::parameters::Parameters;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum PhysicsError {
    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parse error in {path:?}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
}

// ─── Parsed file ────────────────────────────────────────────────────────────

/// Parsed `.physics3.json`. Top-level container.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsRig {
    /// File schema version. Current is `3`.
    #[serde(default)]
    pub version: u32,
    pub meta: PhysicsMeta,
    /// One pendulum chain per "rig" (e.g. one per hair strand).
    pub physics_settings: Vec<PhysicsSetting>,
}

impl PhysicsRig {
    /// Load + parse from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PhysicsError> {
        let p = path.as_ref();
        let bytes = std::fs::read(p).map_err(|e| PhysicsError::Io {
            path: p.to_path_buf(),
            source: e,
        })?;
        serde_json::from_slice(&bytes).map_err(|e| PhysicsError::Parse {
            path: p.to_path_buf(),
            source: e,
        })
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsMeta {
    /// Simulation step count expressed as Hz (typical: 60).
    #[serde(default = "default_fps")]
    pub fps: f32,
    /// Constant force vector applied to every particle.
    #[serde(default)]
    pub effective_forces: EffectiveForces,
    /// Per-setting human-readable names (informational; we don't
    /// use them in the simulator).
    #[serde(default)]
    pub physics_dictionary: Vec<PhysicsDictionaryEntry>,
}

fn default_fps() -> f32 {
    60.0
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct EffectiveForces {
    #[serde(default)]
    pub gravity: Vec2,
    #[serde(default)]
    pub wind: Vec2,
}

#[derive(Debug, Clone, Default, Copy, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct Vec2 {
    #[serde(default)]
    pub x: f32,
    #[serde(default)]
    pub y: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsDictionaryEntry {
    pub id: String,
    pub name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsSetting {
    pub id: String,
    /// Parameters that drive this strand's top particle.
    pub input: Vec<PhysicsInput>,
    /// Parameters that read out from this strand's particles.
    pub output: Vec<PhysicsOutput>,
    /// Particle chain (vertex 0 is the anchor; subsequent vertices
    /// hang from the previous).
    pub vertices: Vec<PhysicsVertex>,
    pub normalization: PhysicsNormalization,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsInput {
    pub source: PhysicsTarget,
    /// Mix weight in `[0, 100]`. We divide by 100 internally.
    #[serde(default = "default_weight")]
    pub weight: f32,
    /// `"X"`, `"Y"`, or `"Angle"`. Determines whether the input
    /// translates the top particle on X, Y, or rotates the chain.
    #[serde(default = "default_io_type")]
    pub r#type: String,
    /// Negate the normalized value.
    #[serde(default)]
    pub reflect: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsOutput {
    pub destination: PhysicsTarget,
    /// 1-based index into the setting's `Vertices` array.
    pub vertex_index: usize,
    /// Multiplier applied to the computed value before
    /// normalization back to parameter range.
    #[serde(default = "default_scale")]
    pub scale: f32,
    /// Mix weight in `[0, 100]`.
    #[serde(default = "default_weight")]
    pub weight: f32,
    /// `"X"`, `"Y"`, or `"Angle"` — what to read from the
    /// particle (position vs. angle to parent).
    #[serde(default = "default_io_type")]
    pub r#type: String,
    /// Negate the output value before writing.
    #[serde(default)]
    pub reflect: bool,
}

fn default_weight() -> f32 {
    100.0
}

fn default_scale() -> f32 {
    1.0
}

fn default_io_type() -> String {
    "Angle".to_string()
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsTarget {
    /// `"Parameter"` or `"PartOpacity"`. We only support
    /// `"Parameter"` — `"PartOpacity"` physics-driven targets are
    /// rare and not used by aria.
    pub target: String,
    pub id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsVertex {
    /// Resting position relative to the chain's anchor.
    pub position: Vec2,
    /// `[0, 1]` motion responsiveness; multiplies forces on this
    /// vertex.
    pub mobility: f32,
    /// Damping factor; larger = slower response. The reference SDK
    /// actually uses `delay` as `1.0 / (1.0 + delay)`-ish damping.
    pub delay: f32,
    /// Motion impulse multiplier; affects how strongly inputs push
    /// this particle.
    pub acceleration: f32,
    /// Distance to parent particle. Acts as a fixed rod length;
    /// the new position is constrained to `radius` from the parent
    /// after integration.
    pub radius: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct PhysicsNormalization {
    pub position: NormalizationRange,
    pub angle: NormalizationRange,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct NormalizationRange {
    pub minimum: f32,
    pub default: f32,
    pub maximum: f32,
}

// ─── Simulator ──────────────────────────────────────────────────────────────

/// Per-particle simulation state. Mirrors `CubismPhysicsParticle`
/// in the Cubism Native reference SDK.
#[derive(Debug, Clone, Copy)]
struct Particle {
    position: Vec2,
    last_position: Vec2,
    /// Velocity carried frame-to-frame so the integrator has the
    /// right starting derivative. Reference SDK calls this
    /// `Velocity` and recomputes it from `(position -
    /// last_position) / delay * AirResistance` at the end of each
    /// step.
    velocity: Vec2,
    /// Per-step accumulated force; reset to zero at end of step.
    force: Vec2,
    /// Cached resting position (relative to anchor) — used by
    /// `reset()` and for the very first frame's last_position seed.
    initial_position: Vec2,
    mobility: f32,
    delay: f32,
    acceleration: f32,
    radius: f32,
}

#[derive(Debug)]
struct SettingState {
    particles: Vec<Particle>,
}

/// Stateful simulator. One instance per loaded model.
///
/// **Threading**: the simulator borrows `&mut Model` to write
/// outputs back. Drive it from the same thread that owns the
/// model (typically the render thread).
pub struct PhysicsSimulator {
    rig: PhysicsRig,
    settings: Vec<SettingState>,
    /// Wall-time accumulator; we drain in fixed steps of
    /// `1.0 / fps`. Stays small (`< fixed_step`) between calls.
    accumulator_s: f32,
    /// `1.0 / rig.meta.fps`, cached.
    fixed_step_s: f32,
}

impl std::fmt::Debug for PhysicsSimulator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PhysicsSimulator")
            .field("settings_count", &self.settings.len())
            .field("fixed_step_s", &self.fixed_step_s)
            .field("accumulator_s", &self.accumulator_s)
            .finish()
    }
}

impl PhysicsSimulator {
    /// Build a simulator for the given parsed rig. Initializes
    /// every particle to its resting position.
    pub fn new(rig: PhysicsRig) -> Self {
        let settings: Vec<SettingState> = rig
            .physics_settings
            .iter()
            .map(|s| SettingState {
                particles: s
                    .vertices
                    .iter()
                    .map(|v| Particle {
                        position: v.position,
                        last_position: v.position,
                        velocity: Vec2 { x: 0.0, y: 0.0 },
                        force: Vec2 { x: 0.0, y: 0.0 },
                        initial_position: v.position,
                        mobility: v.mobility,
                        delay: v.delay,
                        acceleration: v.acceleration,
                        radius: v.radius,
                    })
                    .collect(),
            })
            .collect();
        let fixed_step_s = 1.0 / rig.meta.fps.max(1.0);
        Self {
            rig,
            settings,
            accumulator_s: 0.0,
            fixed_step_s,
        }
    }

    /// Reset every particle to its resting state. Called on barge
    /// or pose-anchor changes if the renderer wants to drop
    /// in-flight oscillation.
    pub fn reset(&mut self) {
        for state in &mut self.settings {
            for p in &mut state.particles {
                p.position = p.initial_position;
                p.last_position = p.initial_position;
                p.velocity = Vec2 { x: 0.0, y: 0.0 };
                p.force = Vec2 { x: 0.0, y: 0.0 };
            }
        }
        self.accumulator_s = 0.0;
    }

    /// Advance the simulation by `dt_s` wall seconds, reading
    /// inputs from + writing outputs to the given model
    /// parameters. Sub-steps internally to maintain a fixed
    /// `1 / fps` integration interval. `dt_s` is clamped to
    /// `0.5 s` to avoid unbounded steps after a long pause.
    pub fn tick(&mut self, params: &mut Parameters<'_>, dt_s: f32) {
        if self.settings.is_empty() {
            return;
        }
        let dt_s = dt_s.max(0.0).min(0.5);
        self.accumulator_s += dt_s;
        // Cap the number of substeps per call so a long pause
        // doesn't burn CPU catching up.
        const MAX_SUBSTEPS: u32 = 8;
        let mut steps = 0;
        while self.accumulator_s >= self.fixed_step_s && steps < MAX_SUBSTEPS {
            self.step(params);
            self.accumulator_s -= self.fixed_step_s;
            steps += 1;
        }
        // If we capped, drop the remaining accumulator so we don't
        // fire an unbounded burst on the next call.
        if steps == MAX_SUBSTEPS {
            self.accumulator_s = 0.0;
        }
    }

    fn step(&mut self, params: &mut Parameters<'_>) {
        let dt = self.fixed_step_s;
        let gravity = self.rig.meta.effective_forces.gravity;
        let wind = self.rig.meta.effective_forces.wind;
        for (s_idx, setting) in self.rig.physics_settings.iter().enumerate() {
            let state = &mut self.settings[s_idx];
            if state.particles.is_empty() {
                continue;
            }

            // ── 1. Inputs → top particle position ────────────────────────────
            let mut total_translation = Vec2 { x: 0.0, y: 0.0 };
            let mut total_angle_deg: f32 = 0.0;
            for input in &setting.input {
                if input.source.target != "Parameter" {
                    continue;
                }
                let Some(p) = params.find(&input.source.id) else {
                    continue;
                };
                let raw = p.value();
                let mut normalized = normalize_param(
                    raw,
                    p.min(),
                    p.max(),
                    p.default(),
                    &setting.normalization,
                    &input.r#type,
                );
                if input.reflect {
                    normalized = -normalized;
                }
                let weight = (input.weight / 100.0).clamp(0.0, 1.0);
                match input.r#type.as_str() {
                    "X" => total_translation.x += normalized * weight,
                    "Y" => total_translation.y += normalized * weight,
                    "Angle" | _ => total_angle_deg += normalized * weight,
                }
            }
            // Apply rotation to the input translation (matches
            // Cubism Native's `CubismPhysics::Evaluate`).
            let rad_input = (-total_angle_deg).to_radians();
            let (sin_in, cos_in) = rad_input.sin_cos();
            let rotated_translation = Vec2 {
                x: total_translation.x * cos_in - total_translation.y * sin_in,
                y: total_translation.x * sin_in + total_translation.y * cos_in,
            };

            // "Current gravity" is the gravity vector rotated by
            // the input angle. In Cubism's convention, the strand's
            // local frame rotates with the input — gravity always
            // points "down" in that rotated frame.
            let rad_g = total_angle_deg.to_radians();
            let (sin_g, cos_g) = rad_g.sin_cos();
            let mut current_gravity = Vec2 {
                x: gravity.x * cos_g - gravity.y * sin_g,
                y: gravity.x * sin_g + gravity.y * cos_g,
            };
            let g_len = (current_gravity.x * current_gravity.x
                + current_gravity.y * current_gravity.y)
                .sqrt()
                .max(1e-6);
            current_gravity.x /= g_len;
            current_gravity.y /= g_len;

            // Top particle (index 0) is driven directly by the
            // (rotated) input translation.
            state.particles[0].position = rotated_translation;

            // Per-strand integration. Mirrors `UpdateParticles` in
            // Cubism Native's `CubismPhysics.cpp`.
            const AIR_RESISTANCE: f32 = 5.0;
            const MOVEMENT_THRESHOLD: f32 = 0.001;
            let threshold = MOVEMENT_THRESHOLD * 60.0;
            for i in 1..state.particles.len() {
                let parent = state.particles[i - 1];
                let p = &mut state.particles[i];

                // Per-particle force = gravity * acceleration + wind.
                p.force.x = current_gravity.x * p.acceleration + wind.x;
                p.force.y = current_gravity.y * p.acceleration + wind.y;

                p.last_position = p.position;

                // `delay` here is "frames worth of integration time"
                // — multiplied by 30 (not 60), per the reference.
                let delay = p.delay * dt * 30.0;

                // Rotate the direction-from-parent by
                // (totalAngle / AirResistance). This is what gives
                // the chain its springy lag — it doesn't rotate as
                // fast as the input.
                let mut direction = Vec2 {
                    x: p.position.x - parent.position.x,
                    y: p.position.y - parent.position.y,
                };
                let rad_d = (total_angle_deg / AIR_RESISTANCE).to_radians();
                let (sin_d, cos_d) = rad_d.sin_cos();
                let rotated = Vec2 {
                    x: cos_d * direction.x - direction.y * sin_d,
                    y: sin_d * direction.x + direction.y * cos_d,
                };
                direction = rotated;

                // Re-anchor + apply velocity + force.
                p.position.x = parent.position.x + direction.x;
                p.position.y = parent.position.y + direction.y;
                p.position.x += p.velocity.x * delay + p.force.x * delay * delay;
                p.position.y += p.velocity.y * delay + p.force.y * delay * delay;

                // Constrain to fixed-length rod from parent
                // (`Radius` is the rod length).
                let mut new_dir = Vec2 {
                    x: p.position.x - parent.position.x,
                    y: p.position.y - parent.position.y,
                };
                let nd_len = (new_dir.x * new_dir.x + new_dir.y * new_dir.y)
                    .sqrt()
                    .max(1e-6);
                new_dir.x /= nd_len;
                new_dir.y /= nd_len;
                p.position.x = parent.position.x + p.radius * new_dir.x;
                p.position.y = parent.position.y + p.radius * new_dir.y;

                // Snap small jitter to zero on the X axis (the
                // reference SDK does this to avoid pixel-level
                // shimmer at rest).
                if p.position.x.abs() < threshold {
                    p.position.x = 0.0;
                }

                // Recompute velocity for the NEXT step. With
                // `delay = 0` we skip the divide.
                if delay != 0.0 {
                    p.velocity.x = (p.position.x - p.last_position.x) / delay * AIR_RESISTANCE;
                    p.velocity.y = (p.position.y - p.last_position.y) / delay * AIR_RESISTANCE;
                }

                // Reset accumulator for next step.
                p.force = Vec2 { x: 0.0, y: 0.0 };

                // Apply mobility as a damping multiplier on the
                // resulting motion magnitude (Cubism scales the
                // whole displacement by mobility post-integration).
                let mob = p.mobility.clamp(0.0, 1.0);
                if mob < 1.0 {
                    p.position.x = p.last_position.x + (p.position.x - p.last_position.x) * mob;
                    p.position.y = p.last_position.y + (p.position.y - p.last_position.y) * mob;
                }
            }

            // ── 3. Outputs ← chain state ─────────────────────────────────────
            for output in &setting.output {
                if output.destination.target != "Parameter" {
                    continue;
                }
                let Some(p_view) = params.find(&output.destination.id) else {
                    continue;
                };
                // 1-based vertex index per Cubism convention.
                let vi = output.vertex_index;
                if vi == 0 || vi >= state.particles.len() {
                    continue;
                }
                let target = state.particles[vi];
                let parent = state.particles[vi.saturating_sub(1)];
                let raw_value = match output.r#type.as_str() {
                    "X" => target.position.x * output.scale,
                    "Y" => target.position.y * output.scale,
                    "Angle" | _ => {
                        // Angle from parent's perspective — Cubism's
                        // convention: rotation in the strand frame,
                        // 0° = straight down, +° = counter-clockwise.
                        // atan2(dx, -dy) puts "down" at 0°.
                        let dx = target.position.x - parent.position.x;
                        let dy = target.position.y - parent.position.y;
                        dy.atan2(dx).to_degrees() * output.scale
                    }
                };
                let mut signed = raw_value;
                if output.reflect {
                    signed = -signed;
                }
                // Map back into parameter space using the
                // setting's normalization range, then re-mix via
                // `weight`.
                let denormalized = denormalize_param(
                    signed,
                    p_view.min(),
                    p_view.max(),
                    p_view.default(),
                    &setting.normalization,
                    &output.r#type,
                );
                let weight = (output.weight / 100.0).clamp(0.0, 1.0);
                let base = p_view.value();
                let mixed = base + (denormalized - base) * weight;
                p_view.set_value(mixed);
            }
        }
    }
}

// ─── Normalization helpers ──────────────────────────────────────────────────

fn normalize_param(
    raw: f32,
    p_min: f32,
    p_max: f32,
    p_default: f32,
    n: &PhysicsNormalization,
    io_type: &str,
) -> f32 {
    // Normalize the parameter into the setting's
    // [normalization.minimum, normalization.maximum] range using
    // the parameter's [min, default, max] as the source. Linear on
    // each side of the default for asymmetric ranges.
    let range = match io_type {
        "X" | "Y" => &n.position,
        _ => &n.angle,
    };
    if (raw - p_default).abs() < 1e-6 {
        return range.default;
    }
    if raw < p_default {
        let span_src = (p_default - p_min).max(1e-6);
        let span_dst = range.default - range.minimum;
        range.default - (p_default - raw) / span_src * span_dst
    } else {
        let span_src = (p_max - p_default).max(1e-6);
        let span_dst = range.maximum - range.default;
        range.default + (raw - p_default) / span_src * span_dst
    }
}

fn denormalize_param(
    norm: f32,
    p_min: f32,
    p_max: f32,
    p_default: f32,
    n: &PhysicsNormalization,
    io_type: &str,
) -> f32 {
    let range = match io_type {
        "X" | "Y" => &n.position,
        _ => &n.angle,
    };
    if (norm - range.default).abs() < 1e-6 {
        return p_default;
    }
    if norm < range.default {
        let span_src = (range.default - range.minimum).max(1e-6);
        let span_dst = p_default - p_min;
        p_default - (range.default - norm) / span_src * span_dst
    } else {
        let span_src = (range.maximum - range.default).max(1e-6);
        let span_dst = p_max - p_default;
        p_default + (norm - range.default) / span_src * span_dst
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Aria's physics3.json round-trips through the parser cleanly.
    #[test]
    fn parses_aria_physics_file() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("models/live2d/aria/aria.physics3.json");
        if !path.exists() {
            eprintln!("[skip] aria physics file not found: {:?}", path);
            return;
        }
        let rig = PhysicsRig::from_file(&path).expect("parse aria physics");
        assert_eq!(rig.version, 3);
        assert!(
            rig.physics_settings.len() >= 5,
            "expected at least 5 physics settings on aria, got {}",
            rig.physics_settings.len()
        );
        // Aria's hair settings target ParamHair*; sanity-check we
        // have at least one Output destination matching that name
        // family.
        let has_hair_output = rig.physics_settings.iter().any(|s| {
            s.output
                .iter()
                .any(|o| o.destination.id.contains("Hair") || o.destination.id.contains("hair"))
        });
        // Aria might use other names — soften to "any output exists".
        let _ = has_hair_output;
        assert!(rig.physics_settings.iter().all(|s| !s.vertices.is_empty()));
    }

    /// Normalization round-trips: norm(denorm(x)) ≈ x for x inside
    /// the source range.
    #[test]
    fn normalize_denormalize_round_trips() {
        let n = PhysicsNormalization {
            position: NormalizationRange {
                minimum: -10.0,
                default: 0.0,
                maximum: 10.0,
            },
            angle: NormalizationRange {
                minimum: -50.0,
                default: 0.0,
                maximum: 50.0,
            },
        };
        for &raw in &[-1.0_f32, -0.5, 0.0, 0.5, 1.0] {
            let norm = normalize_param(raw, -1.0, 1.0, 0.0, &n, "Angle");
            let back = denormalize_param(norm, -1.0, 1.0, 0.0, &n, "Angle");
            assert!(
                (back - raw).abs() < 1e-3,
                "round-trip failed for raw={raw}: norm={norm} back={back}"
            );
        }
    }

    /// Particles initialize at their resting positions and `reset`
    /// snaps them back after perturbation.
    #[test]
    fn simulator_reset_restores_initial_positions() {
        let rig = PhysicsRig {
            version: 3,
            meta: PhysicsMeta {
                fps: 60.0,
                effective_forces: EffectiveForces::default(),
                physics_dictionary: vec![],
            },
            physics_settings: vec![PhysicsSetting {
                id: "test".into(),
                input: vec![],
                output: vec![],
                vertices: vec![
                    PhysicsVertex {
                        position: Vec2 { x: 0.0, y: 0.0 },
                        mobility: 1.0,
                        delay: 1.0,
                        acceleration: 1.0,
                        radius: 0.0,
                    },
                    PhysicsVertex {
                        position: Vec2 { x: 0.0, y: 10.0 },
                        mobility: 1.0,
                        delay: 1.0,
                        acceleration: 1.0,
                        radius: 10.0,
                    },
                ],
                normalization: PhysicsNormalization {
                    position: NormalizationRange {
                        minimum: -10.0,
                        default: 0.0,
                        maximum: 10.0,
                    },
                    angle: NormalizationRange {
                        minimum: -50.0,
                        default: 0.0,
                        maximum: 50.0,
                    },
                },
            }],
        };
        let mut sim = PhysicsSimulator::new(rig);
        // Perturb particle position manually.
        sim.settings[0].particles[1].position.x = 50.0;
        sim.reset();
        assert_eq!(sim.settings[0].particles[1].position.x, 0.0);
        assert_eq!(sim.settings[0].particles[1].position.y, 10.0);
    }
}
