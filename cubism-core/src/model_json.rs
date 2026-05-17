//! `.model3.json` — Cubism's top-level model manifest.
//!
//! A `.model3.json` references every other file in a Live2D model
//! bundle: the rigged mesh (`.moc3`), textures, physics, display
//! info, expression overrides (`.exp3.json`), and motion clips
//! (`.motion3.json`). This module owns the parser + path resolver
//! so callers don't have to reach into `serde_json` directly.
//!
//! The shape mirrors the canonical layout documented in the Cubism
//! SDK + matched to the persona-engine C# reference loader at
//! [`external/handcrafted-persona-engine/.../ModelSettingObj.cs`].
//! Aria validates every field that's exercised here.
//!
//! # What this module does NOT do
//!
//! - Doesn't load the `.moc3`. That's [`Moc::load_from_file`].
//! - Doesn't apply expression / motion data — just exposes parsed
//!   structure. The renderer (M4.4) evaluates motions per tick.
//! - Doesn't load textures / physics. Just exposes resolved paths.
//! - Doesn't run physics simulation. Cubism Core has no physics
//!   API; physics evaluation lives in CubismFramework. We expose
//!   the path; an evaluator is a follow-up.

use crate::Error as CubismError;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Errors specific to the model3.json + sub-file parsers.
///
/// `Error::Io` and `Error::Parse` carry the file path that triggered
/// them so callers can surface useful diagnostics on bundle issues
/// (typo'd file references, malformed JSON, etc.).
#[derive(Debug, thiserror::Error)]
pub enum ModelJsonError {
    /// I/O error while reading a JSON file off disk.
    #[error("io error reading {path}: {source}")]
    Io {
        /// Path of the file we tried to read.
        path: PathBuf,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },

    /// JSON parse error in a file (model3, exp3, or motion3).
    #[error("json parse error in {path}: {source}")]
    Parse {
        /// Path of the file we tried to parse.
        path: PathBuf,
        /// Underlying serde_json error.
        #[source]
        source: serde_json::Error,
    },
}

impl From<ModelJsonError> for CubismError {
    fn from(e: ModelJsonError) -> Self {
        // Surface the underlying I/O for the standard `Error::Io`
        // when applicable; otherwise wrap as Execution-style. The
        // top-level `Error` enum doesn't have a Json variant yet,
        // so we map Parse to Io with an extracted message — this
        // keeps the errors visible without a breaking enum change.
        match e {
            ModelJsonError::Io { source, .. } => CubismError::Io(source),
            ModelJsonError::Parse { path, source } => CubismError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("model3.json parse failed in {}: {}", path.display(), source),
            )),
        }
    }
}

// ─── model3.json ─────────────────────────────────────────────────────────────

/// Parsed `.model3.json` manifest. Field names match the on-disk
/// PascalCase keys via `serde(rename_all = "PascalCase")`.
///
/// Use [`ModelJson::load`] (resolver) rather than `from_file` if you
/// want absolute paths in [`ResolvedModel`].
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct ModelJson {
    /// Manifest schema version. Cubism bumps this when the on-disk
    /// JSON structure changes. Aria + most r5-era models use `3`.
    pub version: u32,

    /// References to all sibling files in the bundle.
    pub file_references: FileReferences,

    /// Parameter / part groups by purpose (e.g. `LipSync`,
    /// `EyeBlink`). The renderer's idle-blink scheduler reads
    /// `EyeBlink` to know which params to drive.
    #[serde(default)]
    pub groups: Vec<Group>,

    /// Hit-test regions for interactive (mouse-tap) handling.
    /// Unused by the avatar pipeline; preserved so round-trip
    /// re-serialization of Aria doesn't drop fields.
    #[serde(default)]
    pub hit_areas: Vec<HitArea>,

    /// Optional layout block (CenterX/CenterY/Width/etc.). Stored
    /// as a generic JSON object since fields vary by model.
    #[serde(default)]
    pub layout: Option<serde_json::Value>,
}

impl ModelJson {
    /// Read + parse a `.model3.json` from disk. Paths inside the
    /// manifest are kept as-authored (relative to the manifest's
    /// directory). Use [`Self::load`] to additionally resolve them
    /// to absolute paths.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelJsonError> {
        let p = path.as_ref();
        let bytes = std::fs::read(p).map_err(|e| ModelJsonError::Io {
            path: p.to_path_buf(),
            source: e,
        })?;
        serde_json::from_slice(&bytes).map_err(|e| ModelJsonError::Parse {
            path: p.to_path_buf(),
            source: e,
        })
    }

    /// Read + parse + resolve. Returns a [`ResolvedModel`] whose
    /// path fields are absolute (joined with the manifest's parent
    /// directory). Most callers want this rather than [`Self::from_file`].
    pub fn load(path: impl AsRef<Path>) -> Result<ResolvedModel, ModelJsonError> {
        let p = path.as_ref();
        let manifest = Self::from_file(p)?;
        let manifest_dir = p
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Ok(ResolvedModel {
            manifest,
            manifest_dir,
            manifest_path: p.to_path_buf(),
        })
    }
}

/// Bundle file references — every other file the model needs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct FileReferences {
    /// Path to the rigged mesh (`.moc3`). Required.
    pub moc: String,

    /// Texture atlas paths in display order. The drawable
    /// `texture_index` indexes into this list.
    #[serde(default)]
    pub textures: Vec<String>,

    /// Optional physics rigging (`.physics3.json`).
    #[serde(default)]
    pub physics: Option<String>,

    /// Optional pose group definitions (`.pose3.json`).
    #[serde(default)]
    pub pose: Option<String>,

    /// Optional display-info file (`.cdi3.json`) — friendly param
    /// names + grouping for editor UIs.
    #[serde(default)]
    pub display_info: Option<String>,

    /// Optional user data (`.userdata3.json`) — typically not
    /// load-bearing for renderers.
    #[serde(default)]
    pub user_data: Option<String>,

    /// Named expressions (`.exp3.json`). Aria ships 5: `neutral`,
    /// `smug`, `happy`, `frustrated`, `sad`.
    #[serde(default)]
    pub expressions: Vec<ExpressionRef>,

    /// Motion groups (`.motion3.json`). Keys are group names
    /// (`Idle`, `Talking`, `Happy`, …); values are clip lists. The
    /// avatar emotion mapping picks a random clip per group.
    #[serde(default)]
    pub motions: std::collections::HashMap<String, Vec<MotionRef>>,
}

/// Reference to one expression file (`.exp3.json`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct ExpressionRef {
    /// Symbolic name (e.g. `happy`, `smug`). Matches the
    /// expression IDs in the persona-engine emoji map.
    pub name: String,
    /// Path to the `.exp3.json`, relative to the model3's dir.
    pub file: String,
}

/// Reference to one motion clip (`.motion3.json`).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct MotionRef {
    /// Path to the `.motion3.json`, relative to the model3's dir.
    pub file: String,
    /// Optional sibling audio file (`.wav` / `.mp3`) to play
    /// alongside the motion.
    #[serde(default)]
    pub sound: Option<String>,
    /// Fade-in time in seconds when this motion starts.
    #[serde(default)]
    pub fade_in_time: f32,
    /// Fade-out time in seconds when this motion ends.
    #[serde(default)]
    pub fade_out_time: f32,
}

/// One named group of parameter or part IDs.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct Group {
    /// `Parameter` or `Part` (Cubism's two group targets).
    pub target: String,
    /// Group name (`LipSync`, `EyeBlink`, …).
    pub name: String,
    /// Member IDs.
    pub ids: Vec<String>,
}

/// One hit-test region.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct HitArea {
    /// Drawable ID this hit area corresponds to.
    pub id: String,
    /// Name shown to the user.
    pub name: String,
}

// ─── ResolvedModel ──────────────────────────────────────────────────────────

/// A parsed `model3.json` paired with the directory it lives in,
/// so relative path lookups all resolve against the right base.
///
/// Returned by [`ModelJson::load`].
#[derive(Debug, Clone)]
pub struct ResolvedModel {
    /// The parsed manifest (still carrying relative paths).
    pub manifest: ModelJson,
    /// The directory the manifest lives in. All relative paths in
    /// `manifest` resolve against this.
    pub manifest_dir: PathBuf,
    /// The original path the manifest was loaded from. Useful for
    /// error messages.
    pub manifest_path: PathBuf,
}

impl ResolvedModel {
    /// Absolute path to the `.moc3`.
    pub fn moc_path(&self) -> PathBuf {
        self.manifest_dir.join(&self.manifest.file_references.moc)
    }

    /// Absolute paths to every texture, in display order.
    pub fn texture_paths(&self) -> Vec<PathBuf> {
        self.manifest
            .file_references
            .textures
            .iter()
            .map(|t| self.manifest_dir.join(t))
            .collect()
    }

    /// Absolute path to the `.physics3.json`, if any.
    pub fn physics_path(&self) -> Option<PathBuf> {
        self.manifest
            .file_references
            .physics
            .as_ref()
            .map(|p| self.manifest_dir.join(p))
    }

    /// Absolute path to the `.pose3.json`, if any.
    pub fn pose_path(&self) -> Option<PathBuf> {
        self.manifest
            .file_references
            .pose
            .as_ref()
            .map(|p| self.manifest_dir.join(p))
    }

    /// Absolute path to the `.cdi3.json`, if any.
    pub fn display_info_path(&self) -> Option<PathBuf> {
        self.manifest
            .file_references
            .display_info
            .as_ref()
            .map(|p| self.manifest_dir.join(p))
    }

    /// Absolute path to the `.userdata3.json`, if any.
    pub fn user_data_path(&self) -> Option<PathBuf> {
        self.manifest
            .file_references
            .user_data
            .as_ref()
            .map(|p| self.manifest_dir.join(p))
    }

    /// Absolute path to the named expression's `.exp3.json`, if it
    /// exists in the manifest.
    pub fn expression_path(&self, name: &str) -> Option<PathBuf> {
        self.manifest
            .file_references
            .expressions
            .iter()
            .find(|e| e.name == name)
            .map(|e| self.manifest_dir.join(&e.file))
    }

    /// Iterate every expression's (name, absolute path) pair.
    pub fn expressions(&self) -> impl Iterator<Item = (&str, PathBuf)> {
        self.manifest
            .file_references
            .expressions
            .iter()
            .map(move |e| (e.name.as_str(), self.manifest_dir.join(&e.file)))
    }

    /// Iterate the named motion group's (absolute path, MotionRef)
    /// pairs. Returns an empty iterator if the group doesn't exist.
    pub fn motions(&self, group: &str) -> impl Iterator<Item = (PathBuf, &MotionRef)> {
        let dir = self.manifest_dir.clone();
        self.manifest
            .file_references
            .motions
            .get(group)
            .into_iter()
            .flat_map(move |list| {
                let dir = dir.clone();
                list.iter().map(move |m| (dir.join(&m.file), m))
            })
    }

    /// All motion group names declared in the manifest.
    pub fn motion_group_names(&self) -> impl Iterator<Item = &str> {
        self.manifest
            .file_references
            .motions
            .keys()
            .map(String::as_str)
    }

    /// The IDs in the named group (e.g. `EyeBlink`), if it exists.
    pub fn group_ids(&self, name: &str) -> Option<&[String]> {
        self.manifest
            .groups
            .iter()
            .find(|g| g.name == name)
            .map(|g| g.ids.as_slice())
    }
}

// ─── Expression file ────────────────────────────────────────────────────────

/// Parsed `.exp3.json` — a list of parameter overrides applied
/// while the expression is active.
///
/// Mirrors the persona-engine emoji-driven expression system:
/// `EmotionExtractorNode` emits an emoji event → renderer maps it
/// to an expression name → renderer applies the parameters from the
/// matching `.exp3.json` on top of the lip-sync animation.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct ExpressionJson {
    /// Always `"Live2D Expression"` for v3 files. We accept the
    /// field but don't validate it — Cubism Editor stamps it.
    #[serde(default, rename = "Type")]
    pub kind: String,
    /// Fade-in time in seconds.
    #[serde(default)]
    pub fade_in_time: f32,
    /// Fade-out time in seconds.
    #[serde(default)]
    pub fade_out_time: f32,
    /// Parameter overrides applied while active.
    #[serde(default)]
    pub parameters: Vec<ExpressionParameter>,
}

impl ExpressionJson {
    /// Parse a `.exp3.json` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelJsonError> {
        let p = path.as_ref();
        let bytes = std::fs::read(p).map_err(|e| ModelJsonError::Io {
            path: p.to_path_buf(),
            source: e,
        })?;
        serde_json::from_slice(&bytes).map_err(|e| ModelJsonError::Parse {
            path: p.to_path_buf(),
            source: e,
        })
    }
}

/// One parameter override inside an expression.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct ExpressionParameter {
    /// Parameter ID (e.g. `ParamExpHappy`, `ParamBrowLY`).
    pub id: String,
    /// Override value.
    pub value: f32,
    /// Blend mode: `Add`, `Multiply`, or `Overwrite` (Cubism v3).
    /// Default is `Add`. We surface the raw string + a typed enum.
    #[serde(default = "default_blend")]
    pub blend: String,
}

impl ExpressionParameter {
    /// Decode [`Self::blend`] into a typed value.
    pub fn blend_kind(&self) -> ExpressionBlend {
        match self.blend.as_str() {
            "Add" => ExpressionBlend::Add,
            "Multiply" => ExpressionBlend::Multiply,
            "Overwrite" => ExpressionBlend::Overwrite,
            other => ExpressionBlend::Unknown(other.to_string()),
        }
    }
}

fn default_blend() -> String {
    "Add".to_string()
}

/// How an expression parameter combines with the underlying value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionBlend {
    /// `out = base + value`.
    Add,
    /// `out = base * value`.
    Multiply,
    /// `out = value` (replaces the underlying value).
    Overwrite,
    /// Unknown blend — surface the raw string rather than panic if
    /// Cubism extends the v3 schema.
    Unknown(String),
}

// ─── Motion file ────────────────────────────────────────────────────────────

/// Parsed `.motion3.json` — a clip of parameter / part / model
/// curves driven over time.
///
/// **Scope note:** this loader exposes the parsed structure but
/// does **not** evaluate curves — the renderer (M4.4) ticks them
/// per frame. Curve segments are surfaced as `Vec<f32>` (Cubism's
/// flat encoding); evaluation requires decoding the segment-type
/// markers + per-type spline math.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct MotionJson {
    /// Manifest schema version. v3 for r5-era models.
    pub version: u32,
    /// Clip metadata (duration, fps, loop flag, curve count, …).
    pub meta: MotionMeta,
    /// Parameter / part / model curves. One per driven target.
    #[serde(default)]
    pub curves: Vec<MotionCurve>,
    /// Optional embedded user-data events (timed Json blobs).
    /// Surfaced raw — we don't define a typed schema.
    #[serde(default)]
    pub user_data: Vec<serde_json::Value>,
}

impl MotionJson {
    /// Parse a `.motion3.json` from disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, ModelJsonError> {
        let p = path.as_ref();
        let bytes = std::fs::read(p).map_err(|e| ModelJsonError::Io {
            path: p.to_path_buf(),
            source: e,
        })?;
        serde_json::from_slice(&bytes).map_err(|e| ModelJsonError::Parse {
            path: p.to_path_buf(),
            source: e,
        })
    }
}

/// Metadata block from a `.motion3.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct MotionMeta {
    /// Total clip duration in seconds.
    pub duration: f32,
    /// Authoring frame rate.
    pub fps: f32,
    /// `true` if the clip loops.
    #[serde(default)]
    pub r#loop: bool,
    /// `true` if Bezier handles are restricted (used by older
    /// Cubism versions to enable a faster sampler path).
    #[serde(default)]
    pub are_beziers_restricted: bool,
    /// Total curve count (cross-check against `Curves.len()`).
    #[serde(default)]
    pub curve_count: u32,
    /// Total segment count across all curves.
    #[serde(default)]
    pub total_segment_count: u32,
    /// Total point count across all curves.
    #[serde(default)]
    pub total_point_count: u32,
    /// User-data event count.
    #[serde(default)]
    pub user_data_count: u32,
    /// User-data total size in bytes.
    #[serde(default)]
    pub total_user_data_size: u32,
    /// Optional fade-in time, can override the model3.json's value.
    #[serde(default)]
    pub fade_in_time: Option<f32>,
    /// Optional fade-out time.
    #[serde(default)]
    pub fade_out_time: Option<f32>,
}

/// One animation curve. The `target` says what kind of property
/// the curve drives:
///
/// - `"Parameter"` — drives the named parameter.
/// - `"PartOpacity"` — drives the named part's opacity.
/// - `"Model"` — drives a model-level property like `Opacity`.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "PascalCase")]
pub struct MotionCurve {
    /// Curve target kind (see above).
    pub target: String,
    /// Target ID — parameter / part / model property name.
    pub id: String,
    /// Optional fade-in time for this specific curve.
    #[serde(default)]
    pub fade_in_time: Option<f32>,
    /// Optional fade-out time for this specific curve.
    #[serde(default)]
    pub fade_out_time: Option<f32>,
    /// Flat segment encoding. Length depends on segment kinds.
    /// Renderer (M4.4) decodes via Cubism's segment-type markers.
    #[serde(default)]
    pub segments: Vec<f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Aria's actual model3.json shape — pinned here so the parser
    /// stays compatible with the real bundle even before the tier-2
    /// test runs.
    const ARIA_MANIFEST: &str = r#"{
        "Version": 3,
        "FileReferences": {
            "Moc": "aria.moc3",
            "Textures": ["aria.8192/texture_00.png"],
            "Physics": "aria.physics3.json",
            "DisplayInfo": "aria.cdi3.json",
            "Expressions": [
                { "Name": "neutral", "File": "expressions/neutral.exp3.json" },
                { "Name": "happy", "File": "expressions/happy.exp3.json" }
            ],
            "Motions": {
                "Idle": [
                    { "File": "animations/idle.motion3.json", "FadeInTime": 1.5, "FadeOutTime": 1.5 }
                ],
                "Happy": []
            }
        },
        "Groups": [
            { "Target": "Parameter", "Name": "LipSync", "Ids": [] },
            { "Target": "Parameter", "Name": "EyeBlink", "Ids": ["ParamEyeLOpen", "ParamEyeROpen"] }
        ],
        "HitAreas": []
    }"#;

    #[test]
    fn parses_aria_shaped_manifest() {
        let m: ModelJson = serde_json::from_str(ARIA_MANIFEST).unwrap();
        assert_eq!(m.version, 3);
        assert_eq!(m.file_references.moc, "aria.moc3");
        assert_eq!(m.file_references.textures.len(), 1);
        assert_eq!(m.file_references.expressions.len(), 2);
        assert_eq!(m.file_references.expressions[1].name, "happy");
        assert_eq!(m.file_references.motions.len(), 2);
        assert_eq!(m.file_references.motions.get("Idle").unwrap().len(), 1);
        assert_eq!(m.groups.len(), 2);
        assert_eq!(m.groups[1].ids.len(), 2);
    }

    #[test]
    fn missing_optional_fields_default_cleanly() {
        // Minimal manifest — just Version + Moc + Textures.
        let json = r#"{
            "Version": 3,
            "FileReferences": { "Moc": "x.moc3", "Textures": [] }
        }"#;
        let m: ModelJson = serde_json::from_str(json).unwrap();
        assert!(m.file_references.physics.is_none());
        assert!(m.file_references.expressions.is_empty());
        assert!(m.file_references.motions.is_empty());
        assert!(m.groups.is_empty());
        assert!(m.hit_areas.is_empty());
    }

    #[test]
    fn resolver_joins_paths_against_manifest_dir() {
        let m: ModelJson = serde_json::from_str(ARIA_MANIFEST).unwrap();
        let r = ResolvedModel {
            manifest: m,
            manifest_dir: PathBuf::from("/some/where"),
            manifest_path: PathBuf::from("/some/where/aria.model3.json"),
        };
        assert_eq!(r.moc_path(), PathBuf::from("/some/where/aria.moc3"));
        assert_eq!(
            r.texture_paths(),
            vec![PathBuf::from("/some/where/aria.8192/texture_00.png")]
        );
        assert_eq!(
            r.physics_path(),
            Some(PathBuf::from("/some/where/aria.physics3.json"))
        );
        assert_eq!(
            r.expression_path("happy"),
            Some(PathBuf::from("/some/where/expressions/happy.exp3.json"))
        );
        assert_eq!(r.expression_path("nonexistent"), None);
    }

    #[test]
    fn motions_iterator_yields_resolved_paths() {
        let m: ModelJson = serde_json::from_str(ARIA_MANIFEST).unwrap();
        let r = ResolvedModel {
            manifest: m,
            manifest_dir: PathBuf::from("/m"),
            manifest_path: PathBuf::from("/m/aria.model3.json"),
        };
        let idle: Vec<_> = r.motions("Idle").collect();
        assert_eq!(idle.len(), 1);
        assert_eq!(idle[0].0, PathBuf::from("/m/animations/idle.motion3.json"));
        assert_eq!(idle[0].1.fade_in_time, 1.5);

        // Happy is declared but empty.
        let happy: Vec<_> = r.motions("Happy").collect();
        assert!(happy.is_empty());

        // Missing group yields empty iterator (no panic).
        let missing: Vec<_> = r.motions("Nonexistent").collect();
        assert!(missing.is_empty());
    }

    #[test]
    fn group_ids_lookup() {
        let m: ModelJson = serde_json::from_str(ARIA_MANIFEST).unwrap();
        let r = ResolvedModel {
            manifest: m,
            manifest_dir: PathBuf::from("/m"),
            manifest_path: PathBuf::from("/m/x.model3.json"),
        };
        assert_eq!(r.group_ids("LipSync"), Some(&[][..]));
        assert_eq!(
            r.group_ids("EyeBlink"),
            Some(&["ParamEyeLOpen".to_string(), "ParamEyeROpen".to_string()][..])
        );
        assert_eq!(r.group_ids("Nonexistent"), None);
    }

    #[test]
    fn parses_expression_json() {
        // Lifted verbatim from Aria's happy.exp3.json shape.
        let json = r#"{
            "Type": "Live2D Expression",
            "FadeInTime": 0.2,
            "FadeOutTime": 0.2,
            "Parameters": [
                { "Id": "ParamExpHappy", "Value": 1.0, "Blend": "Add" },
                { "Id": "ParamBrowLY", "Value": 0.1 }
            ]
        }"#;
        let e: ExpressionJson = serde_json::from_str(json).unwrap();
        assert_eq!(e.kind, "Live2D Expression");
        assert_eq!(e.fade_in_time, 0.2);
        assert_eq!(e.parameters.len(), 2);
        assert_eq!(e.parameters[0].blend_kind(), ExpressionBlend::Add);
        // Default blend is Add when omitted.
        assert_eq!(e.parameters[1].blend_kind(), ExpressionBlend::Add);
    }

    #[test]
    fn expression_blend_decodes_known_kinds() {
        let p = ExpressionParameter {
            id: "x".into(),
            value: 1.0,
            blend: "Multiply".into(),
        };
        assert_eq!(p.blend_kind(), ExpressionBlend::Multiply);
        let p = ExpressionParameter {
            id: "x".into(),
            value: 1.0,
            blend: "Overwrite".into(),
        };
        assert_eq!(p.blend_kind(), ExpressionBlend::Overwrite);
        let p = ExpressionParameter {
            id: "x".into(),
            value: 1.0,
            blend: "Future".into(),
        };
        assert!(matches!(p.blend_kind(), ExpressionBlend::Unknown(_)));
    }

    #[test]
    fn parses_motion_json_skeleton() {
        // Tiny synthetic motion — pins the parser without needing
        // the 100KB Aria animations on disk.
        let json = r#"{
            "Version": 3,
            "Meta": {
                "Duration": 4.5,
                "Fps": 30.0,
                "Loop": true,
                "AreBeziersRestricted": true,
                "CurveCount": 1,
                "TotalSegmentCount": 4,
                "TotalPointCount": 12,
                "UserDataCount": 0,
                "TotalUserDataSize": 0
            },
            "Curves": [
                {
                    "Target": "Parameter",
                    "Id": "ParamAngleX",
                    "Segments": [0.0, 0.0, 1, 1.5, 5.0]
                }
            ]
        }"#;
        let m: MotionJson = serde_json::from_str(json).unwrap();
        assert_eq!(m.version, 3);
        assert_eq!(m.meta.duration, 4.5);
        assert!(m.meta.r#loop);
        assert_eq!(m.curves.len(), 1);
        assert_eq!(m.curves[0].id, "ParamAngleX");
        assert_eq!(m.curves[0].segments.len(), 5);
    }
}
