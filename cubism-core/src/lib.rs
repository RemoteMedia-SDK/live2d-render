//! Safe Rust wrapper around **Live2D Cubism Core**.
//!
//! Cubism Core is the proprietary C runtime that parses `.moc3`
//! rigged-mesh files and computes post-deformer vertex positions
//! from parameter values. This crate wraps the FFI surface from
//! `cubism-core-sys` with idiomatic Rust types — borrow-checked
//! lifetimes, slice-based mesh accessors, typed flag bitfields —
//! so callers don't reach into `unsafe` to render a Live2D model.
//!
//! See [`crates/cubism-core-sys/CUBISM_SDK.md`](../../cubism-core-sys/CUBISM_SDK.md)
//! for SDK acquisition + linkage. This crate has no SDK
//! dependency at runtime — the SDK is statically linked into the
//! `cubism-core-sys` artifact at build time.
//!
//! ## Lifetimes
//!
//! `Moc` owns its parsed buffer; `Model<'moc>` borrows the `Moc`
//! it was initialized from. Drawable / parameter views borrow the
//! `&Model` they came from, so calling `model.update()` (which
//! mutates vertex data in place) requires no outstanding views.
//!
//! ## Threading
//!
//! Cubism Core itself is **not thread-safe per model**. A `Model`
//! is `Send` (you can move it between threads) but not `Sync`
//! (you can't call `csmUpdateModel` concurrently from two
//! threads). A `Moc` is read-only after parse and is both `Send`
//! and `Sync`.
//!
//! ## Example
//!
//! ```ignore
//! use cubism_core::{Moc, Model};
//!
//! let moc = Moc::load_from_file("aria.moc3")?;
//! let mut model = Model::from_moc(&moc)?;
//! model.update();
//!
//! for d in model.drawables().iter() {
//!     println!("{}: {} vertices", d.id(), d.vertex_positions().len());
//! }
//! # Ok::<(), cubism_core::Error>(())
//! ```

#![deny(missing_debug_implementations)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]

mod buffer;
pub mod drawable;
pub mod model_json;
pub mod parameters;
pub mod physics;

use cubism_core_sys as sys;
use std::ffi::CStr;
use std::path::Path;

use buffer::AlignedBuffer;

pub use drawable::{BlendMode, ConstantFlags, DrawableView, Drawables, DynamicFlags};
pub use model_json::{
    ExpressionBlend, ExpressionJson, ExpressionParameter, ExpressionRef, FileReferences, Group,
    HitArea, ModelJson, ModelJsonError, MotionCurve, MotionJson, MotionMeta, MotionRef,
    ResolvedModel,
};
pub use parameters::{ParameterType, ParameterView, Parameters, PartView, Parts};
pub use physics::{
    EffectiveForces, NormalizationRange, PhysicsError, PhysicsInput, PhysicsMeta,
    PhysicsNormalization, PhysicsOutput, PhysicsRig, PhysicsSetting, PhysicsSimulator,
    PhysicsTarget, PhysicsVertex,
};

/// 2-component vector — matches Cubism's `csmVector2` C layout.
/// Returned by `DrawableView::vertex_positions` etc. as borrowed
/// slices over the model's memory.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec2 {
    /// X component.
    pub x: f32,
    /// Y component.
    pub y: f32,
}

/// 4-component vector — matches Cubism's `csmVector4` C layout.
/// Used for multiply / screen colors per drawable.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vec4 {
    /// X component (R if used as colour).
    pub x: f32,
    /// Y component (G).
    pub y: f32,
    /// Z component (B).
    pub z: f32,
    /// W component (A).
    pub w: f32,
}

const _ASSERT_VEC2_LAYOUT: () = {
    assert!(std::mem::size_of::<Vec2>() == std::mem::size_of::<sys::csmVector2>());
    assert!(std::mem::align_of::<Vec2>() == std::mem::align_of::<sys::csmVector2>());
};
const _ASSERT_VEC4_LAYOUT: () = {
    assert!(std::mem::size_of::<Vec4>() == std::mem::size_of::<sys::csmVector4>());
    assert!(std::mem::align_of::<Vec4>() == std::mem::align_of::<sys::csmVector4>());
};

/// Errors surfaced by the safe wrapper.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error reading a `.moc3` file from disk.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// `csmHasMocConsistency` returned 0 — the bytes aren't a valid
    /// `.moc3` (bad magic, truncated, version mismatch, …).
    #[error(
        "moc3 consistency check failed — file is not a valid .moc3 \
         (bad magic, truncated, or unsupported version). Expected \
         a Cubism .moc3 starting with `MOC3` magic bytes."
    )]
    InconsistentMoc,

    /// The .moc3's recorded version is newer than the linked SDK
    /// can decode. Update `LIVE2D_CUBISM_CORE_DIR` to a newer SDK.
    #[error(
        "moc3 version {moc_version:#x} is newer than this SDK's \
         latest supported version {latest:#x} — update \
         LIVE2D_CUBISM_CORE_DIR to a newer Cubism SDK release"
    )]
    UnsupportedMocVersion {
        /// The moc3's stamped version (csmMocVersion enum).
        moc_version: u32,
        /// The linked SDK's `csmGetLatestMocVersion()`.
        latest: u32,
    },

    /// `csmReviveMocInPlace` returned NULL — internal SDK error.
    /// Should be unreachable after a successful consistency check.
    #[error("csmReviveMocInPlace returned NULL — Cubism SDK internal error")]
    ReviveFailed,

    /// `csmInitializeModelInPlace` returned NULL — internal SDK
    /// error.
    #[error("csmInitializeModelInPlace returned NULL — Cubism SDK internal error")]
    ModelInitFailed,
}

/// Result alias for [`cubism_core::Error`](Error).
pub type Result<T> = std::result::Result<T, Error>;

// ─── Moc ─────────────────────────────────────────────────────────────────────

/// A parsed `.moc3` rigged-mesh definition.
///
/// Owns its aligned buffer; the parsed Cubism `csmMoc` lives inside
/// it. After construction the `Moc` is read-only — many `Model`s can
/// be derived from a single `Moc` simultaneously.
pub struct Moc {
    buffer: AlignedBuffer,
    raw: *const sys::csmMoc,
}

// `csmMoc` is read-only after `csmReviveMocInPlace`. Cubism Core's
// docs explicitly note that creating multiple models from one moc is
// safe — the moc data isn't mutated. So `Moc` is `Send` + `Sync`.
unsafe impl Send for Moc {}
unsafe impl Sync for Moc {}

impl std::fmt::Debug for Moc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Moc")
            .field("version", &self.version())
            .field("buffer_len", &self.buffer.len())
            .finish()
    }
}

impl Moc {
    /// Read + parse a `.moc3` from disk. Slurps the entire file into
    /// memory, copies it into a 64-byte-aligned buffer (the
    /// alignment Cubism Core requires), validates consistency, and
    /// revives it.
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::load_from_bytes(&bytes)
    }

    /// Parse a `.moc3` from already-loaded bytes. The bytes are
    /// copied into a freshly-aligned buffer; the input slice can be
    /// dropped after this returns.
    pub fn load_from_bytes(bytes: &[u8]) -> Result<Self> {
        let align = sys::csmAlignofMoc as usize;
        let mut buffer = AlignedBuffer::new(bytes.len(), align);
        // SAFETY: `buffer.as_mut_slice()` returns a properly-aligned,
        // properly-sized mutable slice into the heap-owned buffer;
        // copying into it is straightforward.
        buffer.as_mut_slice().copy_from_slice(bytes);

        // Consistency check first. Cubism docs note the consistency
        // check is non-destructive and ensures the structural
        // invariants the revive call assumes — without it, revive
        // can produce a valid-looking but corrupted moc that later
        // crashes inside csmUpdateModel.
        let size_u32 = u32::try_from(bytes.len()).map_err(|_| Error::InconsistentMoc)?;
        // SAFETY: buffer is aligned + valid for `bytes.len()` bytes;
        // SDK reads them through the consistency probe.
        let consistent =
            unsafe { sys::csmHasMocConsistency(buffer.as_mut_ptr() as *mut _, size_u32) };
        if consistent == 0 {
            return Err(Error::InconsistentMoc);
        }

        // SAFETY: same alignment + size invariants. After revive,
        // the returned `*mut csmMoc` points into our buffer; we
        // store it as `*const csmMoc` since post-revive the moc is
        // read-only.
        let raw = unsafe { sys::csmReviveMocInPlace(buffer.as_mut_ptr() as *mut _, size_u32) };
        if raw.is_null() {
            return Err(Error::ReviveFailed);
        }

        let moc = Self {
            buffer,
            raw: raw as *const sys::csmMoc,
        };

        // Reject mocs newer than the linked SDK can decode. The
        // SDK's revive call accepts up-to-version-N mocs silently
        // even when the binary supports higher versions; rejecting
        // here gives a much better error message than a downstream
        // garbage drawable.
        let moc_version = moc.version();
        // SAFETY: csmGetLatestMocVersion takes no args, returns u32.
        let latest = unsafe { sys::csmGetLatestMocVersion() };
        if moc_version > latest {
            return Err(Error::UnsupportedMocVersion {
                moc_version,
                latest,
            });
        }

        Ok(moc)
    }

    /// Return the moc3 file-format version (a `csmMocVersion_*`
    /// enumeration value). Read this if you want to gate behaviour
    /// on a specific moc revision.
    pub fn version(&self) -> u32 {
        let size_u32 = u32::try_from(self.buffer.len()).unwrap_or(0);
        // SAFETY: `self.raw` is a valid moc pointer for the lifetime
        // of `self`; `self.buffer.len()` bytes are valid behind it.
        // The SDK types this entry point against `*const c_void`
        // (rather than `*const csmMoc`) — historical wart in the
        // header. We pass the same buffer pointer either way.
        unsafe { sys::csmGetMocVersion(self.raw as *const _, size_u32) }
    }

    /// Borrow the underlying `*const csmMoc` for `Model` initialization.
    pub(crate) fn raw(&self) -> *const sys::csmMoc {
        self.raw
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// A parameterized instance of a `Moc`, ready to be ticked and read.
///
/// Holds an aligned heap buffer that Cubism Core writes vertex
/// positions, parameter values, and drawable state into. The
/// lifetime parameter ties the `Model` to the originating `Moc` —
/// dropping the `Moc` while a `Model` is alive is a compile error.
pub struct Model<'moc> {
    /// Heap allocation Cubism Core writes model state into. We
    /// only access it through `raw` (which points inside this
    /// buffer); rustc can't see that, hence the allow.
    #[allow(dead_code)]
    buffer: AlignedBuffer,
    raw: *mut sys::csmModel,
    _moc: std::marker::PhantomData<&'moc Moc>,
}

// Models are not safe to share concurrently (csmUpdateModel mutates
// vertex data in place), but they can be moved between threads.
unsafe impl<'moc> Send for Model<'moc> {}

impl<'moc> std::fmt::Debug for Model<'moc> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let canvas = self.canvas_info();
        f.debug_struct("Model")
            .field("canvas_size_pixels", &(canvas.size.x, canvas.size.y))
            .field("pixels_per_unit", &canvas.pixels_per_unit)
            .field("drawable_count", &self.drawables().len())
            .field("parameter_count", &self.parameters().len())
            .finish()
    }
}

/// Canvas metadata — the model-space bounds Cubism Core reports
/// (NOT the rendered pixel resolution; M4.4 picks that separately).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CanvasInfo {
    /// Canvas size in model-space pixels.
    pub size: Vec2,
    /// Origin offset in model-space pixels.
    pub origin: Vec2,
    /// Pixels per model-unit; multiply with model-space coordinates
    /// to get pixel coordinates inside the canvas rect.
    pub pixels_per_unit: f32,
}

impl<'moc> Model<'moc> {
    /// Initialize a fresh Model from a parsed Moc. Allocates a
    /// per-instance buffer (16-byte-aligned, sized via
    /// `csmGetSizeofModel`); two `Model`s from the same `Moc` have
    /// independent state.
    pub fn from_moc(moc: &'moc Moc) -> Result<Self> {
        // SAFETY: moc.raw() is a valid `*const csmMoc` for the
        // lifetime tied to `&moc`.
        let model_size = unsafe { sys::csmGetSizeofModel(moc.raw()) };
        let align = sys::csmAlignofModel as usize;
        let mut buffer = AlignedBuffer::new(model_size as usize, align);
        // SAFETY: the SDK initializes model state into a properly-
        // aligned buffer of the size it just told us to allocate;
        // the moc pointer is valid; we drop ownership of the buffer
        // bytes to the SDK (it'll mutate them via csmUpdateModel
        // later).
        let raw = unsafe {
            sys::csmInitializeModelInPlace(moc.raw(), buffer.as_mut_ptr() as *mut _, model_size)
        };
        if raw.is_null() {
            return Err(Error::ModelInitFailed);
        }
        Ok(Self {
            buffer,
            raw,
            _moc: std::marker::PhantomData,
        })
    }

    /// Recompute drawable state (vertex positions, render order,
    /// opacities, dynamic flags) from the model's current parameter
    /// + part values. Call this after every parameter change before
    /// reading drawable data.
    pub fn update(&mut self) {
        // SAFETY: self.raw is a valid model pointer for `self`'s
        // lifetime; the SDK mutates state inside `self.buffer`.
        unsafe { sys::csmUpdateModel(self.raw) };
    }

    /// Read the canvas metadata (size, origin, pixels-per-unit).
    /// Stable for the model's lifetime; call once at init.
    pub fn canvas_info(&self) -> CanvasInfo {
        let mut size = sys::csmVector2 { X: 0.0, Y: 0.0 };
        let mut origin = sys::csmVector2 { X: 0.0, Y: 0.0 };
        let mut ppu: f32 = 0.0;
        // SAFETY: self.raw valid; out-pointers are local stack vars.
        unsafe {
            sys::csmReadCanvasInfo(self.raw, &mut size, &mut origin, &mut ppu);
        }
        CanvasInfo {
            size: Vec2 {
                x: size.X,
                y: size.Y,
            },
            origin: Vec2 {
                x: origin.X,
                y: origin.Y,
            },
            pixels_per_unit: ppu,
        }
    }

    /// Borrow the model's drawables (indexed by render order in the
    /// rigged mesh). The returned view borrows `&self`; drop it
    /// before calling [`Self::update`].
    pub fn drawables(&self) -> Drawables<'_> {
        // SAFETY: self.raw valid for `&self`'s lifetime; the SDK
        // reads count from self.buffer.
        let count = unsafe { sys::csmGetDrawableCount(self.raw) } as usize;
        Drawables::new(self.raw, count)
    }

    /// Borrow the model's parameters (read-only). Drops out before
    /// `update()` (same borrow rules as drawables).
    pub fn parameters(&self) -> Parameters<'_> {
        // SAFETY: self.raw valid; bindgen typed `csmGetParameterCount`
        // as `*const csmModel`, which we satisfy via cast.
        let count = unsafe { sys::csmGetParameterCount(self.raw) } as usize;
        Parameters::new(self.raw, count)
    }

    /// Mutably borrow the model's parameters so values can be set.
    /// Take this borrow, write values, drop it, then call `update()`.
    pub fn parameters_mut(&mut self) -> Parameters<'_> {
        // Same constructor — Parameters internally calls the
        // mutable-pointer SDK accessor for value writes; the
        // `&mut self` requirement here is the Rust-side safety net.
        let count = unsafe { sys::csmGetParameterCount(self.raw) } as usize;
        Parameters::new(self.raw, count)
    }

    /// Borrow the model's parts (named groups; control opacity).
    pub fn parts(&self) -> Parts<'_> {
        let count = unsafe { sys::csmGetPartCount(self.raw) } as usize;
        Parts::new(self.raw, count)
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Convert a NUL-terminated C string from the model's heap into a
/// `&'a str`. Cubism's drawable / parameter / part IDs are stored
/// in the model buffer and outlive any `&Model` borrow; the input
/// pointer must come from one of the SDK's `csmGet*Ids` accessors.
///
/// Returns the empty string on a NUL pointer or a non-UTF-8 ID.
/// Cubism's IDs are ASCII in practice (parameter names like
/// `ParamJawOpen`, drawable names like `Drawable_Hair`), so the
/// fallback path is unreachable for well-formed models.
unsafe fn cstr_to_str<'a>(ptr: *const std::os::raw::c_char) -> &'a str {
    if ptr.is_null() {
        return "";
    }
    // SAFETY: caller guarantees `ptr` came from an SDK accessor and
    // is NUL-terminated; the resulting &CStr / &str lifetime is
    // tied to the `&Model` borrow the caller holds.
    let cstr = unsafe { CStr::from_ptr(ptr) };
    cstr.to_str().unwrap_or("")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `csmHasMocConsistency` rejects all-zero bytes, so
    /// `Moc::load_from_bytes` returns `InconsistentMoc`. The bigger
    /// invariant being pinned here: the SDK's consistency check
    /// catches non-moc data without segfaulting on the (aligned but
    /// nonsense) buffer we hand it.
    #[test]
    fn load_from_bytes_rejects_zeros() {
        // 1 KiB of zeros is more than enough for the consistency
        // probe to reach the magic-bytes check.
        let zeros = vec![0u8; 1024];
        let err = Moc::load_from_bytes(&zeros).expect_err("must reject zeros");
        assert!(matches!(err, Error::InconsistentMoc));
    }

    #[test]
    fn load_from_bytes_rejects_truncated_magic() {
        // `MOC` (3 bytes, missing trailing `3`) — short enough that
        // the consistency check fails on size before signature.
        let bytes = b"MOC".to_vec();
        let err = Moc::load_from_bytes(&bytes).expect_err("must reject truncated magic");
        assert!(matches!(err, Error::InconsistentMoc));
    }

    /// `Vec2`/`Vec4` are `#[repr(C)]` and must match the SDK's
    /// `csmVector2`/`csmVector4` byte-for-byte so we can hand out
    /// borrowed slices without copying. `_ASSERT_*_LAYOUT` in lib.rs
    /// enforces this at compile time; this test pins the runtime
    /// invariant (size + alignment) explicitly.
    #[test]
    fn vec_layout_matches_csm_vector() {
        assert_eq!(std::mem::size_of::<Vec2>(), 8);
        assert_eq!(std::mem::size_of::<Vec4>(), 16);
        assert_eq!(std::mem::align_of::<Vec2>(), 4);
        assert_eq!(std::mem::align_of::<Vec4>(), 4);
    }
}
