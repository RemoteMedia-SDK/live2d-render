//! Post-deformer drawable mesh accessors.
//!
//! After `Model::update()`, each drawable's vertex positions, render
//! order, opacity, and dynamic flags are written into the model's
//! buffer. The accessors here borrow that data as Rust slices keyed
//! by the `&Model` lifetime — calling `Model::update()` requires
//! every outstanding `Drawables` / `DrawableView` to be dropped first
//! (`update` takes `&mut self`).
//!
//! # Why every method goes through a `*const csmModel`
//!
//! The SDK exposes drawable data as **arrays-keyed-by-drawable-index**
//! (one big array per attribute, indexed by drawable). To return
//! data for drawable `i` we re-fetch the array pointer for every
//! call. That's free in practice — the SDK pointer is a constant
//! offset into the model buffer — but it keeps the wrapper stateless.

use crate::{cstr_to_str, Vec2, Vec4};
use bitflags::bitflags;
use cubism_core_sys as sys;
use std::marker::PhantomData;

/// Borrowed view of all drawables in a model. Iterate via
/// [`Self::iter`] or index via [`Self::get`].
pub struct Drawables<'a> {
    model: *const sys::csmModel,
    count: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> std::fmt::Debug for Drawables<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Drawables")
            .field("count", &self.count)
            .finish()
    }
}

impl<'a> Drawables<'a> {
    pub(crate) fn new(model: *const sys::csmModel, count: usize) -> Self {
        Self {
            model,
            count,
            _marker: PhantomData,
        }
    }

    /// Number of drawables in the model.
    pub fn len(&self) -> usize {
        self.count
    }

    /// `true` iff the model has no drawables. (Always false in
    /// practice — a moc with zero drawables wouldn't pass
    /// `csmHasMocConsistency`.)
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Borrow the drawable at `index`, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<DrawableView<'a>> {
        if index >= self.count {
            return None;
        }
        Some(DrawableView {
            model: self.model,
            index,
            _marker: PhantomData,
        })
    }

    /// Iterate every drawable in index order.
    pub fn iter(&self) -> DrawablesIter<'a> {
        DrawablesIter {
            model: self.model,
            count: self.count,
            next: 0,
            _marker: PhantomData,
        }
    }
}

/// Iterator over [`DrawableView`]s.
pub struct DrawablesIter<'a> {
    model: *const sys::csmModel,
    count: usize,
    next: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> Iterator for DrawablesIter<'a> {
    type Item = DrawableView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(DrawableView {
            model: self.model,
            index,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count - self.next;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for DrawablesIter<'a> {}

impl<'a> std::fmt::Debug for DrawablesIter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawablesIter")
            .field("count", &self.count)
            .field("next", &self.next)
            .finish()
    }
}

/// Borrowed view of one drawable's post-deformer state.
///
/// Every accessor returns either a `Copy` value or a slice borrowing
/// the model's buffer for `'a`. Slices are stable for the lifetime
/// of the borrow — `Model::update` invalidates them by requiring
/// `&mut self`.
pub struct DrawableView<'a> {
    model: *const sys::csmModel,
    index: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> std::fmt::Debug for DrawableView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DrawableView")
            .field("index", &self.index)
            .field("id", &self.id())
            .field("vertex_count", &self.vertex_positions().len())
            .field("opacity", &self.opacity())
            .field("render_order", &self.render_order())
            .field("blend_mode", &self.blend_mode())
            .finish()
    }
}

impl<'a> DrawableView<'a> {
    /// Drawable's name as authored in the moc (e.g. `Drawable_Hair`).
    pub fn id(&self) -> &'a str {
        // SAFETY: `csmGetDrawableIds` returns a pointer to a stable
        // array of NUL-terminated strings owned by the model
        // buffer; valid for `'a`.
        unsafe {
            let ids = sys::csmGetDrawableIds(self.model);
            if ids.is_null() {
                return "";
            }
            cstr_to_str(*ids.add(self.index))
        }
    }

    /// Index into the drawable array. Stable across calls.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Post-deformer vertex positions in model space. Length matches
    /// [`Self::vertex_uvs`] and corresponds to the indices returned
    /// by [`Self::indices`].
    pub fn vertex_positions(&self) -> &'a [Vec2] {
        // SAFETY: csmGetDrawableVertexPositions returns a pointer
        // to a stable array of `csmVector2*` (one per drawable).
        // The drawable's array is `count` elements long, where
        // `count` comes from csmGetDrawableVertexCounts.
        unsafe {
            let positions = sys::csmGetDrawableVertexPositions(self.model);
            let counts = sys::csmGetDrawableVertexCounts(self.model);
            if positions.is_null() || counts.is_null() {
                return &[];
            }
            let count = *counts.add(self.index) as usize;
            let pos_ptr = *positions.add(self.index) as *const Vec2;
            std::slice::from_raw_parts(pos_ptr, count)
        }
    }

    /// Texture UV coordinates per vertex. Static — doesn't change
    /// across `update()` calls.
    pub fn vertex_uvs(&self) -> &'a [Vec2] {
        // SAFETY: same shape as vertex_positions.
        unsafe {
            let uvs = sys::csmGetDrawableVertexUvs(self.model);
            let counts = sys::csmGetDrawableVertexCounts(self.model);
            if uvs.is_null() || counts.is_null() {
                return &[];
            }
            let count = *counts.add(self.index) as usize;
            let uv_ptr = *uvs.add(self.index) as *const Vec2;
            std::slice::from_raw_parts(uv_ptr, count)
        }
    }

    /// Triangle indices into the vertex/UV arrays. Always a flat
    /// `&[u16]`; renderer feeds it directly to the GPU.
    pub fn indices(&self) -> &'a [u16] {
        // SAFETY: csmGetDrawableIndices returns a pointer-to-pointer
        // shape (one ushort* per drawable), keyed by drawable index;
        // count comes from csmGetDrawableIndexCounts.
        unsafe {
            let indices = sys::csmGetDrawableIndices(self.model);
            let counts = sys::csmGetDrawableIndexCounts(self.model);
            if indices.is_null() || counts.is_null() {
                return &[];
            }
            let count = *counts.add(self.index) as usize;
            let ix_ptr = *indices.add(self.index);
            std::slice::from_raw_parts(ix_ptr, count)
        }
    }

    /// Per-drawable opacity in `[0, 1]`. Multiplied with the
    /// drawable's texture alpha at render time.
    pub fn opacity(&self) -> f32 {
        // SAFETY: csmGetDrawableOpacities returns a stable f32 array
        // indexed by drawable; valid for `'a`.
        unsafe {
            let arr = sys::csmGetDrawableOpacities(self.model);
            if arr.is_null() {
                0.0
            } else {
                *arr.add(self.index)
            }
        }
    }

    /// Render order — drawables are rendered in ascending order.
    /// Distinct from [`Self::draw_order`] (which encodes
    /// authored layering before render-order overrides).
    pub fn render_order(&self) -> i32 {
        // Note the SDK names this `csmGetRenderOrders` (no
        // `Drawable` prefix) — it's the only such accessor; every
        // other drawable attribute uses `csmGetDrawable*`. The
        // returned array is still drawable-indexed.
        unsafe {
            let arr = sys::csmGetRenderOrders(self.model);
            if arr.is_null() {
                0
            } else {
                *arr.add(self.index) as i32
            }
        }
    }

    /// Authored draw order. Most callers want
    /// [`Self::render_order`] — render order is what the SDK
    /// actually expects to be used for sorting.
    pub fn draw_order(&self) -> i32 {
        unsafe {
            let arr = sys::csmGetDrawableDrawOrders(self.model);
            if arr.is_null() {
                0
            } else {
                *arr.add(self.index) as i32
            }
        }
    }

    /// Per-frame dirty/visibility flags. Use to gate re-uploads of
    /// vertex buffers (only re-upload when
    /// [`DynamicFlags::VERTEX_POSITIONS_DID_CHANGE`] is set, etc.).
    pub fn dynamic_flags(&self) -> DynamicFlags {
        unsafe {
            let arr = sys::csmGetDrawableDynamicFlags(self.model);
            if arr.is_null() {
                DynamicFlags::empty()
            } else {
                DynamicFlags::from_bits_truncate(*arr.add(self.index))
            }
        }
    }

    /// Static per-drawable rigging flags (blend mode, double-sided,
    /// inverted-mask). Doesn't change across `update()`.
    pub fn constant_flags(&self) -> ConstantFlags {
        unsafe {
            let arr = sys::csmGetDrawableConstantFlags(self.model);
            if arr.is_null() {
                ConstantFlags::empty()
            } else {
                ConstantFlags::from_bits_truncate(*arr.add(self.index))
            }
        }
    }

    /// Resolved blend mode (Normal / Additive / Multiplicative).
    /// Convenience over [`Self::constant_flags`] — Cubism encodes
    /// the mode as two flag bits.
    pub fn blend_mode(&self) -> BlendMode {
        BlendMode::from_constant_flags(self.constant_flags())
    }

    /// Texture array index — pairs with the texture list in the
    /// `.model3.json` `FileReferences.Textures` field.
    pub fn texture_index(&self) -> i32 {
        unsafe {
            let arr = sys::csmGetDrawableTextureIndices(self.model);
            if arr.is_null() {
                0
            } else {
                *arr.add(self.index) as i32
            }
        }
    }

    /// Indices of mask drawables this drawable is clipped against.
    /// Empty if unmasked. Renderer pre-pass renders masks into a
    /// stencil/alpha buffer first; see Cubism's reference renderer
    /// for the standard mask-prepass shape.
    pub fn masks(&self) -> &'a [i32] {
        unsafe {
            let masks = sys::csmGetDrawableMasks(self.model);
            let counts = sys::csmGetDrawableMaskCounts(self.model);
            if masks.is_null() || counts.is_null() {
                return &[];
            }
            let count = *counts.add(self.index) as usize;
            let mask_ptr = *masks.add(self.index);
            std::slice::from_raw_parts(mask_ptr, count)
        }
    }

    /// Per-drawable multiply colour (rigged colour modulation).
    /// `[1.0, 1.0, 1.0, 1.0]` when no modulation is applied.
    pub fn multiply_color(&self) -> Vec4 {
        unsafe {
            let arr = sys::csmGetDrawableMultiplyColors(self.model);
            if arr.is_null() {
                Vec4 {
                    x: 1.0,
                    y: 1.0,
                    z: 1.0,
                    w: 1.0,
                }
            } else {
                let v = *arr.add(self.index);
                Vec4 {
                    x: v.X,
                    y: v.Y,
                    z: v.Z,
                    w: v.W,
                }
            }
        }
    }

    /// Per-drawable screen colour (rigged additive tint).
    /// `[0.0, 0.0, 0.0, 1.0]` when no tint is applied.
    pub fn screen_color(&self) -> Vec4 {
        unsafe {
            let arr = sys::csmGetDrawableScreenColors(self.model);
            if arr.is_null() {
                Vec4 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                    w: 1.0,
                }
            } else {
                let v = *arr.add(self.index);
                Vec4 {
                    x: v.X,
                    y: v.Y,
                    z: v.Z,
                    w: v.W,
                }
            }
        }
    }

    /// Index of this drawable's parent part (for opacity inheritance).
    /// `None` if the drawable has no parent part.
    pub fn parent_part_index(&self) -> Option<usize> {
        unsafe {
            let arr = sys::csmGetDrawableParentPartIndices(self.model);
            if arr.is_null() {
                return None;
            }
            let raw = *arr.add(self.index);
            // SDK uses -1 to denote "no parent".
            if raw < 0 {
                None
            } else {
                Some(raw as usize)
            }
        }
    }
}

bitflags! {
    /// Static drawable rigging flags (blend mode bits + double-sided
    /// + inverted-mask). Mirror of Cubism's `csmConstantFlags`.
    /// Read once at model init; doesn't change across `update()`.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct ConstantFlags: sys::csmFlags {
        /// Blend mode is `Additive` (else Normal unless multiplicative).
        const BLEND_ADDITIVE = sys::csmBlendAdditive as sys::csmFlags;
        /// Blend mode is `Multiplicative`.
        const BLEND_MULTIPLICATIVE = sys::csmBlendMultiplicative as sys::csmFlags;
        /// Drawable renders both faces (no back-face culling).
        const IS_DOUBLE_SIDED = sys::csmIsDoubleSided as sys::csmFlags;
        /// Mask hierarchy is inverted for this drawable.
        const IS_INVERTED_MASK = sys::csmIsInvertedMask as sys::csmFlags;
    }
}

bitflags! {
    /// Per-frame dirty/visibility flags. Mirror of Cubism's
    /// `csmDynamicFlags`. Set/cleared during each `update()`.
    #[repr(transparent)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct DynamicFlags: sys::csmFlags {
        /// Drawable is currently visible (alpha > 0 + visible part).
        const IS_VISIBLE = sys::csmIsVisible as sys::csmFlags;
        /// Visibility flipped this update.
        const VISIBILITY_DID_CHANGE = sys::csmVisibilityDidChange as sys::csmFlags;
        /// Opacity changed this update.
        const OPACITY_DID_CHANGE = sys::csmOpacityDidChange as sys::csmFlags;
        /// Draw order changed this update.
        const DRAW_ORDER_DID_CHANGE = sys::csmDrawOrderDidChange as sys::csmFlags;
        /// Render order changed this update.
        const RENDER_ORDER_DID_CHANGE = sys::csmRenderOrderDidChange as sys::csmFlags;
        /// Vertex positions changed (re-upload VB).
        const VERTEX_POSITIONS_DID_CHANGE = sys::csmVertexPositionsDidChange as sys::csmFlags;
        /// Multiply / screen colour changed this update.
        const BLEND_COLOR_DID_CHANGE = sys::csmBlendColorDidChange as sys::csmFlags;
    }
}

/// Resolved blend mode for a drawable (decoded from
/// [`ConstantFlags`] bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BlendMode {
    /// Standard alpha blending — `dst = src * src.a + dst * (1 - src.a)`.
    #[default]
    Normal,
    /// Additive blending — `dst = src + dst`.
    Additive,
    /// Multiplicative blending — `dst = src * dst`.
    Multiplicative,
}

impl BlendMode {
    /// Decode the blend mode from a drawable's constant-flag bitfield.
    /// Cubism encodes the three modes as two non-overlapping bits.
    pub fn from_constant_flags(flags: ConstantFlags) -> Self {
        if flags.contains(ConstantFlags::BLEND_ADDITIVE) {
            BlendMode::Additive
        } else if flags.contains(ConstantFlags::BLEND_MULTIPLICATIVE) {
            BlendMode::Multiplicative
        } else {
            BlendMode::Normal
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_mode_decodes_priority_order() {
        // Per Cubism's encoding, the two blend bits are mutually
        // exclusive in well-formed mocs. The decoder favours
        // additive over multiplicative if both are somehow set
        // (defensive — never seen in real models).
        assert_eq!(
            BlendMode::from_constant_flags(ConstantFlags::empty()),
            BlendMode::Normal
        );
        assert_eq!(
            BlendMode::from_constant_flags(ConstantFlags::BLEND_ADDITIVE),
            BlendMode::Additive
        );
        assert_eq!(
            BlendMode::from_constant_flags(ConstantFlags::BLEND_MULTIPLICATIVE),
            BlendMode::Multiplicative
        );
    }

    #[test]
    fn dynamic_flags_bit_values_match_sdk() {
        // Pin the bit-for-bit correspondence with the SDK constants.
        // If bindgen ever returns the constants under different
        // names, this test fails at compile time.
        assert_eq!(DynamicFlags::IS_VISIBLE.bits() as u32, sys::csmIsVisible);
        assert_eq!(
            DynamicFlags::VERTEX_POSITIONS_DID_CHANGE.bits() as u32,
            sys::csmVertexPositionsDidChange
        );
    }

    #[test]
    fn constant_flags_bit_values_match_sdk() {
        assert_eq!(
            ConstantFlags::BLEND_ADDITIVE.bits() as u32,
            sys::csmBlendAdditive
        );
        assert_eq!(
            ConstantFlags::IS_INVERTED_MASK.bits() as u32,
            sys::csmIsInvertedMask
        );
    }
}
