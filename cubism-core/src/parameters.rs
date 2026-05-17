//! Parameter + Part accessors.
//!
//! **Parameters** are the input dials a Live2D model exposes: each
//! has an `id`, `min`/`max`/`default`, current `value`, and a type
//! (Normal / BlendShape). The renderer drives the model by writing
//! into these values then calling `Model::update()`.
//!
//! **Parts** are named groups of drawables; each part has an
//! authored opacity that propagates to its children. Parts are
//! mostly informational here — opacity is set in the moc, not
//! typically driven at runtime — but expression files (`.exp3.json`,
//! M4.2) override part opacities to flip non-mouth facial features.
//!
//! See the spec at [`docs/superpowers/specs/2026-04-27-…`] §3.4 + the
//! Cubism docs for the parameter naming conventions Aria + most
//! commercial models follow (VBridger lip-sync params:
//! `ParamMouthOpenY`, `ParamMouthForm`, `ParamJawOpen`, etc.).

use crate::cstr_to_str;
use cubism_core_sys as sys;
use std::marker::PhantomData;

/// Parameter type — distinguishes "normal" continuous parameters
/// from blendshape-style parameters whose values index into a
/// keyform table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParameterType {
    /// Standard continuous parameter (most VBridger lip-sync params,
    /// expression sliders, etc.).
    Normal,
    /// Blendshape parameter — the parameter's value is a discrete
    /// index into a keyform table.
    BlendShape,
    /// Unknown type. Defensive — Cubism may add new parameter
    /// types in future SDK revs; we surface them rather than panic.
    Unknown(i32),
}

impl ParameterType {
    fn from_raw(raw: sys::csmParameterType) -> Self {
        // The SDK header defines `csmParameterTypeNormal = 0`,
        // `csmParameterTypeBlendShape = 1`. Those constants aren't
        // bound by name (they're loose `_bindgen_ty_*` consts); we
        // match on the raw int.
        match raw {
            0 => ParameterType::Normal,
            1 => ParameterType::BlendShape,
            other => ParameterType::Unknown(other),
        }
    }
}

// ─── Parameters ──────────────────────────────────────────────────────────────

/// Borrowed view of all parameters in a model. Parameters are
/// ordered as authored in the moc; the order is stable.
pub struct Parameters<'a> {
    model: *const sys::csmModel,
    count: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> std::fmt::Debug for Parameters<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parameters")
            .field("count", &self.count)
            .finish()
    }
}

impl<'a> Parameters<'a> {
    pub(crate) fn new(model: *const sys::csmModel, count: usize) -> Self {
        Self {
            model,
            count,
            _marker: PhantomData,
        }
    }

    /// Number of parameters in the model.
    pub fn len(&self) -> usize {
        self.count
    }

    /// `true` iff the model has no parameters.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Borrow the parameter at `index`, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<ParameterView<'a>> {
        if index >= self.count {
            return None;
        }
        Some(ParameterView {
            model: self.model,
            index,
            _marker: PhantomData,
        })
    }

    /// Find a parameter by id (linear scan). For hot paths that read
    /// many parameters per frame, cache the indices instead.
    pub fn find(&self, id: &str) -> Option<ParameterView<'a>> {
        for i in 0..self.count {
            let view = ParameterView {
                model: self.model,
                index: i,
                _marker: PhantomData,
            };
            if view.id() == id {
                return Some(view);
            }
        }
        None
    }

    /// Iterate every parameter in index order.
    pub fn iter(&self) -> ParametersIter<'a> {
        ParametersIter {
            model: self.model,
            count: self.count,
            next: 0,
            _marker: PhantomData,
        }
    }

    /// Set the parameter value at `index`. Panics on out-of-bounds.
    /// Use [`Self::get`] / [`Self::find`] + [`ParameterView::set_value`]
    /// for a panic-free API.
    pub fn set_value(&mut self, index: usize, value: f32) {
        assert!(
            index < self.count,
            "parameter index {} out of bounds",
            index
        );
        // SAFETY: `index < count`, so `add(index)` is in-bounds for
        // the parameter values array. `csmGetParameterValues`
        // returns a `*mut f32` that lives in the model's heap.
        unsafe {
            let arr = sys::csmGetParameterValues(self.model as *mut _);
            *arr.add(index) = value;
        }
    }
}

/// Iterator over [`ParameterView`]s.
pub struct ParametersIter<'a> {
    model: *const sys::csmModel,
    count: usize,
    next: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> Iterator for ParametersIter<'a> {
    type Item = ParameterView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(ParameterView {
            model: self.model,
            index,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.count - self.next;
        (r, Some(r))
    }
}

impl<'a> ExactSizeIterator for ParametersIter<'a> {}

impl<'a> std::fmt::Debug for ParametersIter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParametersIter")
            .field("count", &self.count)
            .field("next", &self.next)
            .finish()
    }
}

/// One parameter's metadata + current value.
pub struct ParameterView<'a> {
    model: *const sys::csmModel,
    index: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> std::fmt::Debug for ParameterView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterView")
            .field("index", &self.index)
            .field("id", &self.id())
            .field("value", &self.value())
            .field("range", &(self.min(), self.max()))
            .field("default", &self.default())
            .field("ty", &self.ty())
            .finish()
    }
}

impl<'a> ParameterView<'a> {
    /// Parameter ID (e.g. `ParamJawOpen`, `ParamMouthForm`).
    pub fn id(&self) -> &'a str {
        unsafe {
            let ids = sys::csmGetParameterIds(self.model);
            if ids.is_null() {
                return "";
            }
            cstr_to_str(*ids.add(self.index))
        }
    }

    /// Index in the parameter array.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Parameter type — Normal or BlendShape.
    pub fn ty(&self) -> ParameterType {
        unsafe {
            let arr = sys::csmGetParameterTypes(self.model);
            if arr.is_null() {
                ParameterType::Unknown(0)
            } else {
                ParameterType::from_raw(*arr.add(self.index))
            }
        }
    }

    /// Minimum allowed value (per the moc rigging).
    pub fn min(&self) -> f32 {
        unsafe {
            let arr = sys::csmGetParameterMinimumValues(self.model);
            if arr.is_null() {
                0.0
            } else {
                *arr.add(self.index)
            }
        }
    }

    /// Maximum allowed value (per the moc rigging).
    pub fn max(&self) -> f32 {
        unsafe {
            let arr = sys::csmGetParameterMaximumValues(self.model);
            if arr.is_null() {
                0.0
            } else {
                *arr.add(self.index)
            }
        }
    }

    /// Authored default value (used as the rest pose / neutral).
    pub fn default(&self) -> f32 {
        unsafe {
            let arr = sys::csmGetParameterDefaultValues(self.model);
            if arr.is_null() {
                0.0
            } else {
                *arr.add(self.index)
            }
        }
    }

    /// Current value. Read after `Model::update()` to see the
    /// effective value the model deformed against.
    pub fn value(&self) -> f32 {
        unsafe {
            // `csmGetParameterValues` is typed `*mut csmModel ->
            // *mut f32`; we only deref-read here. The cast to
            // `*mut` is safe because we don't write through it.
            let arr = sys::csmGetParameterValues(self.model as *mut _);
            if arr.is_null() {
                0.0
            } else {
                *arr.add(self.index)
            }
        }
    }

    /// Set the current value. The next `Model::update()` will
    /// recompute drawable state against this value (clamped to
    /// [`Self::min`]/[`Self::max`] internally by the SDK).
    pub fn set_value(&self, value: f32) {
        unsafe {
            let arr = sys::csmGetParameterValues(self.model as *mut _);
            if !arr.is_null() {
                *arr.add(self.index) = value;
            }
        }
    }
}

// ─── Parts ───────────────────────────────────────────────────────────────────

/// Borrowed view of all parts in the model.
pub struct Parts<'a> {
    model: *const sys::csmModel,
    count: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> std::fmt::Debug for Parts<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Parts").field("count", &self.count).finish()
    }
}

impl<'a> Parts<'a> {
    pub(crate) fn new(model: *const sys::csmModel, count: usize) -> Self {
        Self {
            model,
            count,
            _marker: PhantomData,
        }
    }

    /// Number of parts in the model.
    pub fn len(&self) -> usize {
        self.count
    }

    /// `true` iff the model has no parts.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Borrow the part at `index`, or `None` if out of bounds.
    pub fn get(&self, index: usize) -> Option<PartView<'a>> {
        if index >= self.count {
            return None;
        }
        Some(PartView {
            model: self.model,
            index,
            _marker: PhantomData,
        })
    }

    /// Iterate every part in index order.
    pub fn iter(&self) -> PartsIter<'a> {
        PartsIter {
            model: self.model,
            count: self.count,
            next: 0,
            _marker: PhantomData,
        }
    }
}

/// Iterator over [`PartView`]s.
pub struct PartsIter<'a> {
    model: *const sys::csmModel,
    count: usize,
    next: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> Iterator for PartsIter<'a> {
    type Item = PartView<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next >= self.count {
            return None;
        }
        let index = self.next;
        self.next += 1;
        Some(PartView {
            model: self.model,
            index,
            _marker: PhantomData,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let r = self.count - self.next;
        (r, Some(r))
    }
}

impl<'a> ExactSizeIterator for PartsIter<'a> {}

impl<'a> std::fmt::Debug for PartsIter<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartsIter")
            .field("count", &self.count)
            .field("next", &self.next)
            .finish()
    }
}

/// One part's metadata + current opacity.
pub struct PartView<'a> {
    model: *const sys::csmModel,
    index: usize,
    _marker: PhantomData<&'a sys::csmModel>,
}

impl<'a> std::fmt::Debug for PartView<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartView")
            .field("index", &self.index)
            .field("id", &self.id())
            .field("opacity", &self.opacity())
            .finish()
    }
}

impl<'a> PartView<'a> {
    /// Part ID (e.g. `Part_Hair`, `Part_FaceBase`).
    pub fn id(&self) -> &'a str {
        unsafe {
            let ids = sys::csmGetPartIds(self.model);
            if ids.is_null() {
                return "";
            }
            cstr_to_str(*ids.add(self.index))
        }
    }

    /// Index in the part array.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Current opacity in `[0, 1]`. Multiplied with child
    /// drawables' per-drawable opacity at render time.
    pub fn opacity(&self) -> f32 {
        unsafe {
            let arr = sys::csmGetPartOpacities(self.model as *mut _);
            if arr.is_null() {
                0.0
            } else {
                *arr.add(self.index)
            }
        }
    }

    /// Set the part opacity. Used by expression overrides
    /// (M4.2 `.exp3.json`) to hide/show non-mouth facial features.
    pub fn set_opacity(&self, opacity: f32) {
        unsafe {
            let arr = sys::csmGetPartOpacities(self.model as *mut _);
            if !arr.is_null() {
                *arr.add(self.index) = opacity;
            }
        }
    }

    /// Index of the parent part (for opacity inheritance), or
    /// `None` if this is a top-level part.
    pub fn parent_index(&self) -> Option<usize> {
        unsafe {
            let arr = sys::csmGetPartParentPartIndices(self.model);
            if arr.is_null() {
                return None;
            }
            let raw = *arr.add(self.index);
            if raw < 0 {
                None
            } else {
                Some(raw as usize)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_type_decodes_known_constants() {
        assert_eq!(ParameterType::from_raw(0), ParameterType::Normal);
        assert_eq!(ParameterType::from_raw(1), ParameterType::BlendShape);
        // Unknown values are surfaced rather than swallowed.
        assert_eq!(ParameterType::from_raw(7), ParameterType::Unknown(7));
    }
}
