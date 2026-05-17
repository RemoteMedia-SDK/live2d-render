//! `AlignedBuffer` — heap allocation with a power-of-two alignment
//! requirement higher than `Box`'s default.
//!
//! Cubism Core demands 64-byte alignment for moc buffers and 16-byte
//! alignment for model buffers; `Vec<u8>` only guarantees 1-byte. We
//! over-allocate by `align - 1` bytes and find the first aligned
//! offset inside the resulting allocation. The wasted prefix is
//! bounded (max 63 bytes for moc) and amortizes to zero on real-
//! world model sizes (multi-MB mocs, thousands of bytes for models).
//!
//! # Why not `std::alloc::alloc` + a manual `Layout`?
//!
//! `std::alloc::alloc` is `unsafe` and reasoning about ownership +
//! drop is more complex than necessary. Hiding it behind `Vec<u8>`
//! gives us a heap-managed allocation with a trivial Drop impl,
//! and the over-allocation cost is negligible.

/// A heap allocation guaranteed to start at an `align`-byte boundary.
/// `align` must be a power of two; `len` is the logical length of
/// the aligned region (the underlying allocation may be larger).
pub(crate) struct AlignedBuffer {
    storage: Vec<u8>,
    /// Byte offset within `storage` where the aligned region starts.
    offset: usize,
    /// Logical length of the aligned region.
    len: usize,
}

impl AlignedBuffer {
    /// Allocate a fresh buffer with `len` bytes of usable storage,
    /// starting at an `align`-byte boundary.
    pub(crate) fn new(len: usize, align: usize) -> Self {
        assert!(align.is_power_of_two(), "alignment must be a power of two");
        // Worst case: the allocator hands us a pointer 1 past an
        // alignment boundary, requiring up to `align - 1` bytes of
        // padding before the aligned region.
        let padded = len
            .checked_add(align - 1)
            .expect("AlignedBuffer size + alignment overflow");
        let storage = vec![0u8; padded];
        let raw = storage.as_ptr() as usize;
        let aligned = (raw + align - 1) & !(align - 1);
        let offset = aligned - raw;
        debug_assert!(offset < align);
        debug_assert_eq!(((raw + offset) & (align - 1)), 0);
        debug_assert!(offset + len <= storage.len());
        Self {
            storage,
            offset,
            len,
        }
    }

    /// Pointer to the first byte of the aligned region.
    pub(crate) fn as_ptr(&self) -> *const u8 {
        // SAFETY: `offset` is in-bounds for `storage` by
        // construction (asserted in `new`).
        unsafe { self.storage.as_ptr().add(self.offset) }
    }

    /// Mutable pointer to the first byte of the aligned region.
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        // SAFETY: same as `as_ptr`.
        unsafe { self.storage.as_mut_ptr().add(self.offset) }
    }

    /// Mutable slice over the aligned region. Length is `len`,
    /// start is the aligned offset.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
        let len = self.len;
        let p = self.as_mut_ptr();
        // SAFETY: `p` is in-bounds for `len` bytes by construction.
        unsafe { std::slice::from_raw_parts_mut(p, len) }
    }

    /// Logical length of the aligned region.
    pub(crate) fn len(&self) -> usize {
        self.len
    }
}

impl std::fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuffer")
            .field("ptr", &self.as_ptr())
            .field("len", &self.len)
            .field("offset", &self.offset)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_aligned(p: *const u8, align: usize) -> bool {
        (p as usize) & (align - 1) == 0
    }

    #[test]
    fn aligned_to_64_bytes() {
        let mut b = AlignedBuffer::new(1024, 64);
        assert!(is_aligned(b.as_ptr(), 64));
        assert!(is_aligned(b.as_mut_ptr(), 64));
        assert_eq!(b.len(), 1024);
        assert_eq!(b.as_mut_slice().len(), 1024);
    }

    #[test]
    fn aligned_to_16_bytes() {
        let b = AlignedBuffer::new(256, 16);
        assert!(is_aligned(b.as_ptr(), 16));
        assert_eq!(b.len(), 256);
    }

    #[test]
    fn zero_length_still_aligned() {
        // Edge case: len=0. Used by no-op model inits in ports of
        // the SDK if csmGetSizeofModel ever returns 0.
        let b = AlignedBuffer::new(0, 64);
        assert!(is_aligned(b.as_ptr(), 64));
        assert_eq!(b.len(), 0);
    }

    #[test]
    #[should_panic(expected = "alignment must be a power of two")]
    fn rejects_non_power_of_two_align() {
        let _ = AlignedBuffer::new(64, 17);
    }

    #[test]
    fn allocation_is_zero_initialized() {
        let mut b = AlignedBuffer::new(64, 64);
        for &byte in b.as_mut_slice().iter() {
            assert_eq!(byte, 0);
        }
    }
}
