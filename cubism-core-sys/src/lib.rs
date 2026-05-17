//! Raw FFI bindings to **Live2D Cubism Core** — the proprietary C
//! runtime (`Live2DCubismCore.{a,lib}`) that parses `.moc3` files and
//! computes post-deformer mesh data from parameter values.
//!
//! This is a thin `-sys` crate. It exposes every `csm*` symbol from
//! the SDK's `Core/include/Live2DCubismCore.h` header, generated at
//! build time by `bindgen`. **No safety, no idioms, no allocations.**
//! The safe Rust wrapper lives in the sibling `cubism-core` crate.
//!
//! # Acquiring + linking the SDK
//!
//! See [`CUBISM_SDK.md`](../CUBISM_SDK.md). Briefly:
//!
//! 1. Download the Cubism SDK for Native from
//!    <https://www.live2d.com/sdk/download/native/> (license-gated).
//! 2. Unpack to a directory of your choice.
//! 3. Set `LIVE2D_CUBISM_CORE_DIR=/path/to/CubismSdkForNative-5-r.X`.
//!
//! `build.rs` reads the env var, picks the right static lib for the
//! host triple, and runs `bindgen` against the public header.
//!
//! The SDK is **not redistributed by this crate** — only the binding
//! glue is committed. Each developer + CI host installs its own
//! copy.
//!
//! # API safety
//!
//! Every binding is `unsafe`. In particular:
//!
//! * Pointer lifetimes are not enforced — callers must keep the `Moc`
//!   buffer alive while a `Model` is using it.
//! * `csmInitializeAmountOfMemory` etc. take raw byte buffers; passing
//!   anything other than the result of `csmReviveMocInPlace` is UB.
//! * The Cubism runtime is **not thread-safe**. Don't share a `Model`
//!   across threads without external synchronization.
//!
//! Use the safe wrapper unless you have a reason not to.

#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![allow(clippy::useless_transmute)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

#[cfg(test)]
mod smoke {
    //! A single ABI-presence smoke test. Runs without loading any
    //! `.moc3` — `csmGetVersion` is a static function call that
    //! returns the linked SDK's version number. If the static lib
    //! linked correctly, this returns a non-zero u32; if linkage
    //! broke, the test fails to link, not at runtime.

    use super::*;

    #[test]
    fn linked_sdk_reports_a_nonzero_version() {
        // `csmGetVersion` is the simplest symbol in the SDK — no
        // arguments, no pointers, no allocations. If our build.rs
        // wires up linking correctly, this returns the SDK version
        // packed as `(major << 24) | (minor << 16) | patch`. r5.x
        // is `0x05_00_xx_xx` or higher.
        let version = unsafe { csmGetVersion() };
        assert!(
            version > 0,
            "csmGetVersion returned 0 — SDK linkage may be broken"
        );
        // r5 = 0x05000000, so any post-r5 build will be >= that.
        assert!(
            version >= 0x05_00_00_00,
            "linked Cubism Core is older than r5 ({:#010x}) — \
             update LIVE2D_CUBISM_CORE_DIR to a current SDK",
            version
        );
    }
}
