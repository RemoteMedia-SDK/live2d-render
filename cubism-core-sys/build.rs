//! `cubism-core-sys` build script.
//!
//! Reads `LIVE2D_CUBISM_CORE_DIR` (set by the developer per the
//! crate's CUBISM_SDK.md), picks the right static lib for the host
//! triple, runs bindgen against the public header, and emits the
//! linker incantations. The SDK is **not** redistributed by this
//! crate; the env var is the contract.
//!
//! ## Per-platform layout (Cubism SDK for Native r5.x)
//!
//! ```text
//! $LIVE2D_CUBISM_CORE_DIR/
//!   Core/
//!     include/Live2DCubismCore.h
//!     lib/
//!       macos/{arm64,x86_64}/libLive2DCubismCore.a
//!       linux/x86_64/libLive2DCubismCore.a
//!       windows/{x86,x86_64}/{141,142,143}/Live2DCubismCore_{MD,MDd,MT,MTd}.lib
//! ```
//!
//! On macOS the static archive is fat-by-architecture (one file per
//! arch); we pick by `cfg!(target_arch)`.
//!
//! Windows additionally needs the visual-studio toolset version
//! (`141` = VS 2017, `142` = VS 2019, `143` = VS 2022) and the C
//! runtime flavour (MT/MTd static, MD/MDd dynamic). We pin to `143`
//! + `MD` (matches `cc-rs`'s default for VS 2022 dynamic CRT) since
//! that's what the rest of this workspace uses; fall back via env
//! var override for users on older toolchains.

use std::env;
use std::path::PathBuf;

const ENV_VAR: &str = "LIVE2D_CUBISM_CORE_DIR";

fn main() {
    println!("cargo:rerun-if-env-changed={ENV_VAR}");
    println!("cargo:rerun-if-env-changed=CUBISM_CORE_LIB_KIND");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");

    let sdk_dir = env::var(ENV_VAR).unwrap_or_else(|_| {
        // Fail with an actionable message rather than letting bindgen
        // produce a confusing missing-header error several lines down.
        // Per the avatar plan §M4.0 step 2, this is the expected
        // failure shape when the env var isn't set.
        panic!(
            "{ENV_VAR} is not set.\n\
             Set it to the unpacked Cubism SDK for Native directory.\n\
             See crates/cubism-core-sys/CUBISM_SDK.md for the\n\
             license-gated download + how to point at it."
        );
    });
    let sdk = PathBuf::from(&sdk_dir);

    let header = sdk.join("Core/include/Live2DCubismCore.h");
    if !header.exists() {
        panic!(
            "{ENV_VAR}={sdk_dir} does not contain Core/include/Live2DCubismCore.h.\n\
             Did you set the env var to the SDK *parent* directory by mistake?\n\
             It must point at the directory unpacked from CubismSdkForNative-*.zip."
        );
    }

    let (lib_dir, lib_name) = pick_lib(&sdk);
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static={lib_name}");

    // The Live2DCubismCore static archive on Linux references the
    // standard libm; macOS and Windows libs already pull what they
    // need via Foundation / msvcrt.
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=m");
    }
    // On macOS the lib has no extra deps (no Foundation/CoreVideo).

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", sdk.join("Core/include").display()))
        // Whitelist only the public surface (everything in the header
        // is `csm*`-prefixed). Keeps the generated bindings small +
        // stable across SDK revs.
        .allowlist_function("csm.*")
        .allowlist_type("csm.*")
        .allowlist_var("csm.*")
        // The SDK targets C99; default-derive everything serde-ish
        // can use. `Default` requires every field to also impl it,
        // which is fine for plain structs of primitives.
        .derive_default(true)
        .derive_debug(true)
        .derive_copy(true)
        // Suppress comments — Cubism's headers contain Doxygen blocks
        // with characters that occasionally upset rustdoc.
        .generate_comments(false)
        // Without `layout_tests = false` bindgen emits 30+ size/align
        // tests that bloat compile time and add nothing for our use.
        .layout_tests(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed for Live2DCubismCore.h");

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_dir.join("bindings.rs"))
        .expect("failed to write bindings.rs");
}

/// Resolve `(lib_search_dir, lib_name)` for the host triple. The
/// `lib_name` is the `static=<name>` that goes into the rustc-link-lib
/// directive (no `lib` prefix, no extension).
fn pick_lib(sdk: &PathBuf) -> (PathBuf, String) {
    let core_lib = sdk.join("Core/lib");

    if cfg!(target_os = "macos") {
        let arch = if cfg!(target_arch = "aarch64") {
            "arm64"
        } else if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else {
            panic!("unsupported macOS architecture for Cubism Core");
        };
        return (core_lib.join("macos").join(arch), "Live2DCubismCore".into());
    }

    if cfg!(target_os = "linux") {
        // x86_64 lives under Core/lib/linux/x86_64. ARM64 (e.g. the
        // GitHub-hosted `ubuntu-24.04-arm` matrix runner) and
        // ARMv7 / RPi ship under Core/lib/experimental/. All three are
        // .a static archives with the same link surface, so the
        // `static=Live2DCubismCore` directive at the call site is
        // unchanged — only the search dir differs.
        let sub = if cfg!(target_arch = "x86_64") {
            "linux/x86_64"
        } else if cfg!(target_arch = "aarch64") {
            "experimental/linux/arm64"
        } else if cfg!(target_arch = "arm") {
            // armv7 / armhf — Raspberry Pi 32-bit userspace
            "experimental/rpi"
        } else {
            panic!(
                "Cubism SDK for Native ships static libs for Linux \
                 x86_64 (Core/lib/linux/x86_64), AArch64 \
                 (Core/lib/experimental/linux/arm64), and ARMv7 / RPi \
                 (Core/lib/experimental/rpi). Target arch isn't one of \
                 those."
            );
        };
        return (core_lib.join(sub), "Live2DCubismCore".into());
    }

    if cfg!(target_os = "windows") {
        let arch = if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else {
            "x86"
        };
        // Toolset + CRT flavour: default to VS 2022 + dynamic CRT.
        // Override with `CUBISM_CORE_LIB_KIND=143/MT` etc. if needed.
        let kind = env::var("CUBISM_CORE_LIB_KIND").unwrap_or_else(|_| "143/MD".into());
        let mut parts = kind.splitn(2, '/');
        let toolset = parts.next().unwrap_or("143");
        let crt = parts.next().unwrap_or("MD");
        return (
            core_lib.join("windows").join(arch).join(toolset),
            format!("Live2DCubismCore_{crt}"),
        );
    }

    panic!("unsupported host OS for Cubism Core static linking");
}
