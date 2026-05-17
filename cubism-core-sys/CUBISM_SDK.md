# Cubism SDK for Native — acquisition + linkage

`cubism-core-sys` and the safe wrapper `cubism-core` link **Live2D
Cubism Core**, the proprietary C runtime that parses `.moc3` files
and computes post-deformer mesh data. Cubism Core is **not
redistributed by this repository**. Each developer and CI host
installs their own copy.

## Why the env var, not a vendored copy

Live2D's licence forbids redistributing the SDK as part of an
unrelated repository. The crate's `build.rs` therefore reads
`LIVE2D_CUBISM_CORE_DIR` at build time and links the static library
from that path. The crate's source tree contains only:

- `wrapper.h` — bindgen entry point that `#include`s the SDK header.
- `build.rs` — picks the right static lib per host triple, runs
  bindgen.
- `src/lib.rs` — `include!`s the bindgen output.

No SDK bytes are ever committed.

## Acquisition

1. Visit <https://www.live2d.com/sdk/download/native/>.
2. Accept the **Live2D Open Software License** (the click-through
   EULA — non-commercial + Small-Scale Operator are free; PRO
   Operator is paid; see <https://www.live2d.com/eula/> for the
   exact terms).
3. Download the latest `CubismSdkForNative-5-r.X.zip`.
4. Unpack anywhere convenient. The unpacked directory layout is:

   ```text
   CubismSdkForNative-5-r.X/
     Core/
       include/Live2DCubismCore.h
       lib/
         macos/{arm64,x86_64}/libLive2DCubismCore.a
         linux/x86_64/libLive2DCubismCore.a
         windows/{x86,x86_64}/{141,142,143}/Live2DCubismCore_{MD,MDd,MT,MTd}.lib
     Framework/         (C++ framework — not used by this crate)
     Samples/           (sample projects — not used by this crate)
   ```

5. Export the env var pointing at the **unpacked top-level dir** (the
   one *containing* `Core/`, not `Core/` itself):

   ```bash
   # zsh / bash
   export LIVE2D_CUBISM_CORE_DIR=/path/to/CubismSdkForNative-5-r.X

   # PowerShell
   $env:LIVE2D_CUBISM_CORE_DIR = 'C:\path\to\CubismSdkForNative-5-r.X'
   ```

   For a permanent local setup, add the `export` to your shell rc
   file. For CI, set it as a secret-style env var per host.

## Verifying linkage

After setting the env var:

```bash
cargo build -p cubism-core-sys
```

should succeed. To run the ABI-presence smoke test (calls
`csmGetVersion()` against the linked SDK):

```bash
cargo test -p cubism-core-sys
```

If the env var is unset, the build fails fast with an actionable
error pointing back at this file.

## Per-platform notes

### macOS

The SDK ships fat-by-architecture: separate `arm64` and `x86_64`
static archives under `Core/lib/macos/`. `build.rs` selects via
`cfg!(target_arch)`. Apple Silicon → `arm64`; Rosetta-host or
older Intel Macs → `x86_64`.

### Linux

Only `Core/lib/linux/x86_64/libLive2DCubismCore.a` is wired up.
`Core/dll/experimental/{linux/arm64,rpi}/libLive2DCubismCore.so`
exist in the SDK but aren't used by this crate yet — they are
shared libraries (not static), so the linkage shape is different.
Patch `build.rs` if you need them.

### Windows

The SDK ships static libs for VS 2017 (`141`), VS 2019 (`142`), and
VS 2022 (`143`), each in MT / MTd / MD / MDd flavours. `build.rs`
defaults to `143/MD` (VS 2022 with dynamic CRT — the default for
`cc-rs` and most modern Rust toolchains). Override via:

```powershell
$env:CUBISM_CORE_LIB_KIND = '142/MT'   # VS 2019, static CRT
```

The format is `<toolset>/<crt>` where `<toolset>` is `141|142|143`
and `<crt>` is `MD|MDd|MT|MTd`.

## License

This crate's source is dual-licensed MIT OR Apache-2.0. **The
linked Cubism Core binary is governed separately by Live2D's own
terms** — accepting them is your responsibility, not this crate's.
