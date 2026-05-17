# live2d-render — Live2D Cubism renderer

Standalone Path 3 Rust cdylib that registers `Live2DRenderNode` into the
[RemoteMedia SDK](https://github.com/RemoteMedia-SDK/remotemedia-sdk)
streaming pipeline registry.

This plugin owns the full headless wgpu + Cubism Core stack for rendering
Live2D models. It consumes the renderer's four input envelope shapes on
its main port:

1. `{kind: "blendshapes", arkit_52, pts_ms, turn_id?}` — pushed into the
   state machine's ring; sampled per tick.
2. `{kind: "emotion", emoji, ...}` — updates the active expression /
   motion group.
3. `{kind: "audio_clock", pts_ms, ...}` — anchors the audio timeline
   (typically tapped off the outbound `AudioSender`).
4. `{kind: "barge_in"}` — resets state; next blendshape re-anchors.

The pacer drives the node's `tick()` at the bound wire clock (typically
the outbound audio media clock, falling back to `framerate` Hz at idle).
Each tick samples the state machine, applies the pose to the model
(VBridger params + cross-faded `.exp3.json` expression overlay + physics
tick), renders one frame via wgpu, and emits a `RuntimeData::Video
{format: Rgb24, ...}` stamped with the configured `video_stream_id`
(default `"avatar"`).

## Use from a manifest

```json
{
  "version": "v1",
  "plugins": ["live2d-render@v0.1.0"],
  "nodes": [
    {
      "id": "avatar",
      "node_type": "Live2DRenderNode",
      "params": {
        "modelPath": "/path/to/aria.model3.json",
        "framerate": 30,
        "videoStreamId": "avatar",
        "width": 1280,
        "height": 720
      }
    }
  ]
}
```

The SDK resolver expands `live2d-render@v0.1.0` to
`github.com/RemoteMedia-SDK/live2d-render`, fetches `plugin.toml`, then
falls through to `release-manifest.json` for the platform-specific
prebuilt `.so` / `.dylib` / `.dll` asset.

## Build the cdylib locally

**⚠ Requires the proprietary Live2D Cubism SDK for Native.**

This plugin links Live2D Cubism Core, which is governed by Live2D's
[Open Software License](https://www.live2d.com/eula/) and is *not*
redistributed by this repository. Each developer + CI host installs
its own copy and points the build at it via an environment variable.

### One-time SDK setup

1. Download `CubismSdkForNative-5-r.X.zip` from
   <https://www.live2d.com/sdk/download/native/> (accept the EULA).
2. Unpack anywhere convenient.
3. Export `LIVE2D_CUBISM_CORE_DIR` pointing at the unpacked top-level
   directory (the one *containing* `Core/`, not `Core/` itself):

   ```bash
   # bash / zsh
   export LIVE2D_CUBISM_CORE_DIR=/path/to/CubismSdkForNative-5-r.X

   # PowerShell
   $env:LIVE2D_CUBISM_CORE_DIR = 'C:\path\to\CubismSdkForNative-5-r.X'
   ```

   See [`cubism-core-sys/CUBISM_SDK.md`](cubism-core-sys/CUBISM_SDK.md)
   for per-platform notes (macOS arm64/x86_64 selection,
   Windows VS toolset + CRT flavour overrides, etc.).

### Build

```bash
git clone https://github.com/RemoteMedia-SDK/live2d-render
cd live2d-render
LIVE2D_CUBISM_CORE_DIR=/path/to/CubismSdkForNative-5-r.X cargo build --release
# → target/release/liblive2d_render_plugin.so
```

Building without `LIVE2D_CUBISM_CORE_DIR` set fails fast in
`cubism-core-sys/build.rs` with an actionable error pointing back at
`cubism-core-sys/CUBISM_SDK.md`.

## What it exports

| Node type           | Input                                       | Output                                                                |
|---------------------|---------------------------------------------|-----------------------------------------------------------------------|
| `Live2DRenderNode`  | Json {blendshapes, emotion, audio_clock, barge_in} | `Video {format: Rgb24, ...}` at the bound media-clock rate (default 30 fps) |

The pacer's `clocked_media_addr` is unbound by default; the node falls
back to a wall pacer at `params.framerate` Hz when no outbound media
clock is bound. Once an outbound audio/video clock is bound to it, the
node ticks at content-time rate (perfect audio/video lip-sync).

## Barge handling

The node responds to `RuntimeData::Json {kind: "barge_in"}` on its
input port and additionally to aux-port `barge_in` control messages
from the session router. On receipt it clears the blendshape ring +
audio anchor + playback anchor; the next blendshape envelope re-anchors
the audio timeline. The active emotion is **preserved** so the avatar
doesn't go emotionally blank just because the user interrupted.

## License

This plugin's source is licensed under the RemoteMedia SDK Community
License 1.0 — see [`LICENSE.md`](LICENSE.md).

**The linked Cubism Core binary is governed separately by Live2D's
own terms** (see <https://www.live2d.com/eula/>). Accepting them is
your responsibility, not this crate's.
