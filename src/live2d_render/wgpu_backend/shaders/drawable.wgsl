// Live2D drawable shader.
//
// One shader serves three roles via per-pipeline blend states +
// per-drawable `mask_flags`:
//
// 1. Main draw with Normal blend (premultiplied alpha) — current.
// 2. Main draw with Additive blend (Cubism `BLEND_ADDITIVE`) — uses
//    a different pipeline-side `BlendState` (ONE/ONE color).
// 3. Main draw with Multiplicative blend (Cubism `BLEND_MULTIPLICATIVE`)
//    — pipeline-side `DST_COLOR/ONE_MINUS_SRC_ALPHA`.
// 4. Mask pre-pass — accumulates alpha into a separate offscreen
//    target. Rust-side configures color writes = ALPHA only.
//
// Coordinate convention:
// - vertex positions are in Cubism model-space (units of "model
//   units"; not pixels).
// - the projection uniform converts model-space → NDC.
// - UVs are Y-flipped to match Cubism's bottom-left UV origin.
//
// Mask sampling:
// - `mask_flags.x = 1.0` → sample `mask_tex`, multiply output alpha
//   by mask alpha. `0.0` → ignore mask binding (still bound, just
//   not sampled).
// - `mask_flags.y = 1.0` → invert the mask (1 - mask).

struct VsIn {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    // Mask UV is the screen-space NDC of this fragment, remapped to
    // [0, 1]. Lets us sample `mask_tex` (which was rendered with the
    // same projection) at the same screen position.
    @location(1) mask_uv: vec2<f32>,
};

struct DrawableUniforms {
    projection: mat4x4<f32>,
    multiply: vec4<f32>,
    screen: vec4<f32>,
    // x: mask enabled (1.0/0.0)
    // y: inverted mask (1.0/0.0)
    // z, w: reserved
    mask_flags: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: DrawableUniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var mask_tex: texture_2d<f32>;

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    let clip = u.projection * vec4<f32>(in.position, 0.0, 1.0);
    out.clip_pos = clip;
    // Cubism authors UVs with bottom-left origin; wgpu textures
    // sample with top-left origin. Flip on the way through.
    out.uv = vec2<f32>(in.uv.x, 1.0 - in.uv.y);
    // Map NDC [-1, 1] (Y up) to UV [0, 1] (Y down for wgpu textures).
    out.mask_uv = vec2<f32>((clip.x + 1.0) * 0.5, (1.0 - clip.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, samp, in.uv);
    // Cubism's standard pre-blend formula:
    //   out_rgb = (texel.rgb * multiply.rgb) + screen.rgb * texel.a
    //   out_a   = texel.a * multiply.a
    var modulated_rgb = texel.rgb * u.multiply.rgb + u.screen.rgb * texel.a;
    var modulated_a = texel.a * u.multiply.a;

    // Optional mask sampling. We use the UV in screen space because
    // the mask texture was rendered with the same projection as the
    // main draw target.
    if (u.mask_flags.x > 0.5) {
        let mask_a = textureSample(mask_tex, samp, in.mask_uv).a;
        // `mix(a, 1-a, t)` selects `1-a` when t=1 (inverted mask).
        let m = mix(mask_a, 1.0 - mask_a, u.mask_flags.y);
        modulated_a = modulated_a * m;
    }

    // Premultiply alpha into RGB so the standard
    // `BlendState::PREMULTIPLIED_ALPHA_BLENDING` pipeline produces
    // the right composite (`src + dst*(1-src.a)`). The Additive +
    // Multiplicative blend pipelines configure their factors to
    // play nicely with this same premultiplied output.
    return vec4<f32>(modulated_rgb * modulated_a, modulated_a);
}
