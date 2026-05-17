//! Tier-2 integration test: load the persona-engine **Aria** Live2D
//! model and verify the safe wrapper exposes its drawable + parameter
//! data correctly.
//!
//! Skips cleanly when `LIVE2D_TEST_MODEL_PATH` isn't set. Run locally
//! after `scripts/install-live2d-aria.sh` with:
//!
//! ```bash
//! export LIVE2D_CUBISM_CORE_DIR=$PWD/sdk/CubismSdkForNative-5-r.5
//! export LIVE2D_TEST_MODEL_PATH=$PWD/models/live2d/aria/aria.model3.json
//! cargo test -p cubism-core --test aria_smoke
//! ```
//!
//! `LIVE2D_CUBISM_CORE_DIR` is required at compile time (build-time
//! linkage); `LIVE2D_TEST_MODEL_PATH` is the runtime gate.

use cubism_core::{BlendMode, Moc, Model};
use std::path::{Path, PathBuf};

fn moc_path() -> Option<PathBuf> {
    let path = std::env::var("LIVE2D_TEST_MODEL_PATH").ok()?;
    let model_json = Path::new(&path);
    let parent = model_json.parent()?;
    let bytes = std::fs::read(model_json).ok()?;
    let json: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    let moc_rel = json
        .get("FileReferences")?
        .get("Moc")?
        .as_str()?
        .to_string();
    Some(parent.join(moc_rel))
}

macro_rules! skip_if_no_aria {
    () => {
        match moc_path() {
            Some(p) if p.exists() => p,
            Some(p) => {
                eprintln!(
                    "[skip] LIVE2D_TEST_MODEL_PATH points at a model3.json \
                     whose Moc reference {:?} doesn't exist on disk; skipping",
                    p
                );
                return;
            }
            None => {
                eprintln!(
                    "[skip] LIVE2D_TEST_MODEL_PATH not set or model3.json \
                     unparseable; set it to e.g. \
                     $PWD/models/live2d/aria/aria.model3.json"
                );
                return;
            }
        }
    };
}

#[test]
fn loads_aria_moc_and_reports_a_known_version() {
    let path = skip_if_no_aria!();
    let moc = Moc::load_from_file(&path).expect("load aria.moc3");
    let v = moc.version();
    // Aria ships as a Cubism 4.2+ moc — at minimum >= csmMocVersion_30.
    assert!(
        v >= 1,
        "moc version should be a known csmMocVersion enum value, got {v}"
    );
    eprintln!("Aria moc version: {v}");
}

#[test]
fn aria_model_init_exposes_canvas_and_drawables() {
    let path = skip_if_no_aria!();
    let moc = Moc::load_from_file(&path).expect("load aria.moc3");
    let mut model = Model::from_moc(&moc).expect("init model");
    model.update();

    let canvas = model.canvas_info();
    assert!(canvas.size.x > 0.0, "canvas width must be positive");
    assert!(canvas.size.y > 0.0, "canvas height must be positive");
    assert!(canvas.pixels_per_unit > 0.0);
    eprintln!(
        "Aria canvas: {}x{} px, origin {:?}, ppu {}",
        canvas.size.x, canvas.size.y, canvas.origin, canvas.pixels_per_unit
    );

    // Aria's rigging is non-trivial — expect dozens of drawables
    // (face base, hair, eyes, mouth, etc.). Lower bound is loose
    // enough that any reasonably-rigged anime model passes.
    let drawables = model.drawables();
    assert!(
        drawables.len() >= 20,
        "Aria should expose at least 20 drawables; got {}",
        drawables.len()
    );
    eprintln!("Aria has {} drawables", drawables.len());

    // First drawable: basic shape sanity. Vertex positions must be
    // non-empty + index buffer must reference vertices in range.
    let first = drawables.get(0).expect("first drawable");
    let positions = first.vertex_positions();
    let uvs = first.vertex_uvs();
    let indices = first.indices();
    assert!(!positions.is_empty(), "drawable[0] should have vertices");
    assert_eq!(
        positions.len(),
        uvs.len(),
        "vertex_positions and vertex_uvs must agree"
    );
    assert!(
        !indices.is_empty(),
        "drawable[0] should have an index buffer"
    );
    let max_idx = *indices.iter().max().unwrap();
    assert!(
        (max_idx as usize) < positions.len(),
        "index {max_idx} out of range for {} verts",
        positions.len()
    );

    // Opacity in [0, 1]; render order is a small i32.
    let op = first.opacity();
    assert!((0.0..=1.0).contains(&op), "opacity {op} outside [0, 1]");
    let _ = first.render_order();
    let _ = first.draw_order();

    // BlendMode resolves to one of three. Aria uses a mix —
    // we don't pin which, just that the decode works.
    let _ = first.blend_mode();
    eprintln!(
        "drawable[0]: id={:?}, verts={}, indices={}, opacity={op}, render_order={}, blend={:?}",
        first.id(),
        positions.len(),
        indices.len(),
        first.render_order(),
        first.blend_mode(),
    );
}

#[test]
fn aria_drawables_include_at_least_one_masked_drawable() {
    // Aria's eyes/mouth/etc. clip against face-base masks. Confirms
    // mask plumbing works (the M4.4 wgpu backend will need this).
    let path = skip_if_no_aria!();
    let moc = Moc::load_from_file(&path).expect("load aria.moc3");
    let mut model = Model::from_moc(&moc).expect("init model");
    model.update();

    let mut total_masked = 0usize;
    for d in model.drawables().iter() {
        if !d.masks().is_empty() {
            total_masked += 1;
        }
    }
    assert!(
        total_masked > 0,
        "expected at least one masked drawable in Aria's rig"
    );
    eprintln!("Aria has {total_masked} masked drawables");
}

#[test]
fn aria_parameters_include_vbridger_lipsync_axes() {
    // Per persona-engine's Live2D rigging guide, Aria exposes the
    // VBridger lip-sync parameters the Audio2Face/M2 pipeline
    // ultimately drives. We don't assert specific values — just
    // that the named parameters are findable in the rig.
    let path = skip_if_no_aria!();
    let moc = Moc::load_from_file(&path).expect("load aria.moc3");
    let mut model = Model::from_moc(&moc).expect("init model");
    model.update();

    let params = model.parameters();
    assert!(params.len() > 0, "Aria should expose parameters");
    eprintln!("Aria has {} parameters", params.len());

    // VBridger canonical lip-sync axes Aria is rigged for.
    let want = ["ParamMouthOpenY", "ParamMouthForm", "ParamJawOpen"];
    let found: Vec<_> = want
        .iter()
        .filter(|name| params.find(name).is_some())
        .copied()
        .collect();
    assert!(
        !found.is_empty(),
        "expected at least one of {:?} in Aria's parameters; \
         all parameter ids: {:?}",
        want,
        params
            .iter()
            .map(|p| p.id().to_string())
            .collect::<Vec<_>>()
    );
    eprintln!("Aria exposes lip-sync axes: {:?}", found);

    // Pin one parameter's metadata invariants.
    let p0 = params.get(0).unwrap();
    let (lo, hi) = (p0.min(), p0.max());
    assert!(lo <= hi, "param[0] min ({lo}) must be <= max ({hi})");
    let def = p0.default();
    assert!(
        (lo..=hi).contains(&def) || (lo - def).abs() < 1e-6 || (hi - def).abs() < 1e-6,
        "param[0] default {def} must be inside [{lo}, {hi}]"
    );
}

#[test]
fn parameter_set_value_propagates_to_update() {
    // Smoke: write a parameter value, call update, read it back.
    // Doesn't pin the rendered output, just that the read/write
    // round-trip works.
    let path = skip_if_no_aria!();
    let moc = Moc::load_from_file(&path).expect("load aria.moc3");
    let mut model = Model::from_moc(&moc).expect("init model");
    model.update();

    // Find a param we know exists; fall back to param[0].
    let target_id = {
        let params = model.parameters();
        if let Some(p) = params.find("ParamMouthOpenY") {
            p.id().to_string()
        } else {
            params.get(0).expect("at least one param").id().to_string()
        }
    };

    let (lo, hi) = {
        let params = model.parameters();
        let p = params.find(&target_id).unwrap();
        (p.min(), p.max())
    };
    let target_value = (lo + hi) / 2.0;

    {
        let params = model.parameters_mut();
        params.find(&target_id).unwrap().set_value(target_value);
    }
    model.update();

    let observed = {
        let params = model.parameters();
        params.find(&target_id).unwrap().value()
    };
    let delta = (observed - target_value).abs();
    assert!(
        delta < 1e-3,
        "set_value({target_value}) → update → read = {observed}; delta {delta}"
    );
}

#[test]
fn blend_mode_decodes_consistently() {
    // Pin that constant_flags + blend_mode agree across the whole
    // rig — invariant the M4.4 wgpu backend will rely on.
    let path = skip_if_no_aria!();
    let moc = Moc::load_from_file(&path).expect("load aria.moc3");
    let mut model = Model::from_moc(&moc).expect("init model");
    model.update();

    for d in model.drawables().iter() {
        let flags = d.constant_flags();
        let decoded = BlendMode::from_constant_flags(flags);
        assert_eq!(decoded, d.blend_mode(), "drawable {} disagrees", d.id());
    }
}
