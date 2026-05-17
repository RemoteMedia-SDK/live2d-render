//! Tier-2 integration test: parse the persona-engine **Aria** model
//! manifest end-to-end (model3.json → resolved paths → load Moc +
//! every expression + every motion file).
//!
//! Skips cleanly when `LIVE2D_TEST_MODEL_PATH` isn't set. Companion
//! to `aria_smoke.rs` — that one exercises the moc/model FFI; this
//! one exercises the JSON manifest layer.

use cubism_core::{ExpressionJson, ModelJson, MotionJson};

fn manifest_path() -> Option<std::path::PathBuf> {
    let p = std::env::var("LIVE2D_TEST_MODEL_PATH").ok()?;
    let p = std::path::PathBuf::from(p);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

macro_rules! skip_if_no_aria_manifest {
    () => {
        match manifest_path() {
            Some(p) => p,
            None => {
                eprintln!(
                    "[skip] LIVE2D_TEST_MODEL_PATH not set or file missing; \
                     install Aria via scripts/install-live2d-aria.sh"
                );
                return;
            }
        }
    };
}

#[test]
fn loads_aria_model3_manifest_and_resolves_paths() {
    let path = skip_if_no_aria_manifest!();
    let resolved = ModelJson::load(&path).expect("load aria model3.json");

    // Manifest version + structural sanity.
    assert_eq!(resolved.manifest.version, 3);
    assert_eq!(resolved.manifest.file_references.moc, "aria.moc3");

    // Moc resolves to a file that exists on disk.
    let moc = resolved.moc_path();
    assert!(moc.exists(), "resolved moc path {:?} should exist", moc);

    // Every texture resolves to a real PNG.
    let textures = resolved.texture_paths();
    assert!(
        !textures.is_empty(),
        "Aria should reference at least one texture"
    );
    for t in &textures {
        assert!(t.exists(), "texture {:?} missing", t);
        assert_eq!(
            t.extension().and_then(|s| s.to_str()),
            Some("png"),
            "texture {:?} should be a PNG",
            t
        );
    }

    // Physics + display_info resolved (Aria has both).
    let physics = resolved.physics_path().expect("Aria has physics");
    assert!(physics.exists());
    let cdi = resolved.display_info_path().expect("Aria has display_info");
    assert!(cdi.exists());
}

#[test]
fn loads_every_aria_expression_file() {
    let path = skip_if_no_aria_manifest!();
    let resolved = ModelJson::load(&path).expect("load");
    assert!(
        !resolved.manifest.file_references.expressions.is_empty(),
        "Aria ships expressions"
    );

    let mut names = Vec::new();
    for (name, exp_path) in resolved.expressions() {
        names.push(name.to_string());
        assert!(exp_path.exists(), "expression {:?} missing", exp_path);
        let exp = ExpressionJson::from_file(&exp_path)
            .unwrap_or_else(|e| panic!("parse {:?}: {}", exp_path, e));
        // FadeIn/Out times are positive; zero is allowed but
        // negative would be a rigging bug.
        assert!(exp.fade_in_time >= 0.0);
        assert!(exp.fade_out_time >= 0.0);
        // Aria's expressions all carry the canonical kind string.
        assert_eq!(exp.kind, "Live2D Expression");
    }
    eprintln!("Aria expressions: {:?}", names);

    // Persona-engine's emoji map references at least these
    // expression names; confirm Aria ships them.
    let want = ["neutral", "happy", "sad"];
    for n in want {
        assert!(
            names.iter().any(|x| x == n),
            "Aria should ship expression {n:?}; got {names:?}"
        );
    }
}

#[test]
fn loads_every_aria_motion_file() {
    let path = skip_if_no_aria_manifest!();
    let resolved = ModelJson::load(&path).expect("load");

    // Aria ships at minimum these motion groups (per Live2D.md
    // emoji map: Idle / Talking + a few emotion groups).
    let groups: Vec<_> = resolved.motion_group_names().collect();
    assert!(!groups.is_empty(), "Aria ships motion groups");
    eprintln!("Aria motion groups: {:?}", groups);

    let mut total_motions = 0;
    for group in &groups {
        for (mpath, mref) in resolved.motions(group) {
            total_motions += 1;
            assert!(mpath.exists(), "motion file {:?} missing", mpath);
            assert!(mref.fade_in_time >= 0.0);
            assert!(mref.fade_out_time >= 0.0);

            let m = MotionJson::from_file(&mpath)
                .unwrap_or_else(|e| panic!("parse {:?}: {}", mpath, e));
            assert_eq!(m.version, 3);
            assert!(m.meta.duration > 0.0);
            assert!(m.meta.fps > 0.0);
            // CurveCount metadata should agree with parsed curves.
            assert_eq!(
                m.curves.len(),
                m.meta.curve_count as usize,
                "CurveCount metadata mismatch in {:?}",
                mpath
            );
            // Curves drive at least one parameter (sanity).
            assert!(!m.curves.is_empty(), "motion {:?} has no curves", mpath);
        }
    }
    assert!(
        total_motions >= 5,
        "expected >=5 motion files in Aria; got {total_motions}"
    );
    eprintln!("loaded + parsed {total_motions} Aria motion files");
}

#[test]
fn moc_path_round_trips_to_existing_aria_moc() {
    // Confirms the manifest-driven moc lookup matches what the
    // M4.1 aria_smoke test loads directly. If these diverge, one
    // of the two is broken.
    let path = skip_if_no_aria_manifest!();
    let resolved = ModelJson::load(&path).expect("load");
    let moc_path = resolved.moc_path();
    let bytes = std::fs::read(&moc_path).unwrap_or_else(|e| {
        panic!("read {:?}: {}", moc_path, e);
    });
    // First 4 bytes are the MOC3 magic.
    assert_eq!(
        &bytes[..4],
        b"MOC3",
        "expected MOC3 magic in {:?}",
        moc_path
    );
    assert!(bytes.len() > 1024, ".moc3 should be more than a few KB");
}

#[test]
fn aria_groups_include_lipsync_and_eyeblink_keys() {
    let path = skip_if_no_aria_manifest!();
    let resolved = ModelJson::load(&path).expect("load");
    // Aria's groups currently ship empty Ids arrays, but the keys
    // must be present so the renderer (M4.4) can subscribe.
    assert!(
        resolved.group_ids("LipSync").is_some(),
        "Aria should declare a LipSync group"
    );
    assert!(
        resolved.group_ids("EyeBlink").is_some(),
        "Aria should declare an EyeBlink group"
    );
}

#[test]
fn unknown_expression_returns_none() {
    let path = skip_if_no_aria_manifest!();
    let resolved = ModelJson::load(&path).expect("load");
    assert!(resolved.expression_path("nonexistent_expression").is_none());
}

/// Resolves correctly even when the manifest is loaded by absolute
/// path from outside the repo. (Pins the bug class where a
/// resolver's PathBuf joining drops to relative.)
#[test]
fn resolver_works_with_absolute_manifest_path() {
    let path = skip_if_no_aria_manifest!();
    let absolute: std::path::PathBuf = std::fs::canonicalize(&path).unwrap();
    let resolved = ModelJson::load(&absolute).expect("load with absolute path");
    assert!(resolved.moc_path().is_absolute());
    assert!(resolved.moc_path().exists());
}
