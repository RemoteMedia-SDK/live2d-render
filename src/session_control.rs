//! Inlined session-control envelope helpers.
//!
//! Mirrors `crate::transport::session_control::{aux_port_of,
//! BARGE_IN_PORT, AUX_PORT_ENVELOPE_KEY}` from `remotemedia-core`.
//! Replicated here so this Path-3 plugin doesn't link the host crate.

use remotemedia_plugin_sdk::types::RuntimeData;

/// Envelope field name for aux-port publishes.
pub const AUX_PORT_ENVELOPE_KEY: &str = "__aux_port__";

/// Reserved aux port name for "the user has barged in / cancel the
/// in-flight call".
pub const BARGE_IN_PORT: &str = "barge_in";

/// If `data` is an aux-port envelope, return the port name. Otherwise
/// return `None`.
pub fn aux_port_of(data: &RuntimeData) -> Option<&str> {
    match data {
        RuntimeData::Json(v) => v.get(AUX_PORT_ENVELOPE_KEY).and_then(|p| p.as_str()),
        _ => None,
    }
}
