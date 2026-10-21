//! The ambient decision gate and ambient state machine, ported from naru-ios (naru task 2031):
//! `NaruAmbientGate/InternDecisionGate.swift` and `NaruKit/AmbientMachine.swift`.
//!
//! Docs: `docs/ambient.md`. `server/live.rs` runs each `ambient` line through an [`machine::AmbientMachine`]
//! and the gate (naru task 2208). Everything runs locally, with no API key.
//!
//! * [`machine`]: the pure state machine (no I/O, no clock; `now_ms` is passed in).
//! * [`intern`]: the Intern-Decision-4B prompt, the calibrated A/B/C readout and the routing rule,
//!   all pure, so they are tested without a model.
//! * `llama` (cargo feature `llama`): the llama.cpp inference behind [`intern::InternGate`].

pub mod intern;
pub mod machine;

#[cfg(feature = "llama")]
pub mod llama;

use machine::LogLine;

/// What the gate made of one chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateKind {
    Idea,
    CanHelp,
    Ignore,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GateVerdict {
    pub kind: GateKind,
    pub score: f64,
}

impl GateVerdict {
    /// The fail-open answer: any error is "nothing worth surfacing".
    pub const IGNORE: GateVerdict = GateVerdict {
        kind: GateKind::Ignore,
        score: 0.0,
    };
}

/// Decides whether something he said in ambient mode is worth surfacing. Blocking: a caller on the
/// async runtime must run it under `spawn_blocking`.
pub trait AmbientGate: Send + Sync {
    fn evaluate(&self, text: &str, context: &[LogLine]) -> GateVerdict;
}

/// Ignores everything.
pub struct NullGate;

impl AmbientGate for NullGate {
    fn evaluate(&self, _text: &str, _context: &[LogLine]) -> GateVerdict {
        GateVerdict::IGNORE
    }
}

/// The gate for an optional model file. A missing file, a build without the `llama` feature or a
/// load failure is logged and yields [`NullGate`], so the server always starts.
pub fn load_gate(model: Option<&std::path::Path>) -> std::sync::Arc<dyn AmbientGate> {
    let Some(path) = model else {
        return std::sync::Arc::new(NullGate);
    };
    #[cfg(feature = "llama")]
    match llama::LlamaGate::load(path) {
        Ok(g) => return std::sync::Arc::new(intern::InternGate::new(g)),
        Err(e) => eprintln!("ambient gate: {e}; ambient chunks will be ignored"),
    }
    #[cfg(not(feature = "llama"))]
    eprintln!(
        "ambient gate: built without the `llama` feature, cannot load {}; ambient chunks will be ignored",
        path.display()
    );
    std::sync::Arc::new(NullGate)
}
