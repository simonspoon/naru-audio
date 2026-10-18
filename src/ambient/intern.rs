//! Intern-Decision-4B Q4_K_M as the ambient gate (naru-ios `InternDecisionGate.swift`): prompt `g1`,
//! calibrated A/B/C readout. Prompt, rendering and threshold are the ones measured in
//! system1-survey/docs/ambient-gate-native.md. The pieces here are pure; the model sits behind
//! [`Readout`].

use super::machine::LogLine;
use super::{AmbientGate, GateKind, GateVerdict};

/// Flag when `1 - P(ignore)` is above this.
pub const THRESHOLD: f64 = 0.9;
/// Calibration temperature of the checkpoint's readout.
pub const TEMPERATURE: f32 = 1.992_418_3; // 1.99241824 in the Swift source (same f32)
/// Characters of the chunk kept; the context is 4096 tokens.
pub const MAX_CHUNK_CHARACTERS: usize = 4_000;
/// Tokens the context holds.
pub const CONTEXT_TOKENS: usize = 4096;

const SYSTEM: &str = "You are a careful decision assistant. Use the state and decision schema in the user message to make the requested decisions. For every field, choose exactly one answer symbol (e.g. A, B, C, ...) from its listed options and return one valid JSON object mapping each field name to its chosen symbol. Use the field names and symbols exactly as given. Do not include explanations, Markdown, or extra text.";
const QUESTION: &str =
    "What is this chunk of the user's own speech, overheard by an always-on microphone?";
const OPTIONS: [(&str, &str); 3] = [
    (
        "idea",
        "the user thinking aloud about something worth noting: an idea, a to-do, a plan, or a bug to remember",
    ),
    (
        "can_help",
        "the user stuck on a problem or asking a question that an AI assistant could help with right now",
    ),
    (
        "ignore",
        "side talk, small talk, chatter, an acknowledgement, a remark that needs nothing, TV or radio, or speech recognition noise",
    ),
];

/// Softmax of `logits / temperature`: probabilities (idea, can_help, ignore).
pub fn calibrate(logits: [f32; 3]) -> [f64; 3] {
    let z = logits.map(|l| f64::from(l / TEMPERATURE));
    let top = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e = z.map(|v| (v - top).exp());
    let sum: f64 = e.iter().sum();
    e.map(|v| v / sum)
}

/// Flagged iff `1 - P(ignore) > THRESHOLD`; then idea when `P(idea) >= P(can_help)`.
pub fn route(p: [f64; 3]) -> GateVerdict {
    let score = 1.0 - p[2];
    if score > THRESHOLD {
        let kind = if p[0] >= p[1] {
            GateKind::Idea
        } else {
            GateKind::CanHelp
        };
        GateVerdict { kind, score }
    } else {
        GateVerdict {
            kind: GateKind::Ignore,
            score,
        }
    }
}

/// A JSON string the way serde_json's pretty printer writes it (the measured runner's): only `"`,
/// `\` and control characters are escaped.
fn quoted(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0C}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

pub fn render_state(text: &str) -> String {
    format!(
        "{{\n  \"speaker\": {},\n  \"latest_chunk\": {}\n}}",
        quoted("the user (his own voice)"),
        quoted(text)
    )
}

/// The full prompt for a chunk, cut to `characters` characters.
pub fn render(text: &str, characters: usize) -> String {
    let mut schema = format!("route: {QUESTION}");
    for (symbol, (name, description)) in ["A", "B", "C"].iter().zip(OPTIONS) {
        schema.push_str(&format!("\n    {symbol} = {name}: {description}"));
    }
    let chunk: String = text.chars().take(characters).collect();
    format!(
        "<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\nReturn one answer for every field using the supplied answer symbols.\n\n## State\n{}\n## Decision schema\n{schema}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{{\n    \"route\": \"<decision>\"\n}}<|im_end|>\n",
        render_state(&chunk)
    )
}

/// Runs the model on a rendered prompt and reads the logits of A, B and C at the `<decision>` marker.
/// Implemented by the llama binding; blocking.
pub trait Readout: Send + Sync {
    /// `Err` is any failure (tokenizer, decode, a dead worker); the gate treats it as ignore.
    fn logits(&self, text: &str) -> Result<[f32; 3], String>;
}

pub struct InternGate<R: Readout> {
    readout: R,
}

impl<R: Readout> InternGate<R> {
    pub fn new(readout: R) -> Self {
        Self { readout }
    }

    /// Calibrated probabilities (idea, can_help, ignore).
    pub fn probabilities(&self, text: &str) -> Result<[f64; 3], String> {
        self.readout.logits(text).map(calibrate)
    }
}

impl<R: Readout> AmbientGate for InternGate<R> {
    fn evaluate(&self, text: &str, _context: &[LogLine]) -> GateVerdict {
        match self.probabilities(text) {
            Ok(p) => route(p),
            Err(e) => {
                eprintln!("ambient gate failed, treating the chunk as ignore: {e}");
                GateVerdict::IGNORE
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_threshold_edges() {
        // exactly 0.9 is not above it
        assert_eq!(route([0.05, 0.05, 0.1]).kind, GateKind::Ignore);
        let v = route([0.5, 0.45, 0.05]);
        assert_eq!(v.kind, GateKind::Idea);
        assert!((v.score - 0.95).abs() < 1e-12);
        assert_eq!(route([0.3, 0.65, 0.05]).kind, GateKind::CanHelp);
        // a tie goes to idea
        assert_eq!(route([0.475, 0.475, 0.05]).kind, GateKind::Idea);
        assert_eq!(route([0.0, 0.0, 1.0]).kind, GateKind::Ignore);
    }

    #[test]
    fn calibrate_is_a_softmax_over_scaled_logits() {
        let p = calibrate([0.0, 0.0, 0.0]);
        for v in p {
            assert!((v - 1.0 / 3.0).abs() < 1e-12);
        }
        let p = calibrate([2.0, 0.0, 0.0]);
        let e = (2.0f64 / f64::from(TEMPERATURE)).exp();
        assert!((p[0] - e / (e + 2.0)).abs() < 1e-6);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        // large logits do not overflow
        assert!(calibrate([1e4, 0.0, 0.0])[0] > 0.999);
    }

    #[test]
    fn state_escapes_only_quote_backslash_and_controls() {
        assert_eq!(
            render_state("a \"b\" \\ \n\t\u{1} é/"),
            "{\n  \"speaker\": \"the user (his own voice)\",\n  \"latest_chunk\": \"a \\\"b\\\" \\\\ \\n\\t\\u0001 é/\"\n}"
        );
    }

    #[test]
    fn render_matches_the_swift_prompt() {
        let p = render("hello", MAX_CHUNK_CHARACTERS);
        assert!(p.starts_with("<|im_start|>system\nYou are a careful decision assistant."));
        assert!(p.contains(
            "<|im_start|>user\nReturn one answer for every field using the supplied answer symbols.\n\n## State\n{\n  \"speaker\""
        ));
        assert!(p.contains(
            "\n## Decision schema\nroute: What is this chunk of the user's own speech, overheard by an always-on microphone?\n    A = idea: the user thinking aloud"
        ));
        assert!(p.contains("\n    C = ignore: side talk, small talk,"));
        assert!(p.ends_with(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n{\n    \"route\": \"<decision>\"\n}<|im_end|>\n"
        ));
    }

    #[test]
    fn render_cuts_by_characters_not_bytes() {
        let p = render("ééé", 2);
        assert!(p.contains("\"latest_chunk\": \"éé\""));
        assert!(!p.contains("ééé"));
    }

    struct Fixed(Result<[f32; 3], String>);
    impl Readout for Fixed {
        fn logits(&self, _: &str) -> Result<[f32; 3], String> {
            self.0.clone()
        }
    }

    #[test]
    fn failure_is_ignore_with_score_zero() {
        let g = InternGate::new(Fixed(Err("boom".into())));
        assert_eq!(g.evaluate("x", &[]), GateVerdict::IGNORE);
        let g = InternGate::new(Fixed(Ok([10.0, 0.0, -10.0])));
        assert_eq!(g.evaluate("x", &[]).kind, GateKind::Idea);
    }
}
