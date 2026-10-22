//! The ambient state machine, ported from naru-ios `NaruKit/AmbientMachine.swift` (naru task 2031).
//!
//! Pure like `live_mode.rs`: no I/O, no clock reads, no async. Time is `now_ms` (any monotonic
//! origin) passed to every method. Swift's `await gate.evaluate(..)` becomes a [`GateRequest`] in the
//! returned [`Ingest`]: the caller runs the gate (off the async runtime) and hands the verdict to
//! [`gate_event`]. [`AmbientMachine::ingest_with`] does both for a synchronous gate.

use std::collections::VecDeque;

use super::{AmbientGate, GateKind, GateVerdict};

/// Who a transcribed chunk was attributed to: `Him` is the enrolled voice, `Unknown` a chunk too
/// short to tag (or heard before anyone enrolled), `Other` a voice that scored below the enrolled one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Him,
    Other,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LogLine {
    pub at_ms: u64,
    pub speaker: Speaker,
    /// The named person an `Other` line was matched to, if the speaker library knew them.
    pub person: Option<String>,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Wake,
    IdeaFlagged,
    CanHelp,
    ModeChanged,
}

impl EventKind {
    /// The wire name (snake_case).
    pub fn as_str(self) -> &'static str {
        match self {
            EventKind::Wake => "wake",
            EventKind::IdeaFlagged => "idea_flagged",
            EventKind::CanHelp => "can_help",
            EventKind::ModeChanged => "mode_changed",
        }
    }
}

/// What the machine tells the server. For `ModeChanged`, `text` is the new mode (`"ambient"`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub kind: EventKind,
    pub text: String,
    pub speaker: Speaker,
    pub at_ms: u64,
    /// The last ~30 s of the log before this event, oldest first.
    pub context: Vec<LogLine>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Ambient,
    Conversation,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Ambient => "ambient",
            Mode::Conversation => "conversation",
        }
    }
}

/// One line that sounds like "Hey Naru" and what became of it. Diagnostics only.
#[derive(Debug, Clone, PartialEq)]
pub struct WakeCandidate {
    pub text: String,
    pub word: String,
    pub distance: f64,
    pub similarity: Option<f32>,
    pub accepted: bool,
    pub reason: &'static str,
}

/// What one chunk produced: events for the server, and a line for the live agent (conversation mode,
/// his voice only).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AmbientOutput {
    pub events: Vec<Event>,
    pub utterance: Option<String>,
    pub wake_candidates: Vec<WakeCandidate>,
}

/// A chunk of his the gate has to judge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateRequest {
    pub text: String,
    pub context: Vec<LogLine>,
    pub now_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Ingest {
    pub output: AmbientOutput,
    pub gate: Option<GateRequest>,
}

/// The event a gate verdict turns into, if any.
pub fn gate_event(req: &GateRequest, verdict: GateVerdict) -> Option<Event> {
    let kind = match verdict.kind {
        GateKind::Idea => EventKind::IdeaFlagged,
        GateKind::CanHelp => EventKind::CanHelp,
        GateKind::Ignore => return None,
    };
    Some(Event {
        kind,
        text: req.text.clone(),
        speaker: Speaker::Him,
        at_ms: req.now_ms,
        context: req.context.clone(),
    })
}

// MARK: Rolling transcript

/// The last two minutes of what was heard, as text. In memory only: a ring bounded by age and count.
#[derive(Debug, Default)]
pub struct TranscriptLog {
    lines: VecDeque<LogLine>,
}

impl TranscriptLog {
    pub const LIFETIME_MS: u64 = 120_000;
    pub const CAPACITY: usize = 500;

    pub fn append(&mut self, now_ms: u64, speaker: Speaker, text: &str) {
        self.append_named(now_ms, speaker, None, text);
    }

    pub fn append_named(
        &mut self,
        now_ms: u64,
        speaker: Speaker,
        person: Option<String>,
        text: &str,
    ) {
        self.prune(now_ms);
        self.lines.push_back(LogLine {
            at_ms: now_ms,
            speaker,
            person,
            text: text.to_string(),
        });
        while self.lines.len() > Self::CAPACITY {
            self.lines.pop_front();
        }
    }

    /// Lines newer than `window_ms`, oldest first.
    pub fn recent(&mut self, now_ms: u64, window_ms: u64) -> Vec<LogLine> {
        self.prune(now_ms);
        self.lines
            .iter()
            .filter(|l| l.at_ms + window_ms >= now_ms)
            .cloned()
            .collect()
    }

    fn prune(&mut self, now_ms: u64) {
        self.lines.retain(|l| l.at_ms + Self::LIFETIME_MS >= now_ms);
    }
}

// MARK: Wake word and end phrase

pub mod phrases {
    /// Spellings Parakeet produced for "Naru" across six `say` voices. "Narrow" is left out because
    /// "narrow" is a word people say.
    pub const WAKE_SPELLINGS: [&str; 5] = ["naru", "neru", "niru", "nauru", "naroo"];
    const WAKE_LEAD_INS: [&str; 5] = ["hey", "hi", "ok", "okay", "wake"];
    const END_PHRASES: [&str; 2] = ["thanks thats all", "thank you thats all"];
    const FUSED_LEAD_INS: [&str; 6] = ["hey", "hei", "hay", "hi", "okay", "ok"];
    /// Weighted edit distance at or under which a word counts as "Naru" said by Parakeet.
    pub const SOUND_ALIKE_MAX_DISTANCE: f64 = 0.45;

    /// Lower-case letters, digits and single spaces only; apostrophes vanish so "that's" and "thats"
    /// agree.
    pub fn normalize(text: &str) -> String {
        let kept: String = text
            .to_lowercase()
            .chars()
            .filter(|&c| c != '\'' && c != '\u{2019}')
            .map(|c| if c.is_alphanumeric() { c } else { ' ' })
            .collect();
        kept.split(' ')
            .filter(|w| !w.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn word_list(text: &str) -> Vec<String> {
        normalize(text)
            .split(' ')
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    }

    /// User-defined wake words as word lists: normalized, lead-ins stripped, empties and duplicates
    /// dropped.
    pub fn wake_phrases(custom: &[String]) -> Vec<Vec<String>> {
        let mut phrases: Vec<Vec<String>> = Vec::new();
        for entry in custom {
            let mut words = word_list(entry);
            while words
                .first()
                .is_some_and(|w| WAKE_LEAD_INS.contains(&w.as_str()))
            {
                words.remove(0);
            }
            if !words.is_empty() && !phrases.contains(&words) {
                phrases.push(words);
            }
        }
        phrases
    }

    /// Built-in spellings act as one-word phrases beside the custom ones.
    fn all_phrases(custom: &[Vec<String>]) -> Vec<Vec<String>> {
        let mut sorted = WAKE_SPELLINGS.to_vec();
        sorted.sort();
        sorted
            .into_iter()
            .map(|s| vec![s.to_string()])
            .chain(custom.iter().cloned())
            .collect()
    }

    /// Index just past the longest phrase that opens the line (after an optional lead-in).
    fn phrase_end(tokens: &[String], custom: &[Vec<String>]) -> Option<usize> {
        let mut best: Option<usize> = None;
        for phrase in all_phrases(custom) {
            for start in [0usize, 1] {
                if start + phrase.len() > tokens.len() {
                    continue;
                }
                if start == 1 && !WAKE_LEAD_INS.contains(&tokens[0].as_str()) {
                    continue;
                }
                if tokens[start..start + phrase.len()] == phrase[..] {
                    best = Some(best.unwrap_or(0).max(start + phrase.len()));
                }
            }
        }
        best
    }

    /// The line's normalized words, each with the index of the whitespace token it came from.
    fn words(text: &str) -> (Vec<String>, Vec<usize>) {
        let mut words = Vec::new();
        let mut origin = Vec::new();
        for (i, token) in text.split_whitespace().enumerate() {
            for word in word_list(token) {
                words.push(word);
                origin.push(i);
            }
        }
        (words, origin)
    }

    /// True when the line is only the wake phrase.
    pub fn is_bare_wake(text: &str, custom: &[Vec<String>]) -> bool {
        let (tokens, _) = words(text);
        phrase_end(&tokens, custom).is_some_and(|end| end == tokens.len())
    }

    /// The wake word must open the line; returns the words after it as spoken, or `None`.
    pub fn wake_remainder(text: &str, custom: &[Vec<String>]) -> Option<String> {
        let spoken: Vec<&str> = text.split_whitespace().collect();
        let (tokens, origin) = words(text);
        let end = phrase_end(&tokens, custom)?;
        Some(spoken[origin[end - 1] + 1..].join(" "))
    }

    /// A bare, greeted utterance whose name merely sounds like a wake word, with the candidate words
    /// and their distance from the closest target.
    pub fn sound_alike(text: &str, custom: &[Vec<String>]) -> Option<(String, f64)> {
        let words = word_list(text);
        let candidate: Vec<String> = match words.len() {
            0 => return None,
            1 => {
                let w = &words[0];
                let lead = FUSED_LEAD_INS
                    .iter()
                    .find(|l| w.starts_with(**l) && w.chars().count() - l.chars().count() >= 3)?;
                vec![w.chars().skip(lead.chars().count()).collect()]
            }
            _ => {
                if !WAKE_LEAD_INS.contains(&words[0].as_str()) {
                    return None;
                }
                let c: Vec<String> = words[1..].to_vec();
                if all_phrases(custom).contains(&c) {
                    return None;
                }
                c
            }
        };
        let mut targets: Vec<Vec<String>> = vec![vec!["naru".to_string()]];
        targets.extend(custom.iter().cloned());
        let mut best: Option<f64> = None;
        for target in &targets {
            if target.len() != candidate.len() {
                continue;
            }
            // Short target words (under 4 letters) must match exactly: "Mix" is not "Max".
            if candidate
                .iter()
                .zip(target)
                .any(|(c, t)| c != t && t.chars().count() < 4)
            {
                continue;
            }
            let distance = candidate
                .iter()
                .zip(target)
                .map(|(c, t)| sound_distance(c, t))
                .fold(0.0, f64::max);
            if distance <= SOUND_ALIKE_MAX_DISTANCE && distance < best.unwrap_or(f64::INFINITY) {
                best = Some(distance);
            }
        }
        best.map(|d| (candidate.join(" "), d))
    }

    /// Levenshtein over lower-case letters with phonetic costs: vowel for vowel .25, n/l/m and r/l
    /// swaps .4, dropping or adding a vowel .5, anything else 1. Doubled letters collapse and a final
    /// "ow" reads as "o" first.
    pub fn sound_distance(a: &str, b: &str) -> f64 {
        fn prepared(word: &str) -> Vec<char> {
            let mut chars: Vec<char> = Vec::new();
            for c in word.chars() {
                if chars.last() != Some(&c) {
                    chars.push(c);
                }
            }
            if chars.len() > 2 && chars[chars.len() - 2..] == ['o', 'w'] {
                chars.pop();
            }
            chars
        }
        let is_vowel = |c: char| matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y');
        let is_nasal = |c: char| matches!(c, 'n' | 'l' | 'm');
        let substitute = |x: char, y: char| {
            if x == y {
                0.0
            } else if is_vowel(x) && is_vowel(y) {
                0.25
            } else if (is_nasal(x) && is_nasal(y))
                || (x == 'r' && y == 'l')
                || (x == 'l' && y == 'r')
            {
                0.4
            } else {
                1.0
            }
        };
        let gap = |c: char| if is_vowel(c) { 0.5 } else { 1.0 };
        let s = prepared(a);
        let t = prepared(b);
        let mut row = vec![0.0f64; t.len() + 1];
        for j in 0..t.len() {
            row[j + 1] = row[j] + gap(t[j]);
        }
        for &si in &s {
            let mut diagonal = row[0];
            row[0] += gap(si);
            for j in 0..t.len() {
                let up = row[j + 1];
                row[j + 1] = (up + gap(si))
                    .min(row[j] + gap(t[j]))
                    .min(diagonal + substitute(si, t[j]));
                diagonal = up;
            }
        }
        row[t.len()]
    }

    pub fn is_end_phrase(text: &str) -> bool {
        let n = normalize(text);
        END_PHRASES.iter().any(|p| n.contains(p))
    }
}

// MARK: Mode machine

/// `Ambient` <-> `Conversation`. The 120 s auto-return is checked on every chunk and on [`tick`].
///
/// [`tick`]: AmbientMachine::tick
#[derive(Debug)]
pub struct AmbientMachine {
    mode: Mode,
    log: TranscriptLog,
    last_heard_him: Option<u64>,
    agent_speaking: bool,
    agent_quiet_at: Option<u64>,
    custom_wake: Vec<Vec<String>>,
    /// The person's name for the next ingested line; consumed by it.
    next_person: Option<String>,
}

impl Default for AmbientMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl AmbientMachine {
    pub const SILENCE_RETURN_MS: u64 = 120_000;
    pub const CONTEXT_WINDOW_MS: u64 = 30_000;
    /// In conversation, a segment too short to tag counts as his when it follows his last tagged line
    /// this closely; any later and it is only context.
    pub const UNKNOWN_CONTINUATION_MS: u64 = 10_000;
    /// How long after the agent stops speaking its voice may still be arriving as a transcript.
    pub const AGENT_ECHO_TAIL_MS: u64 = 1_000;

    pub fn new() -> Self {
        Self {
            mode: Mode::Ambient,
            log: TranscriptLog::default(),
            last_heard_him: None,
            agent_speaking: false,
            agent_quiet_at: None,
            custom_wake: Vec::new(),
            next_person: None,
        }
    }

    /// Names who said the next `ingest`ed line, for its log entry.
    pub fn set_next_person(&mut self, person: Option<String>) {
        self.next_person = person;
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// While true (and for the echo tail after), untagged segments are log-only.
    pub fn set_agent_speaking(&mut self, now_ms: u64, speaking: bool) {
        self.agent_speaking = speaking;
        if !speaking {
            self.agent_quiet_at = Some(now_ms + Self::AGENT_ECHO_TAIL_MS);
        }
    }

    /// User-defined wake words, matched beside the built-in "Naru".
    pub fn set_wake_words(&mut self, words: &[String]) {
        self.custom_wake = phrases::wake_phrases(words);
    }

    pub fn log_lines(&mut self, now_ms: u64) -> Vec<LogLine> {
        self.log.recent(now_ms, TranscriptLog::LIFETIME_MS)
    }

    /// Only the silence timeout can fire here.
    pub fn tick(&mut self, now_ms: u64) -> AmbientOutput {
        let quiet = self
            .last_heard_him
            .is_none_or(|t| now_ms.saturating_sub(t) >= Self::SILENCE_RETURN_MS);
        if self.mode != Mode::Conversation || !quiet {
            return AmbientOutput::default();
        }
        AmbientOutput {
            events: vec![self.return_to_ambient(now_ms, now_ms)],
            ..Default::default()
        }
    }

    /// Ends conversation mode now (live ended), so the next "Hey Naru" wakes.
    pub fn end_conversation(&mut self, now_ms: u64) -> AmbientOutput {
        if self.mode != Mode::Conversation {
            return AmbientOutput::default();
        }
        AmbientOutput {
            events: vec![self.return_to_ambient(now_ms, now_ms)],
            ..Default::default()
        }
    }

    /// `ingest` with a synchronous gate: runs the request and appends the verdict's event.
    pub fn ingest_with(
        &mut self,
        gate: &dyn AmbientGate,
        now_ms: u64,
        speaker: Speaker,
        text: &str,
        similarity: Option<f32>,
    ) -> AmbientOutput {
        let Ingest {
            mut output,
            gate: req,
        } = self.ingest(now_ms, speaker, text, similarity);
        if let Some(req) = req {
            let verdict = gate.evaluate(&req.text, &req.context);
            output.events.extend(gate_event(&req, verdict));
        }
        output
    }

    /// `similarity` is the cosine against the enrolled voice for a segment too short to tag; it is
    /// only logged.
    pub fn ingest(
        &mut self,
        now: u64,
        speaker: Speaker,
        text: &str,
        similarity: Option<f32>,
    ) -> Ingest {
        let mut output = self.tick(now);
        let line = text.trim();
        if line.is_empty() {
            return Ingest { output, gate: None };
        }
        let context = self.log.recent(now, Self::CONTEXT_WINDOW_MS);
        let gate = self.ingest_line(now, speaker, line, similarity, &context, &mut output);
        // Swift's `defer`: the line joins the log after every path out.
        let person = self.next_person.take();
        self.log.append_named(now, speaker, person, line);
        Ingest { output, gate }
    }

    fn ingest_line(
        &mut self,
        now: u64,
        speaker: Speaker,
        line: &str,
        similarity: Option<f32>,
        context: &[LogLine],
        output: &mut AmbientOutput,
    ) -> Option<GateRequest> {
        let was_in_conversation = self.mode == Mode::Conversation;
        let agent_audible = self.agent_speaking || self.agent_quiet_at.is_some_and(|q| now < q);
        let continues_him = was_in_conversation
            && speaker == Speaker::Unknown
            && !agent_audible
            && self
                .last_heard_him
                .is_some_and(|t| now.saturating_sub(t) <= Self::UNKNOWN_CONTINUATION_MS);
        // A wake attempt is usually under 2 s, so it can only be untagged. An untagged segment opening
        // with the wake word wakes like a `Him` line; only `Other` is blocked, and none wakes while the
        // agent is audible.
        let bare_wake = !was_in_conversation
            && speaker == Speaker::Unknown
            && !agent_audible
            && phrases::wake_remainder(line, &self.custom_wake).is_some();
        // A sound-alike ("Hey Naro") is gated like the exact bare wake: any voice but a tagged `Other`.
        let mut sound_alike = false;
        if !was_in_conversation
            && let Some((word, distance)) = phrases::sound_alike(line, &self.custom_wake)
        {
            sound_alike = speaker != Speaker::Other;
            output.wake_candidates.push(WakeCandidate {
                text: line.to_string(),
                word,
                distance,
                similarity,
                accepted: sound_alike,
                reason: match speaker {
                    Speaker::Him => "tagged him",
                    Speaker::Other => "tagged other",
                    Speaker::Unknown => "short, untagged",
                },
            });
        }
        let wake_event = |context: &[LogLine]| Event {
            kind: EventKind::Wake,
            text: line.to_string(),
            speaker: Speaker::Him,
            at_ms: now,
            context: context.to_vec(),
        };
        if sound_alike {
            self.mode = Mode::Conversation;
            self.last_heard_him = Some(now);
            output.events.push(wake_event(context));
            return None;
        }
        if !(speaker == Speaker::Him || continues_him || bare_wake) {
            return None;
        }
        if speaker == Speaker::Him {
            self.last_heard_him = Some(now);
        }

        if was_in_conversation {
            if phrases::is_end_phrase(line) {
                let ev = self.return_to_ambient(now, now);
                output.events.push(ev);
            } else {
                output.utterance = Some(line.to_string());
            }
            return None;
        }

        if let Some(remainder) = phrases::wake_remainder(line, &self.custom_wake) {
            self.mode = Mode::Conversation;
            self.last_heard_him = Some(now);
            output.events.push(wake_event(context));
            if !remainder.is_empty() {
                output.utterance = Some(remainder);
            }
            return None;
        }

        Some(GateRequest {
            text: line.to_string(),
            context: context.to_vec(),
            now_ms: now,
        })
    }

    fn return_to_ambient(&mut self, at_ms: u64, now_ms: u64) -> Event {
        self.mode = Mode::Ambient;
        Event {
            kind: EventKind::ModeChanged,
            text: Mode::Ambient.as_str().to_string(),
            speaker: Speaker::Him,
            at_ms,
            context: self.log.recent(now_ms, Self::CONTEXT_WINDOW_MS),
        }
    }
}
