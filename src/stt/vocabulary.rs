//! The hotwords vocabulary, ported from auris `src/vocabulary.rs`
//! (d8644d6): parses `term[ :boost]` lines (the auris `--vocabulary-file`
//! syntax, §2.2 `hotwords`) into the per-stream hotwords string, and
//! [`looks_manufactured`], the prefilter of the manufactured-vocabulary
//! guard.

use std::fmt;

/// Terms past this many are dropped in order; the order is the caller's
/// priority.
pub const MAX_TERMS: usize = 128;

/// Upper end of the accepted `:boost` range, `(0.0, 8.0]`. Past it auris
/// measured the transcript coming apart.
const MAX_BOOST: f32 = 8.0;

/// A longer term is rejected as "not a name".
const MAX_TERM_BYTES: usize = 64;

/// One entry. `boost` overrides the global 3.0
/// ([`super::engine::EngineConfig::hotwords_score`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Term {
    pub text: String,
    pub boost: Option<f32>,
}

/// Each variant carries the 1-based line number and the offending line
/// (comment stripped).
#[derive(Debug)]
pub enum VocabularyError {
    /// `/` is the per-stream phrase separator and has no escape.
    SlashInTerm(usize, String),
    /// sherpa-onnx's `std::stof` crashes on a malformed boost.
    NonNumericBoost(usize, String),
    /// `term:boost`, which sherpa-onnx silently drops as one unknown token.
    NoSpaceBeforeBoost(usize, String),
    /// `mesa : 3.0` or `mesa :3.0 :4.0`.
    StrayBoostToken(usize, String),
    BoostOutOfRange(usize, String, f32),
    EmptyTerm(usize),
    TermTooLong(usize, String),
    ControlCharacter(usize, String),
}

impl fmt::Display for VocabularyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VocabularyError::SlashInTerm(line, content) => write!(
                f,
                "line {line}: term contains '/', the per-stream phrase separator: {content:?}"
            ),
            VocabularyError::NonNumericBoost(line, content) => {
                write!(f, "line {line}: boost is not a number: {content:?}")
            }
            VocabularyError::NoSpaceBeforeBoost(line, content) => write!(
                f,
                "line {line}: \"term:boost\" needs a space before the colon \
                 (\"term :boost\"), or sherpa-onnx silently drops it: {content:?}"
            ),
            VocabularyError::StrayBoostToken(line, content) => write!(
                f,
                "line {line}: a boost token appears where a term word was expected \
                 (a colon separated from its value by whitespace, or a second boost): \
                 {content:?}"
            ),
            VocabularyError::BoostOutOfRange(line, content, boost) => write!(
                f,
                "line {line}: boost {boost} is outside (0.0, 8.0]: {content:?}"
            ),
            VocabularyError::EmptyTerm(line) => write!(f, "line {line}: term is empty"),
            VocabularyError::TermTooLong(line, term) => write!(
                f,
                "line {line}: term is {} bytes, over the 64-byte limit: {term:?}",
                term.len()
            ),
            VocabularyError::ControlCharacter(line, content) => write!(
                f,
                "line {line}: term contains a control character: {content:?}"
            ),
        }
    }
}

impl std::error::Error for VocabularyError {}

/// A parsed, validated, capped vocabulary.
#[derive(Debug)]
pub struct Vocabulary {
    pub terms: Vec<Term>,
    /// How many terms past [`MAX_TERMS`] were dropped; a warning, not an
    /// error.
    pub terms_dropped: usize,
}

impl Vocabulary {
    /// One entry per line, `#` to end of line a comment, blank lines ignored.
    pub fn parse(contents: &str) -> Result<Vocabulary, VocabularyError> {
        let mut terms = Vec::new();
        for (i, raw_line) in contents.lines().enumerate() {
            let line_no = i + 1;
            if let Some(term) = parse_line(line_no, raw_line)? {
                terms.push(term);
            }
        }

        let terms_dropped = terms.len().saturating_sub(MAX_TERMS);
        terms.truncate(MAX_TERMS);
        Ok(Vocabulary {
            terms,
            terms_dropped,
        })
    }

    /// Terms joined with `/`, each `term :boost` or bare. Empty segments are
    /// safe (sherpa-onnx drops them).
    pub fn hotwords_string(&self) -> String {
        self.terms
            .iter()
            .map(|term| match term.boost {
                // `{:?}` keeps 3.0 as "3.0", the file format's own spelling.
                Some(boost) => format!("{} :{boost:?}", term.text),
                None => term.text.clone(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }
}

/// auris measured (task 970) the longest run of consecutive vocabulary words
/// in 88 real-speech transcripts as 1, and every confirmed non-speech
/// hallucination at 3 or more. A false trigger only costs a confirming
/// decode, so the more sensitive 2 is used.
const MANUFACTURED_RUN: usize = 2;

/// A prefilter, not a decision: true means the biased transcript is worth a
/// second, unbiased decode of the same audio. The caller discards the biased
/// transcript only if that decode is empty too, so a false positive never
/// changes the output.
///
/// True when `text` has [`MANUFACTURED_RUN`] or more consecutive words that
/// are each a word of some term, compared case-insensitively with ASCII
/// punctuation trimmed. The run may mix terms ("mesa khora khora").
pub fn looks_manufactured(text: &str, terms: &[Term]) -> bool {
    let vocab_words: std::collections::HashSet<String> = terms
        .iter()
        .flat_map(|t| t.text.split_whitespace())
        .map(|w| w.to_lowercase())
        .collect();
    if vocab_words.is_empty() {
        return false;
    }

    let mut run = 0usize;
    for word in text.split_whitespace() {
        let stripped = word.trim_matches(|c: char| c.is_ascii_punctuation());
        if vocab_words.contains(&stripped.to_lowercase()) {
            run += 1;
            if run >= MANUFACTURED_RUN {
                return true;
            }
        } else {
            run = 0;
        }
    }
    false
}

/// `None` for a blank or comment-only line.
fn parse_line(line_no: usize, raw_line: &str) -> Result<Option<Term>, VocabularyError> {
    let content = match raw_line.find('#') {
        Some(i) => &raw_line[..i],
        None => raw_line,
    };
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return Ok(None);
    }

    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let last = *tokens.last().expect("non-empty trimmed line has a token");

    // The boost is its own token, ":BOOST".
    let (term_tokens, boost) = match last.strip_prefix(':') {
        Some(rest) => {
            let value: f32 = rest
                .parse()
                .map_err(|_| VocabularyError::NonNumericBoost(line_no, trimmed.to_string()))?;
            (&tokens[..tokens.len() - 1], Some(value))
        }
        None => (&tokens[..], None),
    };

    // A leading ':' is a stray boost token; a ':' mid-word is the glued
    // `term:boost` spelling.
    if term_tokens.iter().any(|t| t.starts_with(':')) {
        return Err(VocabularyError::StrayBoostToken(
            line_no,
            trimmed.to_string(),
        ));
    }
    if term_tokens.iter().any(|t| t.contains(':')) {
        return Err(VocabularyError::NoSpaceBeforeBoost(
            line_no,
            trimmed.to_string(),
        ));
    }

    let term = term_tokens.join(" ");
    if term.is_empty() {
        return Err(VocabularyError::EmptyTerm(line_no));
    }
    if term.contains('/') {
        return Err(VocabularyError::SlashInTerm(line_no, trimmed.to_string()));
    }
    if term.len() > MAX_TERM_BYTES {
        return Err(VocabularyError::TermTooLong(line_no, term));
    }
    if term.chars().any(|c| c.is_control()) {
        return Err(VocabularyError::ControlCharacter(
            line_no,
            trimmed.to_string(),
        ));
    }
    if let Some(boost) = boost
        && !(boost > 0.0 && boost <= MAX_BOOST)
    {
        return Err(VocabularyError::BoostOutOfRange(
            line_no,
            trimmed.to_string(),
            boost,
        ));
    }

    Ok(Some(Term { text: term, boost }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(v: &Vocabulary) -> Vec<(&str, Option<f32>)> {
        v.terms.iter().map(|t| (t.text.as_str(), t.boost)).collect()
    }

    #[test]
    fn empty_file_is_an_empty_vocabulary() {
        let v = Vocabulary::parse("").expect("parse");
        assert!(v.terms.is_empty());
        assert_eq!(v.terms_dropped, 0);
    }

    #[test]
    fn comments_and_blank_lines_are_ignored() {
        let v = Vocabulary::parse("# a comment\n\nmesa :3.0\n   \n# trailing\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", Some(3.0))]);
    }

    #[test]
    fn bare_term_has_no_boost() {
        let v = Vocabulary::parse("mesa\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", None)]);
    }

    #[test]
    fn term_with_boost() {
        let v = Vocabulary::parse("khora :6.5\n").expect("parse");
        assert_eq!(terms(&v), vec![("khora", Some(6.5))]);
    }

    #[test]
    fn multi_word_phrase() {
        let v = Vocabulary::parse("wire up :3.0\n").expect("parse");
        assert_eq!(terms(&v), vec![("wire up", Some(3.0))]);
    }

    #[test]
    fn inline_comment_after_a_term_is_stripped() {
        let v = Vocabulary::parse("mesa :3.0 # our own product\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", Some(3.0))]);
    }

    #[test]
    fn cap_truncates_to_first_128_in_file_order_and_reports_it() {
        let mut file = String::new();
        for i in 0..150 {
            file.push_str(&format!("term{i}\n"));
        }
        let v = Vocabulary::parse(&file).expect("parse");
        assert_eq!(v.terms.len(), MAX_TERMS);
        assert_eq!(v.terms_dropped, 22);
        assert_eq!(v.terms[0].text, "term0");
        assert_eq!(v.terms[127].text, "term127");
    }

    #[test]
    fn rejects_slash_in_term() {
        let err = Vocabulary::parse("mesa/auris :3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::SlashInTerm(1, _)));
    }

    #[test]
    fn rejects_non_numeric_boost() {
        let err = Vocabulary::parse("mesa :abc\n").unwrap_err();
        assert!(matches!(err, VocabularyError::NonNumericBoost(1, _)));
    }

    #[test]
    fn rejects_term_colon_boost_with_no_space() {
        let err = Vocabulary::parse("khora:6.5\n").unwrap_err();
        assert!(matches!(err, VocabularyError::NoSpaceBeforeBoost(1, _)));
    }

    #[test]
    fn rejects_space_between_colon_and_boost_value() {
        let err = Vocabulary::parse("mesa : 3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::StrayBoostToken(1, _)));
    }

    #[test]
    fn rejects_a_second_boost_token() {
        let err = Vocabulary::parse("mesa :3.0 :4.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::StrayBoostToken(1, _)));
    }

    #[test]
    fn rejects_boost_of_zero() {
        let err = Vocabulary::parse("mesa :0.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::BoostOutOfRange(1, _, _)));
    }

    #[test]
    fn rejects_boost_above_the_cap() {
        let err = Vocabulary::parse("mesa :8.1\n").unwrap_err();
        assert!(matches!(err, VocabularyError::BoostOutOfRange(1, _, _)));
    }

    #[test]
    fn accepts_boost_at_the_upper_bound() {
        let v = Vocabulary::parse("mesa :8.0\n").expect("parse");
        assert_eq!(terms(&v), vec![("mesa", Some(8.0))]);
    }

    #[test]
    fn rejects_empty_term() {
        let err = Vocabulary::parse(":3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::EmptyTerm(1)));
    }

    #[test]
    fn rejects_term_over_64_bytes() {
        let long_term = "a".repeat(65);
        let err = Vocabulary::parse(&long_term).unwrap_err();
        assert!(matches!(err, VocabularyError::TermTooLong(1, _)));
    }

    #[test]
    fn accepts_term_at_64_bytes() {
        let term = "a".repeat(64);
        let v = Vocabulary::parse(&term).expect("parse");
        assert_eq!(v.terms[0].text.len(), 64);
    }

    #[test]
    fn rejects_control_character_in_term() {
        let err = Vocabulary::parse("me\u{0007}sa :3.0\n").unwrap_err();
        assert!(matches!(err, VocabularyError::ControlCharacter(1, _)));
    }

    /// The production term list used in mesa task 970's measurement table
    /// (auris `bench/fixtures/hotwords.txt`, inlined here so this test doesn't
    /// depend on the model spike being present — same convention as
    /// `hotwords_string_matches_the_production_fixture_shape` above).
    fn task_970_vocabulary() -> Vocabulary {
        let file = "mesa :3.0\nauris :3.0\nkhora :6.5\nqorvex :5.0\nhelios :3.0\nkokoro :3.0\n";
        Vocabulary::parse(file).expect("parse")
    }

    #[test]
    fn measured_hallucinations_are_flagged() {
        let v = task_970_vocabulary();
        // nonspeech-transient.wav @3.0 (mesa task 970's repro).
        assert!(looks_manufactured(
            "The vegetable mesa mesa khora khora khora mesa khan mesa q",
            &v.terms
        ));
        // nonspeech-white.wav @3.0 — the shortest run measured in any
        // confirmed hallucination, at 3, comfortably clear of the
        // threshold: MANUFACTURED_RUN is set from the real-speech ceiling
        // of 1, not from this number.
        assert!(looks_manufactured("mesa mesa mesa", &v.terms));
        // nonspeech-rumble.wav @4.0.
        assert!(looks_manufactured(
            "khora khora khora khora khora khora khora khora",
            &v.terms
        ));
    }

    #[test]
    fn real_speech_transcripts_are_not_flagged() {
        let v = task_970_vocabulary();
        assert!(!looks_manufactured(
            "open the daily notes and add a line about the meeting",
            &v.terms
        ));
        assert!(!looks_manufactured("hey qorvex hey helios", &v.terms));
        assert!(!looks_manufactured(
            "open the daily mesa and add a line about the meeting",
            &v.terms
        ));
    }

    #[test]
    fn empty_text_is_not_flagged() {
        let v = task_970_vocabulary();
        assert!(!looks_manufactured("", &v.terms));
    }

    #[test]
    fn a_run_split_by_a_non_vocabulary_word_does_not_count_as_one_run() {
        // Singleton vocabulary words, each separated by an ordinary word,
        // never accumulate into a run — a non-vocabulary word resets it.
        let v = task_970_vocabulary();
        assert!(!looks_manufactured("mesa the khora the mesa", &v.terms));
        // The contrast: with nothing between them, two adjacent vocabulary
        // words do form a run of MANUFACTURED_RUN (2) and are flagged.
        assert!(looks_manufactured("mesa khora", &v.terms));
    }

    /// The exact boundary [`MANUFACTURED_RUN`] encodes: a single vocabulary
    /// word surrounded by ordinary words is never enough on its own —
    /// that's the shape every real transcript in the corpus takes — but two
    /// of them adjacent, with nothing between, is.
    #[test]
    fn a_single_vocabulary_word_among_ordinary_words_is_not_enough_but_two_adjacent_are() {
        let v = task_970_vocabulary();
        assert!(!looks_manufactured("open mesa and add a note", &v.terms));
        assert!(looks_manufactured(
            "open mesa khora and add a note",
            &v.terms
        ));
    }

    #[test]
    fn case_and_trailing_punctuation_are_handled() {
        // "Mesa," and "MESA" are already two adjacent vocabulary words once
        // case and trailing punctuation are normalised, so this flags on
        // those two alone — khora is not needed to reach the run.
        let v = task_970_vocabulary();
        assert!(looks_manufactured("Mesa, MESA khora.", &v.terms));
    }

    #[test]
    fn hotwords_string_matches_the_production_fixture_shape() {
        // auris bench/fixtures/hotwords.txt, inlined here so this test doesn't
        // depend on the model spike being present.
        let file = "mesa :3.0\nauris :3.0\nkhora :6.5\nqorvex :5.0\nhelios :3.0\nkokoro :3.0\n";
        let v = Vocabulary::parse(file).expect("parse");
        assert_eq!(
            v.hotwords_string(),
            "mesa :3.0/auris :3.0/khora :6.5/qorvex :5.0/helios :3.0/kokoro :3.0"
        );
    }

    #[test]
    fn hotwords_string_bare_term_has_no_colon() {
        let v = Vocabulary::parse("mesa\nauris :3.0\n").expect("parse");
        assert_eq!(v.hotwords_string(), "mesa/auris :3.0");
    }
}
