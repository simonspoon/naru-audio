//! The ambient state machine and gate routing against the scenarios of naru-ios
//! `NaruKitTests/AmbientMachineTests.swift` (naru task 2031). Time is in ms.

use naru_audio::ambient::machine::phrases::*;
use naru_audio::ambient::machine::*;
use naru_audio::ambient::{AmbientGate, GateKind, GateVerdict, NullGate};

const S: u64 = 1000;

struct Fixed(GateKind);
impl AmbientGate for Fixed {
    fn evaluate(&self, _: &str, _: &[LogLine]) -> GateVerdict {
        GateVerdict {
            kind: self.0,
            score: 0.9,
        }
    }
}

struct M {
    m: AmbientMachine,
    now: u64,
    gate: Box<dyn AmbientGate>,
}

impl M {
    fn new() -> Self {
        Self::with(Box::new(NullGate))
    }
    fn with(gate: Box<dyn AmbientGate>) -> Self {
        M {
            m: AmbientMachine::new(),
            now: 1_800_000_000_000,
            gate,
        }
    }
    fn adv(&mut self, ms: u64) {
        self.now += ms;
    }
    fn say(&mut self, who: Speaker, text: &str) -> AmbientOutput {
        self.m.ingest_with(&*self.gate, self.now, who, text, None)
    }
    fn kinds(out: &AmbientOutput) -> Vec<EventKind> {
        out.events.iter().map(|e| e.kind).collect()
    }
}

fn custom() -> Vec<Vec<String>> {
    wake_phrases(&["Jarvis", "Hey Computer", "Okay Big Bird", "hey", "  "].map(String::from))
}

fn rem(t: &str) -> Option<String> {
    wake_remainder(t, &[])
}

#[test]
fn transcript_log_expiry_window_and_capacity() {
    let mut log = TranscriptLog::default();
    log.append(0, Speaker::Him, "old");
    log.append(100 * S, Speaker::Other, "newer");
    let t = |l: Vec<LogLine>| l.into_iter().map(|x| x.text).collect::<Vec<_>>();
    assert_eq!(t(log.recent(100 * S, 120 * S)), ["old", "newer"]);
    assert_eq!(t(log.recent(125 * S, 120 * S)), ["newer"]);
    assert!(log.recent(245 * S, 120 * S).is_empty());

    let mut log = TranscriptLog::default();
    log.append(0, Speaker::Him, "a");
    log.append(45 * S, Speaker::Him, "b");
    assert_eq!(t(log.recent(45 * S, 30 * S)), ["b"]);

    let mut log = TranscriptLog::default();
    for i in 0..TranscriptLog::CAPACITY + 10 {
        log.append(0, Speaker::Him, &i.to_string());
    }
    let r = log.recent(0, 120 * S);
    assert_eq!(r.len(), TranscriptLog::CAPACITY);
    assert_eq!(r[0].text, "10");
}

#[test]
fn wake_matches_opening_naru() {
    assert_eq!(
        rem("Naru, what time is it?").as_deref(),
        Some("what time is it?")
    );
    assert_eq!(rem("NARU").as_deref(), Some(""));
    assert_eq!(rem("Hey Naru, remind me").as_deref(), Some("remind me"));
    assert_eq!(rem("okay naru").as_deref(), Some(""));
    for s in ["Neru,", "Niru,"] {
        assert_eq!(rem(&format!("{s} call Sam")).as_deref(), Some("call Sam"));
    }
    for t in [
        "I think Naru is great",
        "so naru should do it",
        "narrow road",
    ] {
        assert_eq!(rem(t), None, "{t}");
    }
}

#[test]
fn sound_alike_accepts_and_rejects() {
    for t in [
        "Hey Naro",
        "Hey narrow.",
        "Hey Laru",
        "Heineru",
        "Hey, Naro!",
        "OK Naro",
    ] {
        assert!(sound_alike(t, &[]).is_some(), "{t}");
    }
    for t in [
        "narrow road",
        "narrow",
        "Naro",
        "Hey there",
        "hey there Naru",
        "Hey Laura",
        "Hey Nora",
        "Hey Mary",
        "Narrow road ahead",
        "Hey narrow road",
        "Shade arrow",
        "Hello",
        "Hey Naru",
        "Heinrich",
    ] {
        assert!(sound_alike(t, &[]).is_none(), "{t}");
    }
}

#[test]
fn end_phrase() {
    for t in [
        "Thanks, that's all.",
        "thanks that's all",
        "Thank you, that's all!",
        "THANKS THAT\u{2019}S ALL",
    ] {
        assert!(is_end_phrase(t), "{t}");
    }
    assert!(!is_end_phrase("thanks for that"));
    assert!(!is_end_phrase("that's all"));
}

#[test]
fn custom_wake_words() {
    let c = custom();
    let v = |w: &[&str]| w.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(
        wake_phrases(&["Hey Jarvis", "jarvis", "Big Bird!", "Hey", "ok wake"].map(String::from)),
        vec![v(&["jarvis"]), v(&["big", "bird"])]
    );
    assert_eq!(
        c,
        vec![v(&["jarvis"]), v(&["computer"]), v(&["big", "bird"])]
    );
    assert_eq!(
        wake_remainder("Jarvis, lights", &c).as_deref(),
        Some("lights")
    );
    assert_eq!(
        wake_remainder("Hey Jarvis, lights", &c).as_deref(),
        Some("lights")
    );
    assert_eq!(wake_remainder("Hey computer", &c).as_deref(), Some(""));
    assert_eq!(
        wake_remainder("Big Bird, call Sam", &c).as_deref(),
        Some("call Sam")
    );
    assert_eq!(
        wake_remainder("okay big bird call Sam", &c).as_deref(),
        Some("call Sam")
    );
    assert!(is_bare_wake("Hey, Big Bird!", &c));
    assert!(is_bare_wake("Jarvis.", &c));
    assert!(!is_bare_wake("Jarvis lights", &c));
    assert!(!is_bare_wake("Big", &c));
    assert_eq!(wake_remainder("I asked Jarvis", &c), None);
    assert_eq!(wake_remainder("so hey Jarvis", &c), None);
    assert_eq!(wake_remainder("Jarvis lights", &[]), None);
    assert_eq!(
        wake_remainder("Hey Naru, remind me", &c).as_deref(),
        Some("remind me")
    );
    assert!(is_bare_wake("Naru", &c));
    assert_eq!(
        sound_alike("Hey Jarvas", &c).map(|x| x.0).as_deref(),
        Some("jarvas")
    );
    assert!(sound_alike("Hey big bard", &c).is_some());
    assert!(sound_alike("Hey Naro", &c).is_some());
    assert!(sound_alike("Hey Jarvas", &[]).is_none());
    for t in ["Hey Harvey", "Jervis", "Hey big", "Hey Jarvas lights"] {
        assert!(sound_alike(t, &c).is_none(), "{t}");
    }
    assert!(sound_alike("Hey Jarvis", &c).is_none());
    assert!(sound_alike("Hey big bird", &c).is_none());
}

#[test]
fn hyphenated_and_short_custom_words() {
    let bird = wake_phrases(&["Big Bird".to_string()]);
    assert_eq!(
        wake_remainder("Big-Bird, lights", &bird).as_deref(),
        Some("lights")
    );
    assert!(is_bare_wake("Hey big-bird!", &bird));
    let hyphen = wake_phrases(&["Big-Bird".to_string()]);
    assert_eq!(
        wake_remainder("big bird lights", &hyphen).as_deref(),
        Some("lights")
    );
    let max = wake_phrases(&["Max".to_string()]);
    assert!(sound_alike("Hey Mix", &max).is_none());
    assert_eq!(
        wake_remainder("Hey Max, lights", &max).as_deref(),
        Some("lights")
    );
}

#[test]
fn wake_flips_to_conversation_with_thirty_second_context() {
    let mut t = M::new();
    t.say(Speaker::Other, "too old");
    t.adv(40 * S);
    t.say(Speaker::Other, "did you see the game");
    t.adv(5 * S);
    t.say(Speaker::Him, "yeah it was close");
    t.adv(5 * S);
    let out = t.say(Speaker::Him, "Naru, what's the weather");
    assert_eq!(out.events.len(), 1);
    let w = &out.events[0];
    assert_eq!(w.kind, EventKind::Wake);
    assert_eq!(w.speaker, Speaker::Him);
    assert_eq!(w.text, "Naru, what's the weather");
    let ctx: Vec<_> = w.context.iter().map(|l| l.text.as_str()).collect();
    assert_eq!(ctx, ["did you see the game", "yeah it was close"]);
    assert_eq!(out.utterance.as_deref(), Some("what's the weather"));
    assert_eq!(t.m.mode(), Mode::Conversation);
}

#[test]
fn custom_wake_word_wakes() {
    let mut t = M::new();
    t.m.set_wake_words(&["Jarvis".to_string()]);
    let out = t.say(Speaker::Unknown, "Jarvis, call Sam");
    assert_eq!(M::kinds(&out), [EventKind::Wake]);
    assert_eq!(out.utterance.as_deref(), Some("call Sam"));
}

#[test]
fn other_voices_never_wake() {
    let mut t = M::new();
    assert_eq!(
        t.say(Speaker::Other, "Naru, do the thing"),
        AmbientOutput::default()
    );
    assert_eq!(t.m.mode(), Mode::Ambient);
    assert_eq!(
        M::new().say(Speaker::Other, "Naru, do X"),
        AmbientOutput::default()
    );
}

#[test]
fn conversation_sends_his_lines_and_keeps_others_as_context() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    t.adv(5 * S);
    let out = t.say(Speaker::Him, "add milk to the list");
    assert_eq!(out.utterance.as_deref(), Some("add milk to the list"));
    assert!(out.events.is_empty());
    assert_eq!(
        t.say(Speaker::Other, "we are out of eggs too"),
        AmbientOutput::default()
    );
    t.adv(11 * S);
    assert_eq!(t.say(Speaker::Unknown, "hm"), AmbientOutput::default());
    let now = t.now;
    let who: Vec<_> = t.m.log_lines(now).iter().map(|l| l.speaker).collect();
    assert_eq!(
        who,
        [Speaker::Him, Speaker::Him, Speaker::Other, Speaker::Unknown]
    );
}

#[test]
fn short_untagged_line_right_after_his_counts_as_his() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    t.adv(4 * S);
    assert_eq!(
        t.say(Speaker::Unknown, "yes").utterance.as_deref(),
        Some("yes")
    );
    t.adv(7 * S);
    assert_eq!(
        t.say(Speaker::Unknown, "hello over there"),
        AmbientOutput::default()
    );
    assert_eq!(t.say(Speaker::Other, "yes"), AmbientOutput::default());
}

#[test]
fn short_end_phrase_without_a_tag_returns_to_ambient() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru, what's the time");
    t.adv(3 * S);
    let out = t.say(Speaker::Unknown, "Thanks, that's all.");
    assert_eq!(M::kinds(&out), [EventKind::ModeChanged]);
    assert_eq!(out.utterance, None);
}

#[test]
fn unknown_opening_with_wake_word_wakes() {
    for p in [
        "Naru wake up",
        "Wake, Naru.",
        "Naroo, please wake up",
        "Nauru",
        "Naru what time is it",
    ] {
        assert_eq!(
            M::kinds(&M::new().say(Speaker::Unknown, p)),
            [EventKind::Wake],
            "{p}"
        );
    }
    for p in [
        "what time is it",
        "I think Naru is great",
        "wake up the kids",
    ] {
        assert_eq!(
            M::new().say(Speaker::Unknown, p),
            AmbientOutput::default(),
            "{p}"
        );
    }
}

#[test]
fn untagged_wake_blocked_while_the_agent_is_audible() {
    let mut t = M::new();
    let now = t.now;
    t.m.set_agent_speaking(now, true);
    assert_eq!(
        t.say(Speaker::Unknown, "Naru what time is it"),
        AmbientOutput::default()
    );
    let now = t.now;
    t.m.set_agent_speaking(now, false);
    t.adv(500);
    assert_eq!(
        t.say(Speaker::Unknown, "Naru what time is it"),
        AmbientOutput::default()
    );
    t.adv(S);
    assert_eq!(
        M::kinds(&t.say(Speaker::Unknown, "Naru what time is it")),
        [EventKind::Wake]
    );
}

#[test]
fn bare_wake_wakes_from_unknown_but_not_other() {
    for p in ["Naru.", "Hey, Naru!", "okay naru"] {
        let mut t = M::new();
        let out = t.say(Speaker::Unknown, p);
        assert_eq!(M::kinds(&out), [EventKind::Wake], "{p}");
        assert_eq!(out.utterance, None);
        t.adv(5 * S);
        assert_eq!(
            t.say(Speaker::Unknown, "what time is it")
                .utterance
                .as_deref(),
            Some("what time is it")
        );
        assert_eq!(M::new().say(Speaker::Other, p), AmbientOutput::default());
    }
    for p in ["tell Naru", "hey there Naru"] {
        assert_eq!(
            M::new().say(Speaker::Unknown, p),
            AmbientOutput::default(),
            "{p}"
        );
    }
}

#[test]
fn unknown_segments_are_log_only_while_the_agent_speaks_and_a_second_after() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    let now = t.now;
    t.m.set_agent_speaking(now, true);
    t.adv(2 * S);
    assert_eq!(
        t.say(Speaker::Unknown, "sure thing"),
        AmbientOutput::default()
    );
    let now = t.now;
    t.m.set_agent_speaking(now, false);
    t.adv(500);
    assert_eq!(
        t.say(Speaker::Unknown, "of course"),
        AmbientOutput::default()
    );
    t.adv(S);
    assert_eq!(
        t.say(Speaker::Unknown, "yes").utterance.as_deref(),
        Some("yes")
    );
    let now = t.now;
    t.m.set_agent_speaking(now, true);
    assert_eq!(
        t.say(Speaker::Him, "stop").utterance.as_deref(),
        Some("stop")
    );
}

#[test]
fn end_phrase_returns_to_ambient_and_is_not_sent() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    let out = t.say(Speaker::Him, "Thanks, that's all.");
    assert_eq!(out.utterance, None);
    assert_eq!(M::kinds(&out), [EventKind::ModeChanged]);
    assert_eq!(out.events[0].text, "ambient");
    assert_eq!(t.m.mode(), Mode::Ambient);
}

#[test]
fn returns_to_ambient_after_two_quiet_minutes() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    t.adv(100 * S);
    t.say(Speaker::Other, "someone else talking");
    t.adv(19 * S);
    assert!(t.m.tick(t.now).events.is_empty());
    t.adv(S);
    assert_eq!(M::kinds(&t.m.tick(t.now)), [EventKind::ModeChanged]);
    assert_eq!(t.m.mode(), Mode::Ambient);
}

#[test]
fn his_speech_resets_the_silence_timer() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    t.adv(100 * S);
    t.say(Speaker::Him, "one more thing");
    t.adv(100 * S);
    assert!(t.m.tick(t.now).events.is_empty());
}

#[test]
fn gate_verdicts_become_events_and_only_his_lines_reach_the_gate() {
    for (kind, expected) in [
        (GateKind::Idea, EventKind::IdeaFlagged),
        (GateKind::CanHelp, EventKind::CanHelp),
    ] {
        let mut t = M::with(Box::new(Fixed(kind)));
        t.say(Speaker::Other, "context line");
        let out = t.say(Speaker::Him, "we could cache the index");
        assert_eq!(M::kinds(&out), [expected]);
        assert_eq!(out.events[0].context[0].text, "context line");
        assert_eq!(out.utterance, None);
        let mut t = M::with(Box::new(Fixed(kind)));
        for who in [Speaker::Other, Speaker::Unknown] {
            assert_eq!(
                t.say(who, "we could cache the index"),
                AmbientOutput::default()
            );
        }
    }
    assert_eq!(
        M::new().say(Speaker::Him, "a brilliant idea about caching"),
        AmbientOutput::default()
    );
}

#[test]
fn event_wire_names() {
    assert_eq!(EventKind::IdeaFlagged.as_str(), "idea_flagged");
    assert_eq!(EventKind::CanHelp.as_str(), "can_help");
    assert_eq!(EventKind::ModeChanged.as_str(), "mode_changed");
}

#[test]
fn sound_alike_wakes_untagged_and_logs_the_similarity() {
    for (text, sim) in [
        ("Hey Naro", Some(0.13f32)),
        ("Hey narrow.", None),
        ("Hey Laru", Some(0.5)),
        ("Heineru", Some(0.2)),
    ] {
        let mut t = M::new();
        let out =
            t.m.ingest_with(&NullGate, t.now, Speaker::Unknown, text, sim);
        assert_eq!(M::kinds(&out), [EventKind::Wake], "{text}");
        assert_eq!(
            out.wake_candidates
                .iter()
                .map(|c| c.accepted)
                .collect::<Vec<_>>(),
            [true]
        );
        assert_eq!(out.wake_candidates[0].similarity, sim);
        assert_eq!(t.m.mode(), Mode::Conversation);
    }
}

#[test]
fn sound_alike_from_another_voice_is_logged_not_taken() {
    let mut t = M::new();
    let out =
        t.m.ingest_with(&NullGate, t.now, Speaker::Other, "Hey Naro", Some(0.9));
    assert!(out.events.is_empty());
    assert_eq!(
        out.wake_candidates
            .iter()
            .map(|c| c.accepted)
            .collect::<Vec<_>>(),
        [false]
    );
    assert_eq!(t.m.mode(), Mode::Ambient);
}

#[test]
fn ordinary_short_talk_and_exact_wakes_never_produce_a_candidate() {
    for text in [
        "narrow road",
        "Hey there",
        "hey there Naru",
        "Hey Laura",
        "yes",
        "Thanks, that's all.",
    ] {
        let out = M::new().say(Speaker::Unknown, text);
        assert_eq!(out, AmbientOutput::default(), "{text}");
    }
    for (who, text) in [
        (Speaker::Unknown, "Hey Naru"),
        (Speaker::Unknown, "Naru."),
        (Speaker::Him, "Naru, what time is it"),
    ] {
        let out = M::new().say(who, text);
        assert_eq!(M::kinds(&out), [EventKind::Wake]);
        assert!(out.wake_candidates.is_empty());
    }
    let out = M::new().say(Speaker::Him, "narrow road");
    assert!(out.events.is_empty() && out.wake_candidates.is_empty());
}

#[test]
fn sound_alike_tagged_him_wakes_and_in_conversation_is_just_his_line() {
    let out = M::new().say(Speaker::Him, "Hey Naro");
    assert_eq!(M::kinds(&out), [EventKind::Wake]);
    assert_eq!(
        out.wake_candidates
            .iter()
            .map(|c| c.accepted)
            .collect::<Vec<_>>(),
        [true]
    );
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    t.adv(2 * S);
    let out = t.say(Speaker::Him, "Hey Naro");
    assert!(out.events.is_empty() && out.wake_candidates.is_empty());
    assert_eq!(out.utterance.as_deref(), Some("Hey Naro"));
}

#[test]
fn end_conversation_makes_the_next_bare_wake_fire() {
    let mut t = M::new();
    t.say(Speaker::Him, "Naru");
    t.adv(5 * S);
    assert!(t.say(Speaker::Unknown, "Hey Naru").events.is_empty());
    let out = t.m.end_conversation(t.now);
    assert_eq!(M::kinds(&out), [EventKind::ModeChanged]);
    assert_eq!(out.events[0].text, "ambient");
    t.adv(2 * S);
    assert_eq!(
        M::kinds(&t.say(Speaker::Unknown, "Hey Naru")),
        [EventKind::Wake]
    );
    assert_eq!(
        M::kinds(&t.m.end_conversation(t.now)),
        [EventKind::ModeChanged]
    );
    assert_eq!(t.m.end_conversation(t.now), AmbientOutput::default());
}
