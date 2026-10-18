//! The live-mode state machine, extracted from naru-mac `NativeLive.swift` (naru task 2027).
//!
//! Spec: `docs/live-mode-spec.md`. This module is pure: no I/O, no async, no clock reads. Every
//! time-based rule takes the instant as a millisecond argument (`now_ms`, any monotonic origin), and
//! every method that in Swift would kick off work (cut the voice, start the engine, claim the speaker)
//! returns a description of that work instead. Nothing here is wired to a socket or route yet (the speaker guard's scoring is `crate::speaker`), and
//! naru-mac keeps its own copy until the switch task.

/// The page-visible mode, derived from the inputs. Pause and the speaker mute are overlays, not modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// No live session, or this client has not joined it: nothing is captured, nothing is spoken.
    Idle,
    /// Live and joined: capture is open and replies are spoken.
    Live,
    /// Live and joined with the microphone muted: capture is down (a hold keeps the input), replies still play.
    Muted,
    /// The session stays open but capture and the voice are released to ambient listening.
    Ambient,
}

/// How long the mic is held off after another client's reply (seconds in Swift: `otherVoiceHoldSeconds`).
pub const OTHER_VOICE_HOLD_MS: u64 = 1000;
/// The muted input hold's engine start is "settling" this long after it finished.
pub const HOLD_SETTLE_MS: u64 = 1000;
/// A system report within this long of a default-input change is the device's, not the person's.
pub const DEVICE_CHANGE_MS: u64 = 2000;
/// A deferred system unmute fires anyway when no capture frame arrives within this long.
pub const UNMUTE_FALLBACK_MS: u64 = 1000;

/// The live pipeline (session, speaker claim, mute) is wanted while live && joined && not ambient.
/// Pause does not enter into it: a pause keeps all of that and only silences it.
pub fn wants_active(live: bool, joined: bool, ambient: bool) -> bool {
    live && joined && !ambient
}

/// Speak a reply, or (speaker muted) just mark it played.
pub fn should_speak(speech_muted: bool) -> bool {
    !speech_muted
}

/// A reply cut by the speaker mute counts as played, not replayed; otherwise only a full play does.
pub fn counts_as_played(played: bool, cut_by_mute: bool) -> bool {
    played || cut_by_mute
}

/// What a change of pause state does; `None` from [`pause_action`] when nothing changes.
/// The fate of a finished segment after the speaker guard ([`LiveState::segment_fate`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentFate {
    Admit,
    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PauseAction {
    /// Pausing: cut the sounding reply (unstamped), stop capture and the run loop, drop held audio.
    pub cut: bool,
    /// Pausing: the turn that was sounding, to hand back for replay on resume.
    pub remember: Option<i64>,
    /// Resuming: the turn to release in the ledger so it is spoken again from its start.
    pub release: Option<i64>,
    /// Resuming: take the voice back.
    pub reclaim: bool,
}

pub fn pause_action(
    paused: bool,
    applied: bool,
    speaking: Option<i64>,
    remembered: Option<i64>,
) -> Option<PauseAction> {
    if paused == applied {
        return None;
    }
    if paused {
        Some(PauseAction {
            cut: true,
            remember: speaking,
            ..Default::default()
        })
    } else {
        Some(PauseAction {
            release: remembered,
            reclaim: true,
            ..Default::default()
        })
    }
}

/// Admit captured speech: pipeline wanted, not paused, no turn playing, no other voice.
pub fn admits(active: bool, paused: bool, speaking: bool, other_voice: bool) -> bool {
    active && !paused && !speaking && !other_voice
}

/// Whether a finished segment goes on to the speaker guard. Pipeline and pause are judged at the end;
/// "a turn is playing" and "another client holds the voice" are also judged at the speech's start, so a
/// reply starting while the person is still talking does not clip their utterance.
pub fn admits_segment(
    open_at_start: bool,
    active: bool,
    paused: bool,
    speaking: bool,
    other_voice: bool,
) -> bool {
    if !active || paused {
        return false;
    }
    open_at_start || (!speaking && !other_voice)
}

/// Whether another client's voice is in the way. `other_speaker_has_unplayed`: the speaker claim names
/// another client and an assistant turn with text is unplayed. Returns the verdict and the new hold
/// deadline (extended to `now + 1s` each time the first condition holds).
pub fn other_voice(other_speaker_has_unplayed: bool, until_ms: u64, now_ms: u64) -> (bool, u64) {
    if other_speaker_has_unplayed {
        (true, now_ms + OTHER_VOICE_HOLD_MS)
    } else {
        (now_ms < until_ms, until_ms)
    }
}

/// Which system "unmute" reports are the app's own engine starting rather than the person.
pub fn system_unmute_swallows(
    should_mute: bool,
    muted: bool,
    speaking: bool,
    cold_cue_recent: bool,
    hold_settling: bool,
    device_changing: bool,
) -> bool {
    if device_changing && should_mute != muted {
        return true;
    }
    !should_mute && muted && (speaking || cold_cue_recent || hold_settling)
}

/// What `bridge_state` / `apply_capture` decided; the caller carries it out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CaptureChange {
    pub started: bool,
    pub stopped: bool,
    /// The pause action to carry out (see [`PauseAction`]).
    pub pause: Option<PauseAction>,
}

/// What `set_ambient` decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AmbientChange {
    /// Ignored (no live session, or no change); the caller only re-notifies.
    Ignored,
    /// Entering ambient while active: cut the open utterance, send it, give the queue a few seconds,
    /// then capture goes down.
    EnterAfterSending,
    /// Anything else: capture is reconciled at once.
    Applied(CaptureChange),
}

/// What `set_muted` decided.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MuteChange {
    /// Nothing changed (same value).
    pub unchanged: bool,
    /// Muting: cut the open utterance, post it with what was held, reset the idle clock.
    pub cut_and_flush: bool,
    /// Unmuting while active and not paused: claim the voice.
    pub claim: bool,
    /// Set the system input mute on (muting with the system mute armed and out of step).
    pub system_mute: bool,
    /// Unmuting: the system unmute is deferred until capture delivers a frame.
    pub system_unmute_deferred: bool,
    /// The fallback timer must be armed now, with this generation (capture already runs).
    pub arm_fallback: Option<u64>,
}

/// The whole input/overlay state. Fields mirror `NativeLive`; the clock is always passed in.
#[derive(Debug, Clone, Default)]
pub struct LiveState {
    pub live: bool,
    pub joined: bool,
    pub paused: bool,
    pub speech_muted: bool,
    pub muted: bool,
    pub ambient: bool,
    /// Leaving ambient: the page keeps reading ambient=true until the voice claim lands.
    pub leaving_ambient: bool,
    /// The pipeline is wanted (and started).
    pub active: bool,
    /// The pause has been carried out on the pipeline.
    pub pause_applied: bool,
    pub speaking_turn: Option<i64>,
    /// The turn the speaker mute cut; recorded at the cut so a quick unmute cannot lose the verdict.
    pub speech_cut_turn: Option<i64>,
    /// The turn that was sounding when the pause cut it.
    pub paused_turn: Option<i64>,
    pub vad_speaking: bool,
    /// Whether the gate was open when the current/last utterance began.
    pub speech_started_open: bool,
    pub system_mute_armed: bool,
    pub system_unmute_pending: bool,
    pub frames_expected: bool,
    pub unmute_fallback_generation: u64,
    pub hold_starting: bool,
    pub hold_started_at_ms: Option<u64>,
    pub device_changed_at_ms: Option<u64>,
}

impl LiveState {
    pub fn new() -> Self {
        Self {
            speech_started_open: true,
            ..Default::default()
        }
    }

    /// Derived mode (see the spec's mode table).
    pub fn mode(&self) -> Mode {
        if !self.live {
            Mode::Idle
        } else if self.ambient {
            Mode::Ambient
        } else if !self.joined {
            Mode::Idle
        } else if self.muted {
            Mode::Muted
        } else {
            Mode::Live
        }
    }

    /// What the page is told: still ambient until the voice claim after leaving has landed.
    pub fn page_ambient(&self) -> bool {
        self.ambient || self.leaving_ambient
    }

    pub fn can_admit(&self, other_voice: bool) -> bool {
        admits(
            self.active,
            self.paused,
            self.speaking_turn.is_some(),
            other_voice,
        )
    }

    pub fn segment_admitted(&self, other_voice: bool) -> bool {
        admits_segment(
            self.speech_started_open,
            self.active,
            self.paused,
            self.speaking_turn.is_some(),
            other_voice,
        )
    }

    /// A segment is dropped after `admits_segment` too when the mic is muted.
    pub fn segment_goes_to_guard(&self, other_voice: bool) -> bool {
        self.segment_admitted(other_voice) && !self.muted
    }

    /// What happens to a finished segment once the speaker guard has scored
    /// it (`crate::speaker::Verdict::is_him`): it must reach the guard at all
    /// ([`Self::segment_goes_to_guard`]) and then pass it. The guard fails
    /// open, so an unscored segment arrives here with `is_him = true`.
    pub fn segment_fate(&self, other_voice: bool, is_him: bool) -> SegmentFate {
        if self.segment_goes_to_guard(other_voice) && is_him {
            SegmentFate::Admit
        } else {
            SegmentFate::Drop
        }
    }

    /// The VAD's speaking flag changed. On a rising edge the gate's state at the start is recorded.
    pub fn vad_changed(&mut self, speaking: bool, other_voice: bool) {
        if speaking && !self.vad_speaking {
            self.speech_started_open = self.can_admit(other_voice) && !self.muted;
        }
        self.vad_speaking = speaking;
    }

    /// The microphone is open for the idle (auto-send) clock.
    pub fn mic_open(&self, other_voice: bool) -> bool {
        self.active && !self.paused && !self.muted && self.speaking_turn.is_none() && !other_voice
    }

    /// The page's `state` message. Returns whether a speaker-mute cut is wanted and the capture change.
    pub fn bridge_state(
        &mut self,
        live: bool,
        joined: bool,
        paused: bool,
        speech_muted: bool,
    ) -> (bool, CaptureChange) {
        self.live = live;
        self.joined = joined;
        self.paused = paused;
        let cut = self.set_speech_muted(speech_muted);
        if !live && (self.ambient || self.leaving_ambient) {
            self.ambient = false;
            self.leaving_ambient = false;
        }
        (cut, self.apply_capture())
    }

    /// Returns true when the sounding reply must be cut now (the run loop then stamps it played).
    pub fn set_speech_muted(&mut self, m: bool) -> bool {
        if m == self.speech_muted {
            return false;
        }
        self.speech_muted = m;
        if m && self.active && self.speaking_turn.is_some() {
            self.speech_cut_turn = self.speaking_turn;
            return true;
        }
        false
    }

    /// Ambient on or off; only meaningful while a session is live.
    pub fn set_ambient(&mut self, on: bool) -> AmbientChange {
        if !self.live || on == self.ambient {
            return AmbientChange::Ignored;
        }
        self.ambient = on;
        self.leaving_ambient = !on;
        if on && self.active {
            AmbientChange::EnterAfterSending
        } else {
            AmbientChange::Applied(self.apply_capture())
        }
    }

    /// Capture runs only while live && joined && not paused && not ambient (pause is reconciled after).
    pub fn apply_capture(&mut self) -> CaptureChange {
        let mut change = CaptureChange::default();
        if wants_active(self.live, self.joined, self.ambient) {
            if !self.active {
                // A pause that arrives with the join is already in force: nothing is cut.
                if self.paused {
                    self.pause_applied = true;
                }
                self.active = true;
                change.started = true;
            }
        } else if self.active {
            self.stop();
            change.stopped = true;
        }
        if !wants_active(self.live, self.joined, self.ambient) {
            self.finish_leaving_ambient();
        }
        change.pause = self.reconcile_pause();
        change
    }

    /// Teardown: a rejoin must not be silently muted.
    fn stop(&mut self) {
        self.active = false;
        self.pause_applied = false;
        self.muted = false;
        self.vad_speaking = false;
        self.speaking_turn = None;
        self.system_unmute_pending = false;
        self.frames_expected = false;
        self.system_mute_armed = false;
    }

    /// Pause keeps `active` (ambient is never handed the mic); it only silences the pipeline.
    pub fn reconcile_pause(&mut self) -> Option<PauseAction> {
        if !self.active {
            self.pause_applied = false;
            return None;
        }
        let action = pause_action(
            self.paused,
            self.pause_applied,
            self.speaking_turn,
            self.paused_turn,
        )?;
        self.pause_applied = self.paused;
        if action.cut {
            self.paused_turn = action.remember;
            self.speaking_turn = None;
            self.vad_speaking = false;
        } else {
            self.paused_turn = None;
        }
        Some(action)
    }

    /// The page learns ambient=false only once the voice claim landed (or failed). True when it flipped.
    pub fn finish_leaving_ambient(&mut self) -> bool {
        if self.leaving_ambient && !self.ambient {
            self.leaving_ambient = false;
            return true;
        }
        false
    }

    /// The speaker claim attempt's outcome gate: `active` false or paused only finishes the leave.
    /// Returns true when the claim should actually be made.
    pub fn claim_wanted(&mut self) -> bool {
        if !self.active {
            self.leaving_ambient = false;
            return false;
        }
        if self.paused {
            self.finish_leaving_ambient();
            return false;
        }
        true
    }

    /// The page's `mic` message, the media keys, the menu and the hotkey. `system_input_muted` is the
    /// system's current input-mute flag.
    pub fn set_muted(&mut self, m: bool, system_input_muted: bool) -> MuteChange {
        let mut c = MuteChange::default();
        if m == self.muted {
            c.unchanged = true;
            return c;
        }
        self.muted = m;
        // A mute (or a re-unmute) supersedes any unmute still waiting for a frame.
        self.system_unmute_pending = false;
        self.frames_expected = false;
        self.unmute_fallback_generation += 1;
        if self.system_mute_armed && system_input_muted != m {
            if m {
                c.system_mute = true;
            } else {
                self.system_unmute_pending = true;
                self.frames_expected = self.active && self.speaking_turn.is_some();
                c.system_unmute_deferred = true;
                if self.frames_expected {
                    c.arm_fallback = Some(self.arm_unmute_fallback());
                }
            }
        }
        if m {
            c.cut_and_flush = true;
        } else if self.active && !self.paused {
            c.claim = true;
        }
        c
    }

    /// Capture has started for the current unmute: the next level is a real frame.
    /// Returns the fallback generation to arm, if any.
    pub fn capture_started_after_unmute(&mut self) -> Option<u64> {
        if self.system_unmute_pending && !self.muted {
            self.frames_expected = true;
            return Some(self.arm_unmute_fallback());
        }
        None
    }

    fn arm_unmute_fallback(&mut self) -> u64 {
        self.unmute_fallback_generation += 1;
        self.unmute_fallback_generation
    }

    /// The first capture frame after an unmute; true when the system unmute should fire now.
    pub fn first_frame(&mut self) -> bool {
        if self.active && self.system_unmute_pending && self.frames_expected && !self.muted {
            self.system_unmute_pending = false;
            self.frames_expected = false;
            return true;
        }
        false
    }

    /// The fallback timer for `generation` elapsed (after [`UNMUTE_FALLBACK_MS`]); true when the system
    /// unmute should fire anyway.
    pub fn fallback_fires(&mut self, generation: u64) -> bool {
        if generation == self.unmute_fallback_generation
            && self.system_unmute_pending
            && self.frames_expected
            && !self.muted
            && self.active
            && self.system_mute_armed
        {
            self.system_unmute_pending = false;
            self.frames_expected = false;
            return true;
        }
        false
    }

    pub fn begin_hold_start(&mut self) {
        self.hold_starting = true;
    }

    pub fn end_hold_start(&mut self, now_ms: u64) {
        self.hold_starting = false;
        self.hold_started_at_ms = Some(now_ms);
    }

    /// The muted hold's engine is starting, or did within the last second.
    pub fn hold_settling(&self, now_ms: u64) -> bool {
        self.hold_starting
            || self
                .hold_started_at_ms
                .is_some_and(|t| now_ms.saturating_sub(t) < HOLD_SETTLE_MS)
    }

    pub fn device_changed(&mut self, now_ms: u64) {
        self.device_changed_at_ms = Some(now_ms);
    }

    pub fn device_changing(&self, now_ms: u64) -> bool {
        self.device_changed_at_ms
            .is_some_and(|t| now_ms.saturating_sub(t) < DEVICE_CHANGE_MS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bools() -> [bool; 2] {
        [false, true]
    }

    // ---- ported from PauseTests / SegmentGateTests / SystemUnmuteGuardTests ----

    #[test]
    fn pause_cuts_and_remembers_the_sounding_turn() {
        assert_eq!(
            pause_action(true, false, Some(7), None),
            Some(PauseAction {
                cut: true,
                remember: Some(7),
                ..Default::default()
            })
        );
        assert_eq!(
            pause_action(true, false, None, None),
            Some(PauseAction {
                cut: true,
                ..Default::default()
            })
        );
    }

    #[test]
    fn resume_releases_the_turn_and_reclaims_the_voice() {
        let a = pause_action(false, true, None, Some(7)).unwrap();
        assert_eq!(
            a,
            PauseAction {
                release: Some(7),
                reclaim: true,
                ..Default::default()
            }
        );
        assert!(!a.cut);
    }

    #[test]
    fn no_change_no_action() {
        assert_eq!(pause_action(true, true, Some(7), Some(7)), None);
        assert_eq!(pause_action(false, false, Some(7), None), None);
    }

    #[test]
    fn wants_active_truth_table() {
        for live in bools() {
            for joined in bools() {
                for ambient in bools() {
                    assert_eq!(
                        wants_active(live, joined, ambient),
                        live && joined && !ambient
                    );
                }
            }
        }
        assert!(wants_active(true, true, false));
        assert!(!wants_active(true, true, true));
    }

    #[test]
    fn admits_truth_table() {
        let mut admitted = 0;
        for active in bools() {
            for paused in bools() {
                for speaking in bools() {
                    for other in bools() {
                        let want = active && !paused && !speaking && !other;
                        assert_eq!(admits(active, paused, speaking, other), want);
                        admitted += want as u32;
                    }
                }
            }
        }
        assert_eq!(admitted, 1);
    }

    #[test]
    fn admits_segment_truth_table() {
        let mut admitted = 0;
        for start in bools() {
            for active in bools() {
                for paused in bools() {
                    for speaking in bools() {
                        for other in bools() {
                            let want = active && !paused && (start || (!speaking && !other));
                            assert_eq!(
                                admits_segment(start, active, paused, speaking, other),
                                want,
                                "start {start} active {active} paused {paused} speaking {speaking} other {other}"
                            );
                            admitted += want as u32;
                        }
                    }
                }
            }
        }
        // active && !paused: start=true -> 4 cells, start=false -> 1 cell.
        assert_eq!(admitted, 5);
    }

    #[test]
    fn utterance_started_open_survives_a_reply_or_hold_at_its_end() {
        assert!(admits_segment(true, true, false, true, false));
        assert!(admits_segment(true, true, false, false, true));
        assert!(!admits_segment(false, true, false, true, false));
        assert!(!admits_segment(false, true, false, false, true));
        assert!(!admits_segment(true, false, false, false, false));
        assert!(!admits_segment(true, true, true, false, false));
    }

    #[test]
    fn hold_is_short() {
        const { assert!(OTHER_VOICE_HOLD_MS <= 1000) };
    }

    #[test]
    fn muted_speaker_skips_speaking_and_counts_cut_as_played() {
        assert!(!should_speak(true));
        assert!(should_speak(false));
        assert!(counts_as_played(false, true));
        assert!(!counts_as_played(false, false));
        assert!(counts_as_played(true, false));
    }

    #[test]
    fn system_unmute_guard_cases() {
        let s = system_unmute_swallows;
        // hold start swallowed; settled real unmute passes
        assert!(s(false, true, false, false, true, false));
        assert!(!s(false, true, false, false, false, false));
        // existing cases
        assert!(s(false, true, true, false, false, false));
        assert!(s(false, true, false, true, false, false));
        assert!(!s(true, true, true, true, true, false));
        assert!(!s(false, false, true, true, true, false));
        // device change: either direction that differs from the app state
        assert!(s(false, true, false, false, false, true));
        assert!(s(true, false, false, false, false, true));
        assert!(!s(false, false, false, false, false, true));
        assert!(!s(false, true, false, false, false, false));
    }

    // ---- other voice ----

    #[test]
    fn other_voice_holds_for_a_second_after_it_ends() {
        let (v, until) = other_voice(true, 0, 5000);
        assert!(v);
        assert_eq!(until, 6000);
        assert_eq!(other_voice(false, until, 5500), (true, 6000));
        assert_eq!(other_voice(false, until, 6000), (false, 6000));
    }

    // ---- mode derivation ----

    #[test]
    fn mode_table() {
        let mut s = LiveState::new();
        assert_eq!(s.mode(), Mode::Idle);
        s.live = true;
        assert_eq!(s.mode(), Mode::Idle); // not joined
        s.joined = true;
        assert_eq!(s.mode(), Mode::Live);
        s.muted = true;
        assert_eq!(s.mode(), Mode::Muted);
        s.ambient = true;
        assert_eq!(s.mode(), Mode::Ambient);
        s.live = false;
        assert_eq!(s.mode(), Mode::Idle);
    }

    // ---- state machine scenarios ----

    fn joined() -> LiveState {
        let mut s = LiveState::new();
        s.bridge_state(true, true, false, false);
        s
    }

    #[test]
    fn join_starts_the_pipeline() {
        let mut s = LiveState::new();
        let (cut, c) = s.bridge_state(true, true, false, false);
        assert!(!cut);
        assert!(c.started && !c.stopped && c.pause.is_none());
        assert!(s.active);
        assert_eq!(s.mode(), Mode::Live);
    }

    #[test]
    fn pause_then_resume_replays_the_remembered_turn() {
        let mut s = joined();
        s.speaking_turn = Some(9);
        s.vad_speaking = true;
        let (_, c) = s.bridge_state(true, true, true, false);
        assert_eq!(
            c.pause,
            Some(PauseAction {
                cut: true,
                remember: Some(9),
                ..Default::default()
            })
        );
        assert!(s.active, "pause never releases the pipeline");
        assert_eq!(s.speaking_turn, None);
        assert!(!s.vad_speaking);
        assert_eq!(s.paused_turn, Some(9));
        assert!(!s.can_admit(false));
        // a second identical report does nothing
        let (_, c) = s.bridge_state(true, true, true, false);
        assert_eq!(c.pause, None);
        // resume
        let (_, c) = s.bridge_state(true, true, false, false);
        let a = c.pause.unwrap();
        assert_eq!((a.release, a.reclaim, a.cut), (Some(9), true, false));
        assert_eq!(s.paused_turn, None);
        assert!(s.can_admit(false));
    }

    #[test]
    fn a_join_that_arrives_paused_cuts_nothing() {
        let mut s = LiveState::new();
        let (_, c) = s.bridge_state(true, true, true, false);
        assert!(c.started);
        assert_eq!(c.pause, None);
        assert!(s.pause_applied);
        // resume then undoes it
        let (_, c) = s.bridge_state(true, true, false, false);
        assert_eq!(
            c.pause,
            Some(PauseAction {
                reclaim: true,
                ..Default::default()
            })
        );
    }

    #[test]
    fn pause_never_feeds_wants_active() {
        let mut s = joined();
        s.bridge_state(true, true, true, false);
        assert_eq!(s.mode(), Mode::Live);
        assert!(wants_active(s.live, s.joined, s.ambient));
    }

    #[test]
    fn ambient_round_trip_holds_page_ambient_until_the_claim_lands() {
        let mut s = joined();
        assert_eq!(s.set_ambient(true), AmbientChange::EnterAfterSending);
        assert_eq!(s.mode(), Mode::Ambient);
        assert!(s.page_ambient());
        // the caller finishes sending, then reconciles capture
        let c = s.apply_capture();
        assert!(c.stopped);
        assert!(!s.active);
        // leaving
        match s.set_ambient(false) {
            AmbientChange::Applied(c) => assert!(c.started),
            other => panic!("{other:?}"),
        }
        assert!(!s.ambient);
        assert!(
            s.leaving_ambient && s.page_ambient(),
            "still ambient to the page"
        );
        assert!(s.claim_wanted());
        assert!(s.finish_leaving_ambient());
        assert!(!s.page_ambient());
        assert!(!s.finish_leaving_ambient());
    }

    #[test]
    fn leaving_ambient_is_finished_by_a_failed_or_paused_claim() {
        let mut s = joined();
        s.set_ambient(true);
        s.apply_capture();
        s.set_ambient(false);
        s.paused = true;
        assert!(!s.claim_wanted());
        assert!(!s.page_ambient());
    }

    #[test]
    fn set_ambient_is_ignored_without_a_live_session_or_a_change() {
        let mut s = LiveState::new();
        assert_eq!(s.set_ambient(true), AmbientChange::Ignored);
        let mut s = joined();
        assert_eq!(s.set_ambient(false), AmbientChange::Ignored);
    }

    #[test]
    fn live_dropping_clears_ambient() {
        let mut s = joined();
        s.set_ambient(true);
        s.apply_capture();
        let (_, c) = s.bridge_state(false, false, false, false);
        assert!(!s.ambient && !s.leaving_ambient);
        assert!(!c.stopped, "already inactive");
        assert_eq!(s.mode(), Mode::Idle);
        // leaving state cleared too
        let mut s = joined();
        s.set_ambient(true);
        s.apply_capture();
        s.set_ambient(false);
        s.bridge_state(false, false, false, false);
        assert!(!s.page_ambient());
    }

    #[test]
    fn leaving_the_session_resets_mute() {
        let mut s = joined();
        s.set_muted(true, false);
        assert_eq!(s.mode(), Mode::Muted);
        let (_, c) = s.bridge_state(true, false, false, false);
        assert!(c.stopped);
        assert!(!s.muted, "a rejoin must not be silently muted");
    }

    #[test]
    fn speaker_mute_cut_survives_a_quick_unmute() {
        let mut s = joined();
        s.speaking_turn = Some(4);
        assert!(s.set_speech_muted(true));
        assert_eq!(s.speech_cut_turn, Some(4));
        assert!(!s.set_speech_muted(false)); // unmuting needs nothing
        assert!(counts_as_played(false, s.speech_cut_turn == Some(4)));
        // no change, no cut
        assert!(!s.set_speech_muted(false));
        // muting with nothing sounding cuts nothing
        s.speaking_turn = None;
        assert!(!s.set_speech_muted(true));
    }

    #[test]
    fn mute_flows_and_system_unmute_deferral() {
        let mut s = joined();
        s.system_mute_armed = true;
        let c = s.set_muted(true, false);
        assert!(c.cut_and_flush && c.system_mute && !c.claim);
        // unmute: deferred until a frame
        let c = s.set_muted(false, true);
        assert!(c.system_unmute_deferred && c.claim && c.arm_fallback.is_none());
        assert!(s.system_unmute_pending && !s.frames_expected);
        assert!(
            !s.first_frame(),
            "levels before capture is known running are stale"
        );
        assert_eq!(
            s.capture_started_after_unmute(),
            Some(s.unmute_fallback_generation)
        );
        assert!(s.first_frame());
        assert!(!s.system_unmute_pending && !s.first_frame());
    }

    #[test]
    fn unmute_while_a_turn_speaks_arms_the_fallback_at_once() {
        let mut s = joined();
        s.system_mute_armed = true;
        s.set_muted(true, false);
        s.speaking_turn = Some(1);
        let c = s.set_muted(false, true);
        let g = c.arm_fallback.expect("capture is already running");
        assert!(s.frames_expected);
        assert!(s.fallback_fires(g));
        assert!(!s.system_unmute_pending);
    }

    #[test]
    fn a_later_mute_supersedes_a_pending_fallback() {
        let mut s = joined();
        s.system_mute_armed = true;
        s.set_muted(true, false);
        s.speaking_turn = Some(1);
        let g = s.set_muted(false, true).arm_fallback.unwrap();
        s.set_muted(true, false);
        assert!(!s.fallback_fires(g));
        // a re-unmute bumps the generation too, so the old timer is stale
        s.speaking_turn = None;
        let g2 = s.set_muted(false, true);
        assert!(g2.arm_fallback.is_none());
        assert!(!s.fallback_fires(g));
    }

    #[test]
    fn unmute_claims_only_when_active_and_not_paused() {
        let mut s = joined();
        s.set_muted(true, false);
        s.paused = true;
        assert!(!s.set_muted(false, false).claim);
        let mut s = LiveState::new();
        s.live = true;
        s.joined = false;
        s.muted = true;
        assert!(!s.set_muted(false, false).claim);
    }

    #[test]
    fn setting_the_same_mute_is_a_no_op() {
        let mut s = joined();
        let g = s.unmute_fallback_generation;
        assert!(s.set_muted(false, false).unchanged);
        assert_eq!(s.unmute_fallback_generation, g);
    }

    #[test]
    fn hold_start_window_is_one_second() {
        let mut s = joined();
        assert!(!s.hold_settling(0));
        s.begin_hold_start();
        assert!(s.hold_settling(10_000));
        s.end_hold_start(10_000);
        assert!(s.hold_settling(10_999));
        assert!(!s.hold_settling(11_000));
    }

    #[test]
    fn device_change_window_is_two_seconds() {
        let mut s = joined();
        assert!(!s.device_changing(0));
        s.device_changed(5_000);
        assert!(s.device_changing(6_999));
        assert!(!s.device_changing(7_000));
    }

    #[test]
    fn speech_start_records_the_gate_and_survives_a_reply() {
        let mut s = joined();
        s.vad_changed(true, false);
        assert!(s.speech_started_open);
        // a reply starts mid-utterance: the segment is still admitted
        s.speaking_turn = Some(2);
        assert!(s.segment_goes_to_guard(false));
        s.speaking_turn = None;
        s.vad_changed(false, false);
        // speech that starts inside a closed gate is dropped
        s.speaking_turn = Some(2);
        s.vad_changed(true, false);
        assert!(!s.speech_started_open);
        assert!(!s.segment_goes_to_guard(false));
        // another voice at the start closes the gate as well
        s.speaking_turn = None;
        s.vad_changed(false, false);
        s.vad_changed(true, true);
        assert!(!s.speech_started_open);
        // muted mic: the segment never goes to the guard
        let mut m = joined();
        m.muted = true;
        assert!(m.segment_admitted(false));
        assert!(!m.segment_goes_to_guard(false));
    }

    #[test]
    fn segment_fate_needs_the_gate_and_the_guard() {
        let s = joined();
        assert_eq!(s.segment_fate(false, true), SegmentFate::Admit);
        // another voice's segment is rejected by the guard
        assert_eq!(s.segment_fate(false, false), SegmentFate::Drop);
        // a closed gate drops it even when the guard would admit
        let mut m = joined();
        m.muted = true;
        assert_eq!(m.segment_fate(false, true), SegmentFate::Drop);
        let mut p = joined();
        p.paused = true;
        assert_eq!(p.segment_fate(false, true), SegmentFate::Drop);
    }

    #[test]
    fn a_rising_edge_only_records_once() {
        let mut s = joined();
        s.vad_changed(true, false);
        s.speaking_turn = Some(1);
        s.vad_changed(true, false); // still speaking: not a new start
        assert!(s.speech_started_open);
    }

    #[test]
    fn mic_open_for_the_idle_clock() {
        let mut s = joined();
        assert!(s.mic_open(false));
        assert!(!s.mic_open(true));
        s.muted = true;
        assert!(!s.mic_open(false));
        s.muted = false;
        s.paused = true;
        assert!(!s.mic_open(false));
        s.paused = false;
        s.speaking_turn = Some(1);
        assert!(!s.mic_open(false));
    }
}
