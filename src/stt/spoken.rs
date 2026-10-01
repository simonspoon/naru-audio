//! Spoken form of a transcript's words (naru task 1569, part 3): a verbatim
//! reference transcript for voice cloning must read the way the audio
//! sounds, so "42" becomes "forty-two" and "v2.1" "version two point one".
//! Pure text, no model: [`annotate`] turns each [`Word`] into a
//! [`SpokenWord`] with its `spoken` text, whether that `differs` from the
//! written one, and whether it is a `filler`.
//!
//! Rules, applied to a word's core (leading `"'“‘([{` and trailing
//! `.,;:!?"'”’)]}…` stay outside the conversion and are kept as written):
//!
//! - `&` `+` `@` `%` `$` alone: "and" "plus" "at" "percent" "dollars".
//! - `@name`: "at name". `+5`: "plus five". `3+`: "three plus". `-5`:
//!   "minus five".
//! - `$5`: "five dollars" ("$1" "one dollar"); `$2.50`: "two dollars and
//!   fifty cents". `50%`: "fifty percent".
//! - Ordinals `3rd` `21st` `100th`: "third" "twenty-first" "one hundredth".
//! - Ranges `10-20`: "ten to twenty" (each end read like any integer), only
//!   when both ends have at most 4 digits, the left is below the right, and
//!   it is not phone-like (3 digits, then 4): "555-1234" and "24-7" stay.
//! - Integers, with or without `,` grouping: cardinal, no "and"
//!   ("1,000" "one thousand"; "123" "one hundred twenty-three"). More than
//!   15 digits, or a leading zero ("007"), is read digit by digit.
//! - A bare 4-digit integer 1100..=2099 is a year, as people say years
//!   ("1999" "nineteen ninety-nine", "1905" "nineteen oh five", "1900"
//!   "nineteen hundred", "2025" "twenty twenty-five"); 2000..=2009 stay
//!   "two thousand [five]". This is a guess: "1500" the quantity is read
//!   "fifteen hundred", which is also how it is spoken as often as not.
//! - Decimals `2.5`: "two point five"; the fraction is read digit by digit
//!   with 0 as "zero" ("2.05" "two point zero five").
//! - Versions: three or more dotted parts (`3.0.1`), or two or more after a
//!   `v` (`v2.1`; bare "v8" stays), are read part by part joined by "point",
//!   a part of exactly "0" being "oh" ("3.0.1" "three point oh point one"),
//!   and the `v` forms are prefixed "version" ("v2.1" "version two point
//!   one"). A multi-digit part is a cardinal, one with a leading zero is
//!   read digit by digit.
//!
//! Whisper's word timing cuts a decimal at its point ("$2.50" arrives as
//! "$2" ".50,"): [`annotate`] first joins a word of `.` and digits onto a
//! previous word that ends in a digit, keeping the first's start.
//!
//! Anything else (including "AT&T", "C++", "3D", "3:30") is left as written.

use super::Word;

/// A [`Word`] with its spoken form, for the verbatim transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct SpokenWord {
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub spoken: String,
    /// `spoken` is not the same words as `text`, ignoring case and
    /// punctuation.
    pub differs: bool,
    /// An um/uh/er/ah/hmm/mm-type filler.
    pub filler: bool,
}

pub fn annotate(words: &[Word]) -> Vec<SpokenWord> {
    let mut joined: Vec<Word> = Vec::with_capacity(words.len());
    for w in words {
        let fraction = w
            .text
            .strip_prefix('.')
            .is_some_and(|rest| rest.starts_with(|c: char| c.is_ascii_digit()));
        match joined.last_mut() {
            Some(prev) if fraction && prev.text.ends_with(|c: char| c.is_ascii_digit()) => {
                prev.text.push_str(&w.text);
                prev.end = w.end;
            }
            _ => joined.push(w.clone()),
        }
    }
    joined
        .iter()
        .map(|w| {
            let spoken = spoken_form(&w.text);
            SpokenWord {
                start: w.start,
                end: w.end,
                differs: letters(&spoken) != letters(&w.text),
                filler: is_filler(&w.text),
                text: w.text.clone(),
                spoken,
            }
        })
        .collect()
}

/// `text` as it is spoken, its leading and trailing punctuation kept.
pub fn spoken_form(text: &str) -> String {
    let lead_end = text.len() - text.trim_start_matches(is_lead).len();
    let rest = &text[lead_end..];
    let core = rest.trim_end_matches(is_trail);
    let (lead, trail) = (&text[..lead_end], &rest[core.len()..]);
    match core_spoken(core) {
        Some(spoken) => format!("{lead}{spoken}{trail}"),
        None => text.to_string(),
    }
}

pub fn is_filler(text: &str) -> bool {
    matches!(
        letters(text).as_str(),
        "um" | "umm"
            | "uh"
            | "uhh"
            | "uhm"
            | "er"
            | "err"
            | "erm"
            | "ah"
            | "ahh"
            | "hmm"
            | "hmmm"
            | "mm"
            | "mmm"
            | "mhm"
            | "mmhmm"
            | "uhhuh"
    )
}

fn is_lead(c: char) -> bool {
    "\"'“‘([{".contains(c)
}

fn is_trail(c: char) -> bool {
    ".,;:!?\"'”’)]}…".contains(c)
}

/// Lowercase alphanumerics only: what "differs" and "filler" compare.
fn letters(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn digits(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `None` when `core` is already how it is spoken.
fn core_spoken(core: &str) -> Option<String> {
    match core {
        "&" => return Some("and".into()),
        "+" => return Some("plus".into()),
        "@" => return Some("at".into()),
        "%" => return Some("percent".into()),
        "$" => return Some("dollars".into()),
        _ => {}
    }
    if let Some(rest) = core.strip_prefix('-')
        && rest.starts_with(|c: char| c.is_ascii_digit())
    {
        return Some(format!("minus {}", signed_rest(rest)));
    }
    if let Some(rest) = core.strip_prefix('$')
        && rest.starts_with(|c: char| c.is_ascii_digit())
    {
        return dollars(rest);
    }
    if let Some(rest) = core.strip_prefix('@')
        && !rest.is_empty()
    {
        return Some(format!("at {rest}"));
    }
    if let Some(rest) = core.strip_prefix('+')
        && rest.starts_with(|c: char| c.is_ascii_digit())
    {
        return Some(format!("plus {}", signed_rest(rest)));
    }
    for (suffix, word) in [('%', "percent"), ('+', "plus")] {
        if let Some(num) = core.strip_suffix(suffix)
            && num.starts_with(|c: char| c.is_ascii_digit())
            && let Some(spoken) = core_spoken(num).or_else(|| digits(num).then(|| num.into()))
        {
            return Some(format!("{spoken} {word}"));
        }
    }
    if let Some(spoken) = ordinal(core) {
        return Some(spoken);
    }
    if let Some((a, b)) = core.split_once('-')
        && digits(a)
        && digits(b)
        && a.len() <= 4
        && b.len() <= 4
        && !(a.len() == 3 && b.len() == 4)
        && a.parse::<u64>().ok() < b.parse::<u64>().ok()
        && let (Some(a), Some(b)) = (integer(a, true), integer(b, true))
    {
        return Some(format!("{a} to {b}"));
    }
    if let Some(spoken) = version(core) {
        return Some(spoken);
    }
    if let Some((whole, frac)) = core.split_once('.')
        && let Some(whole) = integer(whole, false)
        && digits(frac)
    {
        return Some(format!("{whole} point {}", digit_words(frac)));
    }
    integer(core, true)
}

/// What follows a `-` or `+` sign: a plain integer is a quantity, never a
/// year ("-1990" "minus one thousand nine hundred ninety").
fn signed_rest(rest: &str) -> String {
    if digits(rest) {
        integer(rest, false)
    } else {
        core_spoken(rest)
    }
    .unwrap_or_else(|| rest.into())
}

/// `5` "five dollars", `2.50` "two dollars and fifty cents".
fn dollars(amount: &str) -> Option<String> {
    let (whole, cents) = match amount.split_once('.') {
        Some((w, c)) if c.len() == 2 && digits(c) => (w, Some(c)),
        Some(_) => return None,
        None => (amount, None),
    };
    let n = whole_number(whole)?;
    let mut out = format!(
        "{} {}",
        cardinal(n),
        if n == 1 { "dollar" } else { "dollars" }
    );
    match cents.map(|c| c.parse::<u64>().unwrap()) {
        Some(c) if c > 0 => {
            out += &format!(
                " and {} {}",
                cardinal(c),
                if c == 1 { "cent" } else { "cents" }
            )
        }
        _ => {}
    }
    Some(out)
}

/// A number written with digits, optionally `,`-grouped by threes.
fn whole_number(s: &str) -> Option<u64> {
    if digits(s) {
        return (s.len() <= 15).then(|| s.parse().unwrap());
    }
    let mut groups = s.split(',');
    let first = groups.next()?;
    if !(1..=3).contains(&first.len()) || !digits(first) {
        return None;
    }
    let mut n: u64 = first.parse().ok()?;
    let mut count = 0;
    for g in groups {
        if g.len() != 3 || !digits(g) {
            return None;
        }
        n = n.checked_mul(1000)?.checked_add(g.parse().ok()?)?;
        count += 1;
    }
    (count > 0 && count <= 5).then_some(n)
}

/// A whole number as words; `year` reads a bare 1100..=2099 as a year.
fn integer(s: &str, year: bool) -> Option<String> {
    if digits(s) && (s.len() > 15 || (s.len() > 1 && s.starts_with('0'))) {
        return Some(digit_words(s));
    }
    let n = whole_number(s)?;
    if year && digits(s) && s.len() == 4 && (1100..=2099).contains(&n) && !(2000..2010).contains(&n)
    {
        let (hi, lo) = (n / 100, n % 100);
        return Some(match lo {
            0 => format!("{} hundred", cardinal(hi)),
            1..=9 => format!("{} oh {}", cardinal(hi), cardinal(lo)),
            _ => format!("{} {}", cardinal(hi), cardinal(lo)),
        });
    }
    Some(cardinal(n))
}

fn ordinal(core: &str) -> Option<String> {
    let split = core.len().checked_sub(2)?;
    if !core.is_char_boundary(split) {
        return None;
    }
    let (num, suffix) = core.split_at(split);
    if !["st", "nd", "rd", "th"].contains(&suffix.to_ascii_lowercase().as_str()) {
        return None;
    }
    let words = integer(num, false).filter(|_| digits(num))?;
    let cut = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(cut);
    let last = match last {
        "one" => "first".to_string(),
        "two" => "second".to_string(),
        "three" => "third".to_string(),
        "five" => "fifth".to_string(),
        "eight" => "eighth".to_string(),
        "nine" => "ninth".to_string(),
        "twelve" => "twelfth".to_string(),
        l if l.ends_with('y') => format!("{}ieth", &l[..l.len() - 1]),
        l => format!("{l}th"),
    };
    Some(format!("{head}{last}"))
}

fn version(core: &str) -> Option<String> {
    let (prefixed, nums) = match core.strip_prefix(['v', 'V']) {
        Some(rest) => (true, rest),
        None => (false, core),
    };
    let parts: Vec<&str> = nums.split('.').collect();
    if !parts.iter().all(|p| digits(p)) || parts.len() < if prefixed { 2 } else { 3 } {
        return None;
    }
    let spoken: Vec<String> = parts
        .iter()
        .map(|p| match *p {
            "0" => "oh".to_string(),
            p => integer(p, false).unwrap_or_else(|| digit_words(p)),
        })
        .collect();
    let spoken = spoken.join(" point ");
    Some(if prefixed {
        format!("version {spoken}")
    } else {
        spoken
    })
}

fn digit_words(s: &str) -> String {
    const NAMES: [&str; 10] = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ];
    s.bytes()
        .map(|b| NAMES[(b - b'0') as usize])
        .collect::<Vec<_>>()
        .join(" ")
}

fn cardinal(n: u64) -> String {
    const SMALL: [&str; 20] = [
        "zero",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
        "twelve",
        "thirteen",
        "fourteen",
        "fifteen",
        "sixteen",
        "seventeen",
        "eighteen",
        "nineteen",
    ];
    const TENS: [&str; 8] = [
        "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
    ];
    const SCALES: [(u64, &str); 5] = [
        (1_000_000_000_000, "trillion"),
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
        (100, "hundred"),
    ];
    if n < 20 {
        return SMALL[n as usize].to_string();
    }
    if n < 100 {
        let tens = TENS[(n / 10 - 2) as usize];
        return match n % 10 {
            0 => tens.to_string(),
            u => format!("{tens}-{}", SMALL[u as usize]),
        };
    }
    for (size, name) in SCALES {
        if n >= size {
            let head = format!("{} {name}", cardinal(n / size));
            return match n % size {
                0 => head,
                rest => format!("{head} {}", cardinal(rest)),
            };
        }
    }
    unreachable!()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str) -> Word {
        Word {
            start: 0.0,
            end: 1.0,
            text: text.to_string(),
        }
    }

    fn one(text: &str) -> SpokenWord {
        annotate(&[word(text)]).remove(0)
    }

    #[test]
    fn spoken_forms() {
        for (written, spoken) in [
            ("42", "forty-two"),
            ("0", "zero"),
            ("100", "one hundred"),
            ("123", "one hundred twenty-three"),
            ("1,000", "one thousand"),
            (
                "1,234,567",
                "one million two hundred thirty-four thousand five hundred sixty-seven",
            ),
            ("12345", "twelve thousand three hundred forty-five"),
            ("007", "zero zero seven"),
            ("2.5", "two point five"),
            ("3.14", "three point one four"),
            ("2.05", "two point zero five"),
            ("1999", "nineteen ninety-nine"),
            ("2025", "twenty twenty-five"),
            ("1905", "nineteen oh five"),
            ("1900", "nineteen hundred"),
            ("2000", "two thousand"),
            ("2005", "two thousand five"),
            ("2010", "twenty ten"),
            ("2100", "two thousand one hundred"),
            ("999", "nine hundred ninety-nine"),
            ("v2.1", "version two point one"),
            ("3.0.1", "three point oh point one"),
            ("v10.04.2", "version ten point zero four point two"),
            ("3rd", "third"),
            ("1st", "first"),
            ("2nd", "second"),
            ("12th", "twelfth"),
            ("21st", "twenty-first"),
            ("20th", "twentieth"),
            ("100th", "one hundredth"),
            ("50%", "fifty percent"),
            ("2.5%", "two point five percent"),
            ("$5", "five dollars"),
            ("$1", "one dollar"),
            ("$2.50", "two dollars and fifty cents"),
            ("$1,000", "one thousand dollars"),
            ("&", "and"),
            ("+", "plus"),
            ("@", "at"),
            ("%", "percent"),
            ("@simon", "at simon"),
            ("+5", "plus five"),
            ("3+", "three plus"),
            ("-5", "minus five"),
            ("-1990", "minus one thousand nine hundred ninety"),
            ("+1999", "plus one thousand nine hundred ninety-nine"),
            ("10-20", "ten to twenty"),
            ("1999-2005", "nineteen ninety-nine to two thousand five"),
        ] {
            assert_eq!(spoken_form(written), spoken, "{written}");
        }
    }

    #[test]
    fn punctuation_stays_outside_the_conversion() {
        assert_eq!(spoken_form("42."), "forty-two.");
        assert_eq!(spoken_form("(1999),"), "(nineteen ninety-nine),");
        assert_eq!(spoken_form("\"3rd!\""), "\"third!\"");
        assert_eq!(spoken_form("2.5."), "two point five.");
        assert_eq!(spoken_form("$5?"), "five dollars?");
        assert_eq!(spoken_form("v2.1,"), "version two point one,");
    }

    #[test]
    fn unconvertible_words_are_left_alone() {
        for w in [
            "hello",
            "Hello,",
            "AT&T",
            "C++",
            "3D",
            "mp3",
            "3:30",
            "well-known",
            "1,00",
            "1,2345",
            "v",
            "a.b.c",
            "$x",
            "555-1234",
            "24-7",
            "20-10",
            "10-10",
            "V8",
            "v6",
            "12345-67",
        ] {
            assert_eq!(spoken_form(w), w, "{w}");
        }
    }

    #[test]
    fn differs_ignores_case_and_punctuation() {
        assert!(!one("Hello,").differs);
        assert!(!one("well-known.").differs);
        assert!(!one("I-").differs);
        assert!(one("42.").differs);
        assert!(one("$5").differs);
        assert!(one("&").differs);
        assert!(one("3rd").differs);
        assert!(!one("3D").differs);
    }

    #[test]
    fn fillers() {
        for f in [
            "um", "Um,", "UH", "uh-huh", "Hmm...", "mm", "er", "ah", "Mm-hmm",
        ] {
            assert!(one(f).filler, "{f}");
        }
        for w in ["umbrella", "hello", "ahead", "error", "42"] {
            assert!(!one(w).filler, "{w}");
        }
    }

    #[test]
    fn a_decimal_cut_at_its_point_is_joined() {
        let w = |start: f64, text: &str| Word {
            start,
            end: start + 0.5,
            text: text.to_string(),
        };
        let words = annotate(&[
            w(0.0, "for"),
            w(1.0, "$2"),
            w(2.0, ".50,"),
            w(3.0, "version"),
            w(4.0, "2"),
            w(5.0, ".1"),
            w(6.0, "plan."),
            w(7.0, ".5"),
        ]);
        let spoken: Vec<_> = words.iter().map(|w| w.spoken.as_str()).collect();
        assert_eq!(
            spoken,
            [
                "for",
                "two dollars and fifty cents,",
                "version",
                "two point one",
                "plan.",
                ".5"
            ]
        );
        assert_eq!((words[1].start, words[1].end), (1.0, 2.5));
        assert_eq!(words[1].text, "$2.50,");
    }

    #[test]
    fn annotate_keeps_times_and_text() {
        let words = annotate(&[
            Word {
                start: 0.5,
                end: 0.9,
                text: "um,".into(),
            },
            Word {
                start: 1.0,
                end: 1.4,
                text: "42".into(),
            },
        ]);
        assert_eq!(words[0].start, 0.5);
        assert_eq!(words[0].text, "um,");
        assert_eq!(words[0].spoken, "um,");
        assert!(words[0].filler && !words[0].differs);
        assert_eq!(words[1].end, 1.4);
        assert_eq!(words[1].spoken, "forty-two");
        assert!(words[1].differs && !words[1].filler);
    }
}
